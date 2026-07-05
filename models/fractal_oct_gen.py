"""
FractalOctGen：三维分形自回归生成模型（核心模块，通用 N 层版本）

将 FractalGen (He et al., 2025) 的递归分形架构从二维图像拓展至三维占据八叉树，
并推广为「任意深度」的分层生成器：

  * 前 NUM_AR_LEVELS 个较粗层级使用自回归 Transformer（OctAR），
    逐节点生成，建模兄弟相关，决定整体轮廓（机身/机翼/尾翼）。
  * 其余更细层级使用并行占据 MLP（OccupancyMLP）级联，
    给定父层条件 + 位置一次性预测占据，把表面逐级细化到 2^(depth-1)³。

对 OCTREE_DEPTH=7：AR_DEPTHS=[2,3]（8³ 粗结构），OCC_DEPTHS=[4,5,6]（→ 64³）。

─────────────────────────────────────────────────────────────────────────────
训练（广度优先，teacher forcing，全层联合 BCE）：

  class_emb ─→ [AR depth-2] ─cond→ [AR depth-3] ─cond→ [OCC 4] ─cond→ [OCC 5] ─cond→ [OCC 6]
                 loss_d2          loss_d3          loss_d4      loss_d5      loss_d6
  每一层的父条件 = 上一层输出 cond 按 parent_idx gather。

推理（深度优先，自回归 / 级联采样）：

  AR depth-2 (固定 64) → AR depth-3 (全局自回归)
    → OCC 4 → OCC 5 → OCC 6（逐级展开 split==1 节点的 8 个子节点并预测占据）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from models.octree_ar import OctAR, OccupancyMLP, sinusoidal_pos_enc_3d
from utils.octree_utils import (
    OCTREE_DEPTH, FULL_DEPTH, AR_DEPTHS, OCC_DEPTHS, ALL_PRED_DEPTHS,
)


# ─── 默认超参数（可通过 config 覆盖）──────────────────────────────────────────
#
# 所有层级共享同一 embed_dim（= cond_dim），便于父→子条件级联。
# ar_blocks：每个自回归层的 Transformer block 数（长度 = len(AR_DEPTHS)，
#            也可给标量自动广播）。

DEFAULT_CONFIG = {
    'embed_dim': 512,
    'num_heads': 8,
    'ar_blocks': [14, 10],    # depth-2 较深（决定整体轮廓），depth-3 次之
    'occ_hidden_dim': 512,
    'occ_num_layers': 6,
    'num_classes': 1,         # 单类无条件生成
    'dropout': 0.1,
    # 占据层同层局部协调（窗口化双向注意力）
    'occ_use_attn': True,
    'occ_window': 64,         # z-order 窗口 = 8 个兄弟组
    'occ_attn_heads': 8,      # 须整除 occ_hidden_dim（512/8=64）
    'occ_attn_layers': 3,
    # 细层辅助监督：用同父节点的软概率平滑二值目标，缓解薄壳/空洞
    # Split-only fine-tune override (2026-07-04):
    # Focus on occupancy/split BCE first; restore these three weights after
    # d4-d6 structure becomes stable.
    # Original values: aux_soft_weight=0.20, field_loss_weight=0.50,
    # grad_loss_weight=0.30.
    'aux_soft_weight': 0.05,
    'aux_soft_blend': 0.35,
    'aux_soft_start_depth': 4,
    'field_loss_weight': 0.0,
    'field_loss_start_depth': 4,
    'grad_loss_weight': 0.0,
    'grad_loss_start_depth': 6,
}

# 大模型配置（显存充足时追求极致质量）
LARGE_CONFIG = {
    'embed_dim': 768,
    'num_heads': 12,
    'ar_blocks': [20, 14],
    'occ_hidden_dim': 768,
    'occ_num_layers': 8,
    'num_classes': 1,
    'dropout': 0.1,
    'occ_use_attn': True,
    'occ_window': 64,
    'occ_attn_heads': 12,     # 768/12=64
    'occ_attn_layers': 4,
    'aux_soft_weight': 0.20,
    'aux_soft_blend': 0.35,
    'aux_soft_start_depth': 4,
    'field_loss_weight': 0.50,
    'field_loss_start_depth': 4,
    'grad_loss_weight': 0.30,
    'grad_loss_start_depth': 6,
}

# 小模型配置（快速烟雾测试）
SMALL_CONFIG = {
    'embed_dim': 96,
    'num_heads': 4,
    'ar_blocks': [2, 2],
    'occ_hidden_dim': 96,
    'occ_num_layers': 2,
    'num_classes': 1,
    'dropout': 0.0,
    'occ_use_attn': True,
    'occ_window': 32,
    'occ_attn_heads': 4,      # 96/4=24
    'occ_attn_layers': 1,
    'aux_soft_weight': 0.20,
    'aux_soft_blend': 0.35,
    'aux_soft_start_depth': 4,
    'field_loss_weight': 0.50,
    'field_loss_start_depth': 4,
    'grad_loss_weight': 0.30,
    'grad_loss_start_depth': 6,
}


class FractalOctGen(nn.Module):
    """三维分形自回归生成模型（通用分层版本）。"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg
        self.field_loss_weight = float(self.cfg.get('field_loss_weight', 0.0))
        self.field_loss_start_depth = int(self.cfg.get('field_loss_start_depth', 10**9))

        self.ar_depths: List[int] = list(AR_DEPTHS)
        self.occ_depths: List[int] = list(OCC_DEPTHS)
        self.all_depths: List[int] = list(ALL_PRED_DEPTHS)
        self.full_depth = FULL_DEPTH
        self.octree_depth = OCTREE_DEPTH

        D = cfg['embed_dim']

        # 类别嵌入（单类无条件时 num_classes=1，始终用 class_id=0）
        self.class_emb = nn.Embedding(cfg['num_classes'] + 1, D)

        # ── 自回归层（OctAR）─────────────────────────────────────────────────
        ar_blocks = cfg['ar_blocks']
        if isinstance(ar_blocks, int):
            ar_blocks = [ar_blocks] * len(self.ar_depths)
        assert len(ar_blocks) == len(self.ar_depths), \
            f'ar_blocks 长度 {len(ar_blocks)} 应等于 AR 层数 {len(self.ar_depths)}'

        self.ar_modules = nn.ModuleList([
            OctAR(embed_dim=D, num_heads=cfg['num_heads'],
                  num_blocks=ar_blocks[i], cond_dim=D, dropout=cfg['dropout'])
            for i in range(len(self.ar_depths))
        ])

        # ── 占据层（OccupancyMLP 级联，含同层局部协调注意力）─────────────────
        self.occ_modules = nn.ModuleList([
            OccupancyMLP(cond_dim=D, hidden_dim=cfg['occ_hidden_dim'],
                         num_layers=cfg['occ_num_layers'],
                         use_attn=cfg.get('occ_use_attn', True),
                         window=cfg.get('occ_window', 64),
                         attn_heads=cfg.get('occ_attn_heads', 4),
                         attn_layers=cfg.get('occ_attn_layers', 2),
                         dropout=cfg['dropout'])
            for _ in self.occ_depths
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # 训练前向（广度优先，teacher forcing）
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_id = batch['class_id']
        B = class_id.shape[0]

        class_cond = self.class_emb(class_id).unsqueeze(1)   # [B, 1, D]

        losses: Dict[int, torch.Tensor] = {}
        prev_cond_full: Optional[torch.Tensor] = None        # [B, N_prev, D]
        aux_soft_weight = float(self.cfg.get('aux_soft_weight', 0.0))
        aux_soft_blend = float(self.cfg.get('aux_soft_blend', 0.0))
        aux_soft_start_depth = int(self.cfg.get('aux_soft_start_depth', 10**9))
        field_loss_weight = float(self.cfg.get('field_loss_weight', 0.0))
        field_loss_start_depth = int(self.cfg.get('field_loss_start_depth', 10**9))
        grad_loss_weight = float(self.cfg.get('grad_loss_weight', 0.0))
        grad_loss_start_depth = int(self.cfg.get('grad_loss_start_depth', 10**9))

        # ── 自回归层 ──────────────────────────────────────────────────────────
        for i, d in enumerate(self.ar_depths):
            split_d = batch[f'split_{d}']
            field_d = batch.get(f'field_{d}', split_d.float())
            xyz_d = batch[f'xyz_{d}']
            mask_d = batch[f'mask_{d}']

            if i == 0:
                parent_cond = class_cond                      # [B,1,D]，广播
            else:
                parent_cond = _gather_parent_cond(
                    prev_cond_full, batch[f'parent_idx_{d}'], mask_d)

            cond_full, logits, field_pred, grad_pred = self.ar_modules[i](
                split_labels=split_d.long(),
                xyz=xyz_d,
                parent_cond=parent_cond,
                padding_mask=mask_d,
            )
            layer_loss = _masked_bce_loss(
                logits, split_d.float(), mask_d,
                pos_weight=_auto_pos_weight(split_d, mask_d))
            if d >= field_loss_start_depth and field_loss_weight > 0:
                depth_weight = 1.0 + 0.25 * max(0, d - field_loss_start_depth)
                layer_loss = layer_loss + field_loss_weight * depth_weight * _masked_field_loss(
                    field_pred, field_d.float(), mask_d)
            grad_key = f'grad_{d}'
            if d >= grad_loss_start_depth and grad_loss_weight > 0 and grad_key in batch:
                depth_weight = 1.0 + 0.25 * max(0, d - grad_loss_start_depth)
                layer_loss = layer_loss + grad_loss_weight * depth_weight * _masked_grad_loss(
                    grad_pred, batch[grad_key], mask_d)
            losses[d] = layer_loss
            prev_cond_full = cond_full

        # ── 占据层级联 ────────────────────────────────────────────────────────
        for j, d in enumerate(self.occ_depths):
            split_d = batch[f'split_{d}']
            field_d = batch.get(f'field_{d}', split_d.float())
            xyz_d = batch[f'xyz_{d}']
            mask_d = batch[f'mask_{d}']
            parent_idx_d = batch.get(f'parent_idx_{d}')

            parent_cond = _gather_parent_cond(
                prev_cond_full, batch[f'parent_idx_{d}'], mask_d)

            logits, field_pred, grad_pred, cond_full = self.occ_modules[j](
                xyz=xyz_d, parent_cond=parent_cond, mask=mask_d)
            layer_loss = _masked_bce_loss(
                logits, split_d.float(), mask_d,
                pos_weight=_auto_pos_weight(split_d, mask_d))
            if parent_idx_d is not None and d >= aux_soft_start_depth and aux_soft_weight > 0:
                soft_targets = _sibling_smoothed_targets(
                    split_d.float(), parent_idx_d, mask_d, blend=aux_soft_blend)
                soft_loss = _masked_bce_loss(logits, soft_targets, mask_d)
                depth_weight = 1.0 + 0.25 * max(0, d - aux_soft_start_depth)
                layer_loss = layer_loss + aux_soft_weight * depth_weight * soft_loss
            if d >= field_loss_start_depth and field_loss_weight > 0:
                depth_weight = 1.0 + 0.25 * max(0, d - field_loss_start_depth)
                layer_loss = layer_loss + field_loss_weight * depth_weight * _masked_field_loss(
                    field_pred, field_d.float(), mask_d)
            grad_key = f'grad_{d}'
            if d >= grad_loss_start_depth and grad_loss_weight > 0 and grad_key in batch:
                depth_weight = 1.0 + 0.25 * max(0, d - grad_loss_start_depth)
                layer_loss = layer_loss + grad_loss_weight * depth_weight * _masked_grad_loss(
                    grad_pred, batch[grad_key], mask_d)
            losses[d] = layer_loss
            prev_cond_full = cond_full

        total = sum(losses.values())
        out = {'loss': total}
        for d, l in losses.items():
            out[f'loss_d{d}'] = l.detach()
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # 推理采样（深度优先）
    # ─────────────────────────────────────────────────────────────────────────

    # 采样安全阀：单层最多展开的父分裂节点数（防止未训练/异常模型节点数爆炸）。
    MAX_EXPAND_NODES = 24000

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        class_id: int = 0,
        temperature_l0: float = 1.0,
        temperature_l1: float = 1.0,
        temperature_l2: float = 0.0,
        device: torch.device = None,
        **kwargs,
    ) -> Dict[str, object]:
        """深度优先采样生成形状。

        温度映射：
          temperature_l0 → 第一层 AR（depth-2）
          temperature_l1 → 其余 AR 层（depth-3 ...）
          temperature_l2 → 所有占据 MLP 层（depth-4/5/6 ...）

        Returns:
            dict:
              'split': {d: [B 个 per-shape tensor]}  各深度分裂/占据标签
              'xyz':   {d: [B 个 per-shape tensor]}  各深度节点坐标
              'depths': 预测深度列表
              'finest_depth': 最细占据深度（用于体素化）
        """
        if device is None:
            device = next(self.parameters()).device
        B = batch_size

        class_ids = torch.full((B,), class_id, dtype=torch.long, device=device)
        class_cond = self.class_emb(class_ids).unsqueeze(1)   # [B,1,D]

        # ── 第一层 AR（depth = ar_depths[0]，固定 4×4×4 = 64 节点）─────────────
        d0 = self.ar_depths[0]
        assert d0 == 2, '当前实现假设第一层 AR 为 depth-2（64 节点）'
        xyz_2 = _generate_depth2_xyz(B, device)               # [B,64,3]

        generated = torch.zeros(B, 0, dtype=torch.long, device=device)
        cond_steps = []
        field_steps = []
        for step in range(64):
            sampled, cond_s, field_s, _ = self.ar_modules[0].sample_one_step(
                generated_so_far=generated,
                xyz=xyz_2,
                parent_cond=class_cond.expand(B, 64, -1),
                step=step,
                temperature=temperature_l0,
            )
            generated = torch.cat([generated, sampled.unsqueeze(1)], dim=1)
            cond_steps.append(cond_s)
            field_steps.append(field_s)
        split_2 = generated                                   # [B,64]
        cond_2 = torch.stack(cond_steps, dim=1)               # [B,64,D]
        field_2 = torch.stack(field_steps, dim=1)             # [B,64]

        # ── 逐形状级联 ────────────────────────────────────────────────────────
        out_split = {d: [] for d in self.all_depths}
        out_field = {d: [] for d in self.all_depths}
        out_xyz = {d: [] for d in self.all_depths}

        for b in range(B):
            cur_split = split_2[b]                            # [64]
            cur_xyz = xyz_2[b]                                # [64,3]
            cur_cond = cond_2[b]                              # [64,D]
            cur_field = field_2[b]                            # [64]
            cur_depth = d0
            out_split[d0].append(cur_split)
            out_field[d0].append(cur_field)
            out_xyz[d0].append(cur_xyz)

            # 其余 AR 层（全局自回归）
            for i in range(1, len(self.ar_depths)):
                d = self.ar_depths[i]
                cur_split, cur_field, cur_xyz, cur_cond = self._sample_ar_level(
                    parent_split=cur_split, parent_xyz=cur_xyz, parent_cond=cur_cond,
                    parent_depth=cur_depth, ar_module=self.ar_modules[i],
                    temperature=temperature_l1, device=device)
                cur_depth = d
                out_split[d].append(cur_split)
                out_field[d].append(cur_field)
                out_xyz[d].append(cur_xyz)

            # 占据层级联（并行 MLP）
            for j, d in enumerate(self.occ_depths):
                cur_split, cur_field, cur_xyz, cur_cond = self._sample_occ_level(
                    parent_split=cur_split, parent_xyz=cur_xyz, parent_cond=cur_cond,
                    parent_depth=cur_depth, occ_module=self.occ_modules[j],
                    temperature=temperature_l2, device=device)
                cur_depth = d
                out_split[d].append(cur_split)
                out_field[d].append(cur_field)
                out_xyz[d].append(cur_xyz)

        return {
            'split': out_split,
            'field': out_field,
            'xyz': out_xyz,
            'depths': self.all_depths,
            'finest_depth': self.occ_depths[-1] if self.occ_depths else self.ar_depths[-1],
        }

    @torch.no_grad()
    def sample_with_prefix(
        self,
        prefix: Dict[str, torch.Tensor],
        prefix_depth: int = 3,
        batch_size: int = 1,
        class_id: int = 0,
        temperature_l0: float = 1.0,
        temperature_l1: float = 1.0,
        temperature_l2: float = 0.0,
        device: torch.device = None,
        **kwargs,
    ) -> Dict[str, object]:
        """用真实粗层八叉树作为 prefix，然后从下一层继续采样。

        prefix 至少需要包含 split_d/xyz_d；当 prefix_depth >= 3 时还需要
        parent_idx_d。关键点是 prefix 层仍通过 teacher forcing 跑一遍模型，
        得到后续层需要的 cond_full，而不是只把 split/xyz 复制进输出。
        """
        if device is None:
            device = next(self.parameters()).device
        if prefix_depth not in self.all_depths:
            raise ValueError(f'prefix_depth={prefix_depth} 不在预测层 {self.all_depths} 中')

        B = batch_size
        class_ids = torch.full((B,), class_id, dtype=torch.long, device=device)
        class_cond = self.class_emb(class_ids).unsqueeze(1)

        def take(key: str, dtype=None) -> torch.Tensor:
            if key not in prefix:
                raise KeyError(f'prefix 缺少 {key}')
            x = prefix[key].to(device)
            if dtype is not None:
                x = x.to(dtype=dtype)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            elif x.dim() == 2 and key.startswith('xyz_'):
                x = x.unsqueeze(0)
            if x.shape[0] == 1 and B > 1:
                x = x.expand(B, *x.shape[1:]).contiguous()
            if x.shape[0] != B:
                raise ValueError(f'{key} batch={x.shape[0]} 与 batch_size={B} 不一致')
            return x

        prefix_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        prev_cond_full: Optional[torch.Tensor] = None

        # Prefix 中的 AR 层：用 teacher forcing 计算 cond。
        for i, d in enumerate(self.ar_depths):
            if d > prefix_depth:
                break
            split_d = take(f'split_{d}', torch.long)
            xyz_d = take(f'xyz_{d}', torch.float32)
            field_key = f'field_{d}'
            field_d = take(field_key, torch.float32) if field_key in prefix else split_d.float()
            mask_d = torch.ones(split_d.shape, dtype=torch.bool, device=device)

            if i == 0:
                parent_cond = class_cond
            else:
                parent_idx_d = take(f'parent_idx_{d}', torch.long)
                parent_cond = _gather_parent_cond(prev_cond_full, parent_idx_d, mask_d)

            cond_full, _, _, _ = self.ar_modules[i](
                split_labels=split_d,
                xyz=xyz_d,
                parent_cond=parent_cond,
                padding_mask=mask_d,
            )
            prefix_cache[d] = (split_d, field_d, xyz_d, cond_full)
            prev_cond_full = cond_full

        # 可选支持固定到 d4/d5：OCC prefix 同样前向一次得到 cond。
        for j, d in enumerate(self.occ_depths):
            if d > prefix_depth:
                break
            split_d = take(f'split_{d}', torch.long)
            xyz_d = take(f'xyz_{d}', torch.float32)
            field_key = f'field_{d}'
            field_d = take(field_key, torch.float32) if field_key in prefix else split_d.float()
            mask_d = torch.ones(split_d.shape, dtype=torch.bool, device=device)
            parent_idx_d = take(f'parent_idx_{d}', torch.long)
            parent_cond = _gather_parent_cond(prev_cond_full, parent_idx_d, mask_d)

            _, _, _, cond_full = self.occ_modules[j](
                xyz=xyz_d, parent_cond=parent_cond, mask=mask_d)
            prefix_cache[d] = (split_d, field_d, xyz_d, cond_full)
            prev_cond_full = cond_full

        if prefix_depth not in prefix_cache:
            raise ValueError(f'无法从 prefix 计算到 depth {prefix_depth}')

        out_split = {d: [] for d in self.all_depths}
        out_field = {d: [] for d in self.all_depths}
        out_xyz = {d: [] for d in self.all_depths}

        ar_module_by_depth = {d: self.ar_modules[i] for i, d in enumerate(self.ar_depths)}
        occ_module_by_depth = {d: self.occ_modules[i] for i, d in enumerate(self.occ_depths)}

        for b in range(B):
            for d in self.all_depths:
                if d > prefix_depth:
                    break
                split_d, field_d, xyz_d, _ = prefix_cache[d]
                out_split[d].append(split_d[b])
                out_field[d].append(field_d[b])
                out_xyz[d].append(xyz_d[b])

            cur_split, cur_field, cur_xyz, cur_cond = (
                item[b] for item in prefix_cache[prefix_depth]
            )
            cur_depth = prefix_depth

            for d in self.all_depths:
                if d <= prefix_depth:
                    continue
                if d in ar_module_by_depth:
                    cur_split, cur_field, cur_xyz, cur_cond = self._sample_ar_level(
                        parent_split=cur_split, parent_xyz=cur_xyz, parent_cond=cur_cond,
                        parent_depth=cur_depth, ar_module=ar_module_by_depth[d],
                        temperature=temperature_l1, device=device)
                else:
                    cur_split, cur_field, cur_xyz, cur_cond = self._sample_occ_level(
                        parent_split=cur_split, parent_xyz=cur_xyz, parent_cond=cur_cond,
                        parent_depth=cur_depth, occ_module=occ_module_by_depth[d],
                        temperature=temperature_l2, device=device)
                cur_depth = d
                out_split[d].append(cur_split)
                out_field[d].append(cur_field)
                out_xyz[d].append(cur_xyz)

        return {
            'split': out_split,
            'field': out_field,
            'xyz': out_xyz,
            'depths': self.all_depths,
            'finest_depth': self.occ_depths[-1] if self.occ_depths else self.ar_depths[-1],
        }

    # ── 单形状：AR 层全局自回归采样 ───────────────────────────────────────────
    def _sample_ar_level(
        self,
        parent_split: torch.Tensor, parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor, parent_depth: int,
        ar_module: OctAR, temperature: float, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """展开 parent_split==1 节点的 8 个子节点，沿全局 z-order 自回归采样。

        与训练 teacher forcing 的序列完全对应（消除 train-inference 不一致）。
        """
        D = ar_module.cond_dim
        split_idx = (parent_split == 1).nonzero(as_tuple=True)[0]
        if len(split_idx) == 0:
            return (torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, device=device),
                    torch.zeros(0, 3, device=device),
                    torch.zeros(0, D, device=device))
        if len(split_idx) > self.MAX_EXPAND_NODES:
            split_idx = split_idx[:self.MAX_EXPAND_NODES]

        child_xyz = []
        child_pcond = []
        for pidx in split_idx:
            ch = _compute_children_xyz(parent_xyz[pidx], depth_parent=parent_depth).to(device)
            child_xyz.append(ch)
            child_pcond.append(parent_cond[pidx].unsqueeze(0).expand(8, -1))
        xyz = torch.cat(child_xyz, dim=0)                     # [N,3]
        pcond = torch.cat(child_pcond, dim=0)                 # [N,D]
        N = xyz.shape[0]

        xyz_b = xyz.unsqueeze(0)
        pcond_b = pcond.unsqueeze(0)
        generated = torch.zeros(1, 0, dtype=torch.long, device=device)
        cond_steps = []
        field_steps = []
        for step in range(N):
            sampled, cond_s, field_s, _ = ar_module.sample_one_step(
                generated_so_far=generated, xyz=xyz_b,
                parent_cond=pcond_b, step=step, temperature=temperature)
            generated = torch.cat([generated, sampled.unsqueeze(1)], dim=1)
            cond_steps.append(cond_s[0])
            field_steps.append(field_s[0])
        split = generated[0]                                  # [N]
        cond = torch.stack(cond_steps, dim=0)                 # [N,D]
        field = torch.stack(field_steps, dim=0)               # [N]
        return split, field, xyz, cond

    # ── 单形状：占据层并行采样 ────────────────────────────────────────────────
    def _sample_occ_level(
        self,
        parent_split: torch.Tensor, parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor, parent_depth: int,
        occ_module: OccupancyMLP, temperature: float, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """展开 parent_split==1 节点的 8 个子节点，MLP 并行预测占据。"""
        D = occ_module.cond_dim
        split_idx = (parent_split == 1).nonzero(as_tuple=True)[0]
        if len(split_idx) == 0:
            return (torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, device=device),
                    torch.zeros(0, 3, device=device),
                    torch.zeros(0, D, device=device))
        if len(split_idx) > self.MAX_EXPAND_NODES:
            split_idx = split_idx[:self.MAX_EXPAND_NODES]

        parent_xyz_s = parent_xyz[split_idx]                  # [n,3]
        parent_cond_s = parent_cond[split_idx]                # [n,D]
        n = len(split_idx)

        child_xyz = []
        for i in range(n):
            ch = _compute_children_xyz(parent_xyz_s[i], depth_parent=parent_depth).to(device)
            child_xyz.append(ch)
        xyz = torch.cat(child_xyz, dim=0)                     # [n*8,3]
        pcond = parent_cond_s.repeat_interleave(8, dim=0)     # [n*8,D]

        logits, field, _, cond = occ_module(xyz=xyz.unsqueeze(0), parent_cond=pcond.unsqueeze(0))
        logits = logits[0]                                    # [n*8]
        field = field[0]                                      # [n*8]
        cond = cond[0]                                        # [n*8,D]

        if temperature == 0.0:
            split = (logits > 0).long()
        else:
            prob = torch.sigmoid(logits / temperature)
            split = torch.bernoulli(prob).long()
        return split, field, xyz, cond


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────

def _auto_pos_weight(targets: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    """从当前 batch 的有效目标动态估计 BCE 正样本权重 = neg/pos（clamp 到 [0.5,10]）。

    避免硬编码各层正样本率，自适应不同深度/数据集的类别不均衡。
    """
    with torch.no_grad():
        t = targets[mask]
        if t.numel() == 0:
            return None
        pos = (t == 1).sum().float()
        neg = (t == 0).sum().float()
        if pos < 1:
            return None
        w = (neg / pos).clamp(0.5, 10.0)
    return w.detach()


def _masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                     mask: torch.Tensor,
                     pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """仅在有效节点（mask=True）上计算 BCE 损失。"""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    pw = pos_weight.to(logits.device) if pos_weight is not None else None
    return F.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], pos_weight=pw, reduction='mean'
    )


def _masked_field_loss(pred: torch.Tensor, targets: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """只在有效节点上回归连续场。"""
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], targets[mask], reduction='mean')


def _masked_grad_loss(pred_grad: torch.Tensor, target_grad: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """只在有效节点上回归 SDF 空间梯度（法向）。"""
    if mask.sum() == 0:
        return pred_grad.sum() * 0.0
    return F.mse_loss(pred_grad[mask], target_grad[mask], reduction='mean')


def _gather_parent_cond(
    cond: torch.Tensor,
    parent_idx: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """根据父节点索引从父层 conditioning 矩阵中 gather 对应行。

    Args:
        cond:       [B, N_parent, D]
        parent_idx: [B, N_child]
        mask:       [B, N_child] bool
    Returns:
        [B, N_child, D]
    """
    B, N_child = parent_idx.shape
    D = cond.shape[-1]
    idx = parent_idx.clamp(0, cond.shape[1] - 1)
    gathered = torch.gather(cond, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D))
    gathered = gathered * mask.unsqueeze(-1).to(gathered.dtype)
    return gathered


def _sibling_smoothed_targets(
    targets: torch.Tensor,
    parent_idx: torch.Tensor,
    mask: torch.Tensor,
    blend: float = 0.35,
) -> torch.Tensor:
    """把二值目标混合为“硬标签 + 同父节点平均概率”的软目标。

    这样做能给细层提供连续概率监督，缓解薄壳过窄、局部孔洞和抖动。
    """
    blend = float(max(0.0, min(1.0, blend)))
    soft = targets.float().clone()
    if parent_idx is None or mask.sum() == 0:
        return soft

    B, _ = parent_idx.shape
    for b in range(B):
        valid = mask[b]
        if not torch.any(valid):
            continue

        idx = parent_idx[b, valid].long()
        vals = targets[b, valid].float()
        if idx.numel() == 0:
            continue

        n_groups = int(idx.max().item()) + 1
        group_sum = torch.zeros(n_groups, device=targets.device, dtype=vals.dtype)
        group_cnt = torch.zeros(n_groups, device=targets.device, dtype=vals.dtype)
        group_sum.index_add_(0, idx, vals)
        group_cnt.index_add_(0, idx, torch.ones_like(vals))
        sibling_mean = group_sum / group_cnt.clamp(min=1.0)
        soft[b, valid] = (1.0 - blend) * vals + blend * sibling_mean[idx]

    return soft


def _generate_depth2_xyz(batch_size: int, device: torch.device) -> torch.Tensor:
    """生成 depth-2 的 64 个固定节点坐标（4×4×4 网格，z-order，归一化 [0,1]）。"""
    coords = []
    for key in range(64):
        x = _morton_decode_axis(key, 0)
        y = _morton_decode_axis(key, 1)
        z = _morton_decode_axis(key, 2)
        coords.append([x, y, z])
    xyz = torch.tensor(coords, dtype=torch.float32, device=device) / 3.0
    return xyz.unsqueeze(0).expand(batch_size, -1, -1)


def _compute_children_xyz(parent_xyz: torch.Tensor, depth_parent: int) -> torch.Tensor:
    """计算父节点的 8 个子节点归一化坐标（depth_parent → depth_parent+1）。

    与训练 _keys_to_xyz 一致：
        child_int = 2 * parent_int + {0,1}^3
        child_xyz = child_int / (2^(depth_parent+1) - 1)
    """
    max_parent = float(2 ** depth_parent - 1)
    max_child = float(2 ** (depth_parent + 1) - 1)
    parent_int = parent_xyz.float().cpu() * max_parent
    offsets = torch.tensor([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], dtype=torch.float32)
    child_int = 2.0 * parent_int.unsqueeze(0) + offsets
    return child_int / max_child


def _morton_decode_axis(key: int, axis: int) -> int:
    """从 Morton code 解码指定轴坐标（depth-2，每轴 2 bit → 取值 0..3）。"""
    coord = 0
    for i in range(2):
        coord |= ((key >> (3 * i + axis)) & 1) << i
    return coord


# ─── 模型工厂函数 ──────────────────────────────────────────────────────────────

def fractal_oct_gen_small(**kwargs) -> FractalOctGen:
    """小型 FractalOctGen（快速测试用）。"""
    return FractalOctGen({**SMALL_CONFIG, **kwargs})


def fractal_oct_gen_base(**kwargs) -> FractalOctGen:
    """标准 FractalOctGen。"""
    return FractalOctGen({**DEFAULT_CONFIG, **kwargs})


def fractal_oct_gen_large(**kwargs) -> FractalOctGen:
    """大型 FractalOctGen（显存充足、追求质量）。"""
    return FractalOctGen({**LARGE_CONFIG, **kwargs})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
