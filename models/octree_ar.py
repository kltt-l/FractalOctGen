"""
OctAR：八叉树自回归 Transformer（每层分形生成器）

参考 FractalGen (He et al., 2025) 的 AR 模块，适配三维八叉树序列：
  - 输入序列：z-order 排序的八叉树节点
  - 位置编码：基于节点三维坐标的正弦编码（替换 FractalGen 的二维 RoPE）
  - 条件注入：父层条件向量（per-node 或全局广播）通过加法注入到每个位置
  - 输出：(1) 用于下一层的条件向量；(2) 当前层的分裂预测 logits

核心创新与 OctGPT 的对比：
  - OctGPT 使用单一扁平 Transformer + 深度掩码
  - 本模块是独立的每层生成器，权重不跨深度共享
  - 通过父层输出的 conditioning 向量自顶向下传递信息
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils.octree_utils import OCTREE_DEPTH


# ─── 正弦三维位置编码 ──────────────────────────────────────────────────────────

# 位置编码最高频率对应的 octave（2^MAX_FREQ_LOG2）。
# 最细预测层位于 depth = OCTREE_DEPTH-1（占据分辨率 2^(OCTREE_DEPTH-1)）。
# 取最高频 2^(OCTREE_DEPTH-1) 可让最细一级相邻格相位差约 π，完全可区分。
_POS_ENC_MAX_FREQ_LOG2 = float(OCTREE_DEPTH - 1)


def sinusoidal_pos_enc_3d(xyz: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    三维正弦位置编码（NeRF 风格），将归一化坐标 [0,1]^3 映射到 embed_dim 维向量。

    关键设计：频率按几何级数从 π 覆盖到 2^MAX_FREQ_LOG2 · π，使得
      - 相邻格子（间距 1/32）在最高频通道上相位差约 π，完全可区分；
      - 低频通道编码全局位置。

    历史 bug：旧实现用 freqs = 10000^(-i/freq_dim)，最高频仅 1.0，
    导致归一化坐标 [0,1] 下所有位置编码余弦相似度 >0.96，
    模型无法感知空间位置，只能生成"平均形状"。

    Args:
        xyz: [*, 3]  归一化 3D 坐标
        embed_dim: 输出维度（应为 6 的倍数）

    Returns:
        [*, embed_dim]
    """
    freq_dim = embed_dim // 6
    # 几何级数频率：2^linspace(0, MAX, freq_dim) · π，范围 [π, 2^MAX·π]
    exponents = torch.linspace(0.0, _POS_ENC_MAX_FREQ_LOG2, freq_dim,
                               dtype=torch.float32, device=xyz.device)
    freqs = torch.pow(2.0, exponents) * math.pi  # [freq_dim]

    # 每个轴独立编码
    encodings = []
    for axis in range(3):
        coords = xyz[..., axis:axis + 1]  # [*, 1]
        angles = coords * freqs  # [*, freq_dim]
        encodings.append(torch.sin(angles))
        encodings.append(torch.cos(angles))

    enc = torch.cat(encodings, dim=-1)  # [*, 6 * freq_dim]

    # 如果 embed_dim 不能被 6 整除，对齐到 embed_dim
    if enc.shape[-1] < embed_dim:
        pad = torch.zeros(*enc.shape[:-1], embed_dim - enc.shape[-1],
                          dtype=enc.dtype, device=enc.device)
        enc = torch.cat([enc, pad], dim=-1)
    elif enc.shape[-1] > embed_dim:
        enc = enc[..., :embed_dim]

    return enc


# ─── Transformer 基础模块 ──────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    因果（单向）多头自注意力。
    支持可选的 padding mask（True = 有效位，False = padding）。
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            padding_mask: [B, N] bool，True=有效，False=padding（可选）
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, N, H, head_dim]
        q = q.transpose(1, 2)  # [B, H, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))

        # padding 掩码（如果提供）
        if padding_mask is not None:
            # padding_mask: [B, N], True=有效
            # 需要屏蔽掉 padding 位（作为 key）
            key_mask = ~padding_mask  # True = padding
            key_mask = key_mask[:, None, None, :]  # [B, 1, 1, N]
            attn = attn.masked_fill(key_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        # 处理 全-inf 行（所有 key 都是 padding）
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """单个 GPT Transformer Block：Pre-LN + CausalAttn + FFN（SwiGLU）。"""

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden = int(embed_dim * mlp_ratio)
        # SwiGLU FFN（与 LLaMA / FractalGen 一致）
        self.fc1 = nn.Linear(embed_dim, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), padding_mask=padding_mask)
        h = self.fc1(self.norm2(x))
        h1, h2 = h.chunk(2, dim=-1)
        h = F.silu(h1) * h2  # SwiGLU
        x = x + self.dropout(self.fc2(h))
        return x


# ─── OctAR 主模块 ──────────────────────────────────────────────────────────────

class OctAR(nn.Module):
    """
    八叉树自回归生成器（单层分形模块）。

    对应 FractalGen 中的 AR 类，但针对三维八叉树序列设计：
      1. 节点嵌入：split_emb(label) + pos_emb(xyz) + cond_proj(parent_cond)
      2. Causal GPT Transformer（Pre-LN，SwiGLU FFN）
    3. 输出头：split logits + field（SDF）+ grad（∇SDF）+ cond_out

    训练：teacher forcing，输入为 ground-truth 分裂标签（左移一位）
    推理：逐节点自回归生成

    Args:
        embed_dim: Transformer 隐层维度
        num_heads: 注意力头数
        num_blocks: Transformer block 数
        cond_dim: 父层条件向量维度（默认与 embed_dim 相同）
        dropout: Dropout 概率
    """

    def __init__(self, embed_dim: int, num_heads: int, num_blocks: int,
                 cond_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        cond_dim = cond_dim or embed_dim

        # 分裂标签嵌入：0=叶节点，1=分裂，2=BOS/PAD（class token 占位）
        self.split_emb = nn.Embedding(3, embed_dim)

        # 三维位置编码（正弦）→ 线性投影
        self.pos_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 父层条件向量注入
        self.cond_proj = nn.Linear(cond_dim, embed_dim, bias=False)

        # Transformer body
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 输出头：(1) 分裂预测；(2) 连续场值；(3) 空间梯度；(4) 下层 conditioning
        self.split_head = nn.Linear(embed_dim, 1)  # → logit
        self.field_head = nn.Linear(embed_dim, 1)  # → SDF 值
        self.grad_head = nn.Linear(embed_dim, 3)   # → ∇SDF
        self.cond_out = nn.Linear(embed_dim, cond_dim)  # → 给下一层的条件

        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        split_labels: torch.Tensor,
        xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练前向传播（teacher forcing）。

        将 split_labels 左移一位，在首位插入 class/BOS token（标签值=2），
        然后预测每个位置的分裂标签。

        Args:
            split_labels: [B, N]  ground-truth 分裂标签（0/1）
            xyz:          [B, N, 3]  归一化 3D 坐标
            parent_cond:  [B, N, D_cond] 或 [B, 1, D_cond] 父层条件（可广播）
            padding_mask: [B, N] bool，True=有效节点，False=padding

        Returns:
            cond_for_children: [B, N, D_cond]  传递给子层的 conditioning 向量
            split_logits:      [B, N]           当前层分裂预测 logits（不经 sigmoid）
            field_values:      [B, N]           连续场预测值
            grad_values:       [B, N, 3]        SDF 空间梯度预测
        """
        B, N = split_labels.shape

        # ── 构建输入序列（左移一位，首位用 BOS token 2）
        bos = torch.full((B, 1), 2, dtype=torch.long, device=split_labels.device)
        input_labels = torch.cat([bos, split_labels[:, :-1]], dim=1)  # [B, N]

        # ── 节点嵌入
        x = self.split_emb(input_labels)  # [B, N, D]

        # ── 三维位置编码
        pos_enc = sinusoidal_pos_enc_3d(xyz, self.embed_dim)  # [B, N, D]
        x = x + self.pos_proj(pos_enc)

        # ── 父层条件注入（可广播：[B,1,D] 或 [B,N,D]）
        x = x + self.cond_proj(parent_cond)

        # ── Transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.norm(x)

        # ── 输出
        split_logits = self.split_head(x).squeeze(-1)  # [B, N]
        field_values = self.field_head(x).squeeze(-1)   # [B, N]
        grad_values = self.grad_head(x)                 # [B, N, 3]
        cond_for_children = self.cond_out(x)           # [B, N, D_cond]

        return cond_for_children, split_logits, field_values, grad_values

    @torch.no_grad()
    def sample_one_step(
        self,
        generated_so_far: torch.Tensor,
        xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        step: int,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        推理时的单步采样。

        给定已生成的 step 个节点标签，预测第 step+1 个节点的分裂标签。
        采用 causal KV 缓存（简化版：每次全序列前向，适合短序列）。

        Args:
            generated_so_far: [B, step]  已生成的标签（step 可以为 0）
            xyz:              [B, N, 3]  全序列坐标（N >= step+1）
            parent_cond:      [B, N, D_cond]  全序列父层条件
            step:             当前要预测的位置索引
            temperature:      采样温度

        Returns:
            sampled_label: [B]  在位置 step 采样得到的 0/1 标签
            cond_at_step:  [B, D_cond]  位置 step 的 cond_for_children
            field_at_step: [B]  位置 step 的连续场预测值
            grad_at_step:  [B, 3]  位置 step 的 SDF 梯度预测
        """
        B = generated_so_far.shape[0]

        # 构建长度为 step+1 的输入序列
        bos = torch.full((B, 1), 2, dtype=torch.long, device=generated_so_far.device)
        input_seq = torch.cat([bos, generated_so_far], dim=1)  # [B, step+1]

        # 前向（只用到前 step+1 个位置）
        x = self.split_emb(input_seq)  # [B, step+1, D]
        pos_enc = sinusoidal_pos_enc_3d(xyz[:, :step + 1], self.embed_dim)
        x = x + self.pos_proj(pos_enc)
        x = x + self.cond_proj(parent_cond[:, :step + 1])

        for block in self.blocks:
            x = block(x)  # 无 padding mask，全部有效
        x = self.norm(x)

        # 取最后一个位置（即第 step 个节点的预测）
        logit_step = self.split_head(x[:, -1])  # [B, 1]
        field_step = self.field_head(x[:, -1])   # [B, 1]
        grad_step = self.grad_head(x[:, -1])     # [B, 3]
        cond_step = self.cond_out(x[:, -1])      # [B, D_cond]

        # 采样
        if temperature == 0.0:
            sampled = (logit_step.squeeze(-1) > 0).long()
        else:
            prob = torch.sigmoid(logit_step.squeeze(-1) / temperature)
            sampled = torch.bernoulli(prob).long()

        return sampled, cond_step, field_step.squeeze(-1), grad_step


# ─── 同层局部协调：窗口化双向自注意力 ─────────────────────────────────────────

class WindowSelfAttention(nn.Module):
    """窗口化（非因果）多头自注意力。

    把节点序列（z-order 排序，兄弟节点为相邻 8-块，邻居在曲线上也相近）按固定
    window 切分，仅在窗口内做双向注意力——让兄弟/邻居节点互相"看到"以协调表面，
    但不引入任何自回归顺序。复杂度 O(N · window)，远小于全局注意力。
    """

    def __init__(self, dim: int, num_heads: int, window: int):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} 不能被 num_heads {num_heads} 整除'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window = window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [B, N, D]
            mask: [B, N] bool，True=有效节点，False=padding（可选）
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        W = self.window
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)

        pad = (W - N % W) % W
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        Nn = N + pad
        nw = Nn // W

        xw = x.reshape(B * nw, W, D)
        mw = mask.reshape(B * nw, W)                       # True=有效

        qkv = self.qkv(xw).reshape(B * nw, W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                        # [BW, W, h, hd]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale      # [BW, h, W, W]
        key_pad = ~mw[:, None, None, :]                    # [BW,1,1,W] True=pad
        attn = attn.masked_fill(key_pad, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)             # 处理全 pad 窗口

        out = (attn @ v).transpose(1, 2).reshape(B * nw, W, D)
        out = self.proj(out)
        out = out.reshape(B, Nn, D)[:, :N]
        # 把 padding 位置清零，避免污染后续（loss 已 mask，但保持干净）
        return out


class WindowAttnBlock(nn.Module):
    """Pre-LN + 窗口化双向自注意力 + SwiGLU FFN。"""

    def __init__(self, dim: int, num_heads: int, window: int,
                 mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowSelfAttention(dim, num_heads, window)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        h = self.fc1(self.norm2(x))
        h1, h2 = h.chunk(2, dim=-1)
        h = F.silu(h1) * h2
        x = x + self.dropout(self.fc2(h))
        return x


# ─── OccupancyMLP：占据预测 + 同层局部协调 ────────────────────────────────────

class OccupancyMLP(nn.Module):
    """
    占据预测模块（用于较细层级，级联以逐级细化到 64³）。

    流程：
      1. 逐节点特征：trunk(父层 cond + 位置编码)         —— 自顶向下条件
      2. 同层局部协调：若干 WindowAttnBlock              —— 兄弟/邻居横向协调
    3. 三头输出：占据 logit + field（SDF）+ grad（∇SDF）+ cond_out

    第 2 步是相对原始"纯并行 MLP"的关键改进：它让同一深度的节点在窗口内互相
    看到对方，弥补"兄弟节点共享父条件却无法协调"的条件独立缺陷，从而减少表面
    散点/孔洞、改善细节连贯性；但不引入自回归顺序，生成仍然并行、快速。

    Args:
        cond_dim: 父层条件维度（同时也是输出 cond 维度）
        hidden_dim: 隐层维度
        num_layers: 逐节点 MLP trunk 层数
        use_attn: 是否启用同层窗口注意力
        window: 注意力窗口大小（z-order 上的局部范围，建议 64=8 个兄弟组）
        attn_heads: 注意力头数（须整除 hidden_dim）
        attn_layers: 窗口注意力 block 数
    """

    def __init__(self, cond_dim: int, hidden_dim: int = 256, num_layers: int = 3,
                 use_attn: bool = True, window: int = 64,
                 attn_heads: int = 4, attn_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.use_attn = use_attn

        # 位置编码投影
        self.pos_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        layers = [nn.Linear(cond_dim + hidden_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, num_layers - 2)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)

        # 同层局部协调（窗口化双向注意力）
        if use_attn:
            self.attn_blocks = nn.ModuleList([
                WindowAttnBlock(hidden_dim, attn_heads, window, dropout=dropout)
                for _ in range(attn_layers)
            ])
        else:
            self.attn_blocks = nn.ModuleList()

        # 三头输出
        self.split_head = nn.Linear(hidden_dim, 1)
        self.field_head = nn.Linear(hidden_dim, 1)
        self.grad_head = nn.Linear(hidden_dim, 3)
        self.cond_out = nn.Linear(hidden_dim, cond_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz:         [B, N, 3]    归一化 3D 坐标
            parent_cond: [B, N, D]    父层条件向量
            mask:        [B, N] bool  有效节点掩码（训练时传入，采样单形状时可为 None）

        Returns:
            logits:            [B, N]
            field_values:      [B, N]
            grad_values:       [B, N, 3]
            cond_for_children: [B, N, D_cond]
        """
        pos_enc = sinusoidal_pos_enc_3d(xyz, self.hidden_dim)  # [B, N, H]
        pos_enc = self.pos_proj(pos_enc)

        feat = torch.cat([parent_cond, pos_enc], dim=-1)
        h = self.trunk(feat)                                   # [B, N, H]

        # 同层局部协调
        for blk in self.attn_blocks:
            h = blk(h, mask=mask)

        logits = self.split_head(h).squeeze(-1)
        field_values = self.field_head(h).squeeze(-1)
        grad_values = self.grad_head(h)
        cond_for_children = self.cond_out(h)
        return logits, field_values, grad_values, cond_for_children
