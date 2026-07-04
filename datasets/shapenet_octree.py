"""
ShapeNet 八叉树数据集（通用多深度版本）

加载预处理后的 .npz 文件（每个文件含一个形状的八叉树分裂标签），
为 FractalOctGen 提供任意深度（ALL_PRED_DEPTHS = [2,3,...,depth-1]）的批次。

每个深度 d 的张量：
  split_{d}:      [B, N_d]     分裂/占据标签（0/1）
    field_{d}:      [B, N_d]     连续场目标（SDF/连续占据概率）
  xyz_{d}:        [B, N_d, 3]  归一化 3D 坐标
  mask_{d}:       [B, N_d]     有效位掩码（True=有效）
  parent_idx_{d}: [B, N_d]     指向父层（depth d-1）节点的索引（d≥3）

depth-2 固定 64 节点；其余深度按 batch 内最大长度动态 padding（更省显存），
并对每个样本设较高的安全上限（防止个别超密网格 OOM）。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any

from utils.octree_utils import FULL_DEPTH, OCTREE_DEPTH, ALL_PRED_DEPTHS

# ─── 各深度每样本安全上限（理论最大 = 8^(d-1)，此处取较宽松值防 OOM）──────────
#   对飞机表面八叉树，实际节点数远小于这些上限，几乎不会触发截断。
PER_SAMPLE_CAP = {
    2: 64,
    3: 512,        # 8^(3-2) × 64? depth-3 理论最大 512
    4: 4096,
    5: 16384,
    6: 65536,
    7: 262144,
}

CLASS_NAMES = ['airplane', 'car', 'chair', 'rifle', 'table']


def _cap_for(d: int) -> int:
    return PER_SAMPLE_CAP.get(d, 8 ** (d - 1))


class ShapeNetOctreeDataset(Dataset):
    """加载预处理的 ShapeNet/ModelNet 八叉树 .npz 数据集（通用多深度）。"""

    def __init__(self, data_dir: str, split: str = 'train',
                 class_id: int = 0,
                 pred_depths: Optional[List[int]] = None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.class_id = class_id
        self.pred_depths = list(pred_depths) if pred_depths is not None else list(ALL_PRED_DEPTHS)

        list_path = os.path.join(data_dir, f'{split}.txt')
        if os.path.exists(list_path):
            with open(list_path) as f:
                self.model_ids = [l.strip() for l in f if l.strip()]
        else:
            self.model_ids = [
                os.path.splitext(f)[0]
                for f in os.listdir(data_dir) if f.endswith('.npz')
            ]
            if not self.model_ids:
                raise FileNotFoundError(
                    f'在 {data_dir} 中未找到 {split}.txt 或 .npz 文件')

        print(f'[Dataset] {split} 集：{len(self.model_ids)} 个样本，'
              f'预测深度 {self.pred_depths}')

    def __len__(self) -> int:
        return len(self.model_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        model_id = self.model_ids[idx]
        npz_path = os.path.join(self.data_dir, f'{model_id}.npz')
        data = np.load(npz_path, allow_pickle=False)

        sample = {'class_id': self.class_id, 'model_id': model_id}

        for d in self.pred_depths:
            cap = _cap_for(d)
            split_d = data[f'split_{d}'].astype(np.int64)
            xyz_d = data[f'xyz_{d}'].astype(np.float32)
            field_key = f'field_{d}'
            if field_key in data.files:
                field_d = data[field_key].astype(np.float32)
            else:
                field_d = split_d.astype(np.float32)
            n = len(split_d)
            n_eff = min(n, cap)

            grad_key = f'grad_{d}'
            if grad_key in data.files:
                grad_d = data[grad_key].astype(np.float32)
            else:
                grad_d = np.zeros((n, 3), dtype=np.float32)

            sample[f'split_{d}'] = torch.from_numpy(np.ascontiguousarray(split_d[:n_eff]))
            sample[f'field_{d}'] = torch.from_numpy(np.ascontiguousarray(field_d[:n_eff]))
            sample[f'grad_{d}'] = torch.from_numpy(np.ascontiguousarray(grad_d[:n_eff]))
            sample[f'xyz_{d}'] = torch.from_numpy(np.ascontiguousarray(xyz_d[:n_eff]))

            if d > self.pred_depths[0]:
                pkey = f'parent_idx_{d}'
                if pkey in data.files:
                    pidx = data[pkey].astype(np.int64)[:n_eff]
                else:
                    pidx = np.zeros(n_eff, dtype=np.int64)
                sample[f'parent_idx_{d}'] = torch.from_numpy(np.ascontiguousarray(pidx))

        return sample


def _pad_stack(tensors: List[torch.Tensor], length: int, is_xyz: bool = False,
                is_grad: bool = False, is_field: bool = False):
    """把变长 1D/2D 张量列表 padding 到固定 length 并 stack，同时返回 mask。"""
    B = len(tensors)
    if is_xyz:
        out = torch.zeros(B, length, 3, dtype=torch.float32)
    elif is_grad:
        out = torch.zeros(B, length, 3, dtype=torch.float32)
    elif is_field:
        out = torch.zeros(B, length, dtype=torch.float32)
    else:
        out = torch.zeros(B, length, dtype=torch.long)
    mask = torch.zeros(B, length, dtype=torch.bool)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n > 0:
            out[i, :n] = t
            mask[i, :n] = True
    return out, mask


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """按 batch 内最大长度动态 padding 各深度张量。"""
    pred_depths = [int(k.split('_')[1]) for k in batch[0]
                   if k.startswith('split_')]
    pred_depths = sorted(pred_depths)

    out: Dict[str, Any] = {}
    out['class_id'] = torch.tensor([b['class_id'] for b in batch], dtype=torch.long)
    out['model_ids'] = [b['model_id'] for b in batch]

    for d in pred_depths:
        max_len = max(b[f'split_{d}'].shape[0] for b in batch)
        max_len = max(max_len, 1)  # 至少 1，避免空张量
        split_pad, mask = _pad_stack([b[f'split_{d}'] for b in batch], max_len)
        field_pad, _ = _pad_stack([b[f'field_{d}'] for b in batch], max_len, is_field=True)
        grad_pad, _ = _pad_stack([b[f'grad_{d}'] for b in batch], max_len, is_grad=True)
        xyz_pad, _ = _pad_stack([b[f'xyz_{d}'] for b in batch], max_len, is_xyz=True)
        out[f'split_{d}'] = split_pad
        out[f'field_{d}'] = field_pad.float()
        out[f'grad_{d}'] = grad_pad.float()
        out[f'xyz_{d}'] = xyz_pad
        out[f'mask_{d}'] = mask
        if f'parent_idx_{d}' in batch[0]:
            pidx_pad, _ = _pad_stack([b[f'parent_idx_{d}'] for b in batch], max_len)
            out[f'parent_idx_{d}'] = pidx_pad

    return out


def build_dataloader(data_dir: str, split: str = 'train',
                     batch_size: int = 8, num_workers: int = 4,
                     class_id: int = 0) -> DataLoader:
    dataset = ShapeNetOctreeDataset(data_dir, split=split, class_id=class_id)
    shuffle = (split == 'train')
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=(split == 'train'),
    )


# ─── 合成数据：用于无 ShapeNet 时的单元/烟雾测试 ──────────────────────────────

class SyntheticOctreeDataset(Dataset):
    """用随机球体网格生成合成八叉树数据，验证整条管线（任意深度）。"""

    def __init__(self, size: int = 64, depth: int = OCTREE_DEPTH,
                 full_depth: int = FULL_DEPTH, seed: int = 42,
                 num_points: int = 20000,
                 pred_depths: Optional[List[int]] = None):
        import trimesh as tr
        from utils.octree_utils import (
            build_octree_from_mesh, extract_octree_data, compute_parent_indices,
            mesh_to_sdf_volume)

        self.size = size
        self.depth = depth
        self.full_depth = full_depth
        self.pred_depths = list(pred_depths) if pred_depths is not None else list(ALL_PRED_DEPTHS)
        self._cache = []

        rng = np.random.default_rng(seed)
        print(f'[SyntheticDataset] 生成 {size} 个合成样本（depth={depth}）...')

        for i in range(size):
            # 使用简单基元（球/胶囊/长方体/圆锥）的组合，避免复杂凸包导致 SDF 计算过慢或 OOM
            primitives = []
            n_prims = rng.integers(1, 4)
            for _ in range(n_prims):
                kind = rng.integers(0, 4)
                center = rng.uniform(-0.35, 0.35, 3).astype(np.float32)
                if kind == 0:
                    radius = float(rng.uniform(0.15, 0.40))
                    prim = tr.creation.icosphere(subdivisions=rng.integers(2, 4), radius=radius)
                elif kind == 1:
                    radius = float(rng.uniform(0.08, 0.18))
                    height = float(rng.uniform(0.3, 0.8))
                    prim = tr.creation.capsule(radius=radius, height=height, count=(12, 12))
                elif kind == 2:
                    extents = rng.uniform(0.2, 0.7, 3).astype(np.float32)
                    prim = tr.creation.box(extents=extents)
                else:
                    radius = float(rng.uniform(0.15, 0.35))
                    height = float(rng.uniform(0.3, 0.7))
                    prim = tr.creation.cone(radius=radius, height=height, sections=24)
                prim.vertices += center
                primitives.append(prim)
            if len(primitives) == 1:
                mesh = primitives[0]
            else:
                mesh = tr.util.concatenate(primitives)

            octree = build_octree_from_mesh(mesh, depth=depth, full_depth=full_depth,
                                            num_points=num_points)
            field_volume = mesh_to_sdf_volume(mesh, resolution=64)
            data = extract_octree_data(octree, field_volume=field_volume)
            for d in range(full_depth, depth - 1):
                data[f'parent_idx_{d+1}'] = compute_parent_indices(octree, parent_depth=d)
            self._cache.append(data)

        print('[SyntheticDataset] 生成完成')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self._cache[idx]
        sample = {'class_id': 0, 'model_id': f'synth_{idx}'}
        first = self.pred_depths[0]
        for d in self.pred_depths:
            cap = _cap_for(d)
            split_d = data.get(f'split_{d}', np.zeros(0, dtype=np.int8)).astype(np.int64)
            xyz_d = data.get(f'xyz_{d}', np.zeros((0, 3), dtype=np.float32)).astype(np.float32)
            field_d = data.get(f'field_{d}', split_d.astype(np.float32)).astype(np.float32)
            grad_d = data.get(f'grad_{d}', np.zeros((len(split_d), 3), dtype=np.float32)).astype(np.float32)
            n_eff = min(len(split_d), cap)
            sample[f'split_{d}'] = torch.from_numpy(np.ascontiguousarray(split_d[:n_eff]))
            sample[f'field_{d}'] = torch.from_numpy(np.ascontiguousarray(field_d[:n_eff]))
            sample[f'grad_{d}'] = torch.from_numpy(np.ascontiguousarray(grad_d[:n_eff]))
            sample[f'xyz_{d}'] = torch.from_numpy(np.ascontiguousarray(xyz_d[:n_eff]))
            if d > first:
                pidx = data.get(f'parent_idx_{d}', np.zeros(0, dtype=np.int32)).astype(np.int64)
                pidx = pidx[:n_eff] if len(pidx) >= n_eff else np.zeros(n_eff, dtype=np.int64)
                sample[f'parent_idx_{d}'] = torch.from_numpy(np.ascontiguousarray(pidx))
        return sample
