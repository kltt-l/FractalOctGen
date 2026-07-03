"""
快速管线测试：不依赖外部数据，验证所有模块在当前（通用多深度）架构下可正确
加载、前向、反向与采样。
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

print('=' * 60)
print('FractalOctGen 管线验证（通用多深度架构）')
print('=' * 60)

# ── 1. 导入模块 ────────────────────────────────────────────────────────────────
print('\n[1] 导入模块...')
from utils.octree_utils import (build_octree_from_mesh, extract_octree_data,
                                 compute_parent_indices, save_octree_data,
                                 reconstruct_octree, octree_to_voxel_grid,
                                 OCTREE_DEPTH, FULL_DEPTH,
                                 AR_DEPTHS, OCC_DEPTHS, ALL_PRED_DEPTHS)
from models.octree_ar import OctAR, OccupancyMLP, sinusoidal_pos_enc_3d
from models.fractal_oct_gen import (FractalOctGen, fractal_oct_gen_small,
                                     fractal_oct_gen_base, count_parameters)
from datasets.shapenet_octree import (SyntheticOctreeDataset, ShapeNetOctreeDataset,
                                       collate_fn)
print('  [OK] 所有模块导入成功')
print(f'  OCTREE_DEPTH={OCTREE_DEPTH}  AR_DEPTHS={AR_DEPTHS}  OCC_DEPTHS={OCC_DEPTHS}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  设备: {device}')

# ── 2. 八叉树构建 ──────────────────────────────────────────────────────────────
print('\n[2] 测试八叉树构建（随机网格，depth=%d）...' % OCTREE_DEPTH)
import trimesh
sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
octree = build_octree_from_mesh(sphere, depth=OCTREE_DEPTH, full_depth=FULL_DEPTH,
                                num_points=8000)
print(f'  nnum: {octree.nnum.tolist()}')
data = extract_octree_data(octree)
for d in ALL_PRED_DEPTHS:
    print(f'  depth-{d} 节点数: {len(data[f"split_{d}"])}  占据率 {data[f"split_{d}"].mean():.2%}')
print('  [OK] 八叉树构建成功')

# ── 3. 位置编码 ────────────────────────────────────────────────────────────────
print('\n[3] 测试正弦三维位置编码...')
enc = sinusoidal_pos_enc_3d(torch.rand(4, 64, 3), embed_dim=384)
assert enc.shape == (4, 64, 384), enc.shape
print('  [OK] 位置编码形状', tuple(enc.shape))

# ── 4. OctAR ──────────────────────────────────────────────────────────────────
print('\n[4] 测试 OctAR 前向...')
oct_ar = OctAR(embed_dim=128, num_heads=4, num_blocks=2, cond_dim=128).to(device)
cond_out, logits, field, grad = oct_ar(torch.randint(0, 2, (2, 64)).to(device),
                                       torch.rand(2, 64, 3).to(device),
                                       torch.rand(2, 1, 128).to(device),
                                       torch.ones(2, 64, dtype=torch.bool).to(device))
assert cond_out.shape == (2, 64, 128) and logits.shape == (2, 64) and field.shape == (2, 64) and grad.shape == (2, 64, 3)
print('  [OK] OctAR：cond', tuple(cond_out.shape), 'logits', tuple(logits.shape),
      'field', tuple(field.shape), 'grad', tuple(grad.shape))

# ── 5. OccupancyMLP（输出 logits + field + grad + cond）───────────────────────
print('\n[5] 测试 OccupancyMLP 前向（级联输出）...')
mlp = OccupancyMLP(cond_dim=128, hidden_dim=128, num_layers=2).to(device)
lg, field, grad, cd = mlp(torch.rand(2, 128, 3).to(device), torch.rand(2, 128, 128).to(device))
assert lg.shape == (2, 128) and field.shape == (2, 128) and grad.shape == (2, 128, 3) and cd.shape == (2, 128, 128)
print('  [OK] OccupancyMLP：logits', tuple(lg.shape), 'field', tuple(field.shape),
      'grad', tuple(grad.shape), 'cond', tuple(cd.shape))

# ── 6. 合成数据 + collate + 完整前向 ─────────────────────────────────────────
print('\n[6] 测试 FractalOctGen 完整训练前向（合成数据）...')
ds = SyntheticOctreeDataset(size=3, depth=OCTREE_DEPTH, num_points=4000, seed=0)
batch = collate_fn([ds[i] for i in range(3)])
batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
model = fractal_oct_gen_small().to(device)
print(f'  模型参数量: {count_parameters(model):,}')
# 检查 SDF 梯度目标是否存在
for d in ALL_PRED_DEPTHS:
    if f'grad_{d}' in batch:
        g = batch[f'grad_{d}']
        print(f'    depth-{d} grad 目标: {tuple(g.shape)}')
out = model(batch)
print(f'  总损失: {out["loss"].item():.4f}')
for k in sorted(k for k in out if k.startswith('loss_d')):
    print(f'    {k}: {out[k].item():.4f}')
print('  [OK] 完整前向正确')

# ── 7. 反向传播 ────────────────────────────────────────────────────────────────
print('\n[7] 测试反向传播...')
out['loss'].backward()
grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
assert len(grad_norms) > 0
print(f'  [OK] 有梯度参数 {len(grad_norms)} 组，范数均值 {np.mean(grad_norms):.4f}')

# ── 8. 采样生成 ────────────────────────────────────────────────────────────────
print('\n[8] 测试推理采样（2 个形状）...')
model.eval()
with torch.no_grad():
    so = model.sample(batch_size=2, temperature_l0=1.0, temperature_l1=1.0,
                      temperature_l2=0.0, device=device)
fd = so['finest_depth']
print(f'  finest_depth={fd}')
for d in ALL_PRED_DEPTHS:
    print(f'    depth-{d}: 形状0 节点数 {len(so["split"][d][0])}')
print('  [OK] 采样生成正确')

# ── 9. 体素转网格 ──────────────────────────────────────────────────────────────
print('\n[9] 测试体素转网格...')
from generate import sample_to_voxel_grid, voxel_to_mesh, sample_to_field_grid, field_to_mesh
voxel = sample_to_voxel_grid(so['split'][fd][0], so['xyz'][fd][0], finest_depth=fd)
print(f'  体素网格形状 {voxel.shape}，占据 {int(voxel.sum())}')
field = sample_to_field_grid(so['field'][fd][0], so['xyz'][fd][0], finest_depth=fd)
print(f'  连续场网格形状 {field.shape}，范围 [{field.min():.3f}, {field.max():.3f}]')
mesh_voxel = voxel_to_mesh(voxel.astype(float))
print(f'  二值体素网格顶点 {len(mesh_voxel.vertices)}，三角面 {len(mesh_voxel.faces)}')
if field.min() < 0.0 < field.max():
    mesh_field = field_to_mesh(field)
    print(f'  连续场网格顶点 {len(mesh_field.vertices)}，三角面 {len(mesh_field.faces)}')
else:
    print('  [WARN] 连续场无有效零交叉（随机权重下正常）')
print('  [OK] 体素/连续场转网格正确')

print('\n' + '=' * 60)
print('所有测试通过！管线验证成功。')
print('=' * 60)
