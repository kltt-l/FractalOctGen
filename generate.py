"""
FractalOctGen 形状生成脚本

从训练好的模型生成三维形状，导出为体素网格（.npy）或三角网格（.obj/.ply）。

用法：
  # 使用最优 checkpoint 生成 16 个形状
  python generate.py --checkpoint runs/airplane/checkpoint_best.pt \
                     --num_samples 16 --output_dir outputs/shapes

  # 快速测试（随机初始化模型）
  python generate.py --random_model --num_samples 4 --output_dir outputs/test
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure
from scipy import ndimage

from models.fractal_oct_gen import (
    fractal_oct_gen_base, fractal_oct_gen_small, fractal_oct_gen_large)
from utils.octree_utils import OCTREE_DEPTH, FULL_DEPTH


# ─── 体素后处理：薄壳 → 实体（封孔 + 填充 + 去碎块）────────────────────────────

def postprocess_voxel_grid(
    grid: np.ndarray,
    solidify: bool = True,
    keep_largest: bool = True,
    closing_iters: int = 1,
) -> np.ndarray:
    """把"空心薄壳"体素整理为干净的"实体"体素，再交给 Marching Cubes。

    管线：形态学闭运算（封住薄壳上的针孔）→ 填充内部空腔（薄壳变实体）
         → 保留最大连通分量（去掉漂浮碎块）。

    实体化后做 MC 只会得到一张朝外的水密表面，从根本上消除薄壳双壁/孔洞。

    Args:
        grid: [R,R,R] bool/float 体素（薄壳）
        solidify: 是否填充内部（实体化）
        keep_largest: 是否只保留最大连通分量
        closing_iters: 闭运算迭代次数（越大越能封大孔，但过大会粘连薄机翼）
    Returns:
        [R,R,R] bool 体素
    """
    g = grid.astype(bool)
    if g.sum() == 0:
        return g

    if closing_iters > 0:
        g = ndimage.binary_closing(g, iterations=closing_iters)

    if solidify:
        # 沿三个轴分别填洞，对未完全封闭的薄壳更鲁棒，再整体填一次
        g = ndimage.binary_fill_holes(g)

    if keep_largest:
        labels, n = ndimage.label(g)
        if n > 1:
            counts = np.bincount(labels.ravel())
            counts[0] = 0  # 背景不计
            g = labels == counts.argmax()

    return g


# ─── 体素转网格（Marching Cubes）──────────────────────────────────────────────

def voxel_to_mesh(voxel_grid: np.ndarray, level: float = 0.5,
                  smooth_iters: int = 10) -> trimesh.Trimesh:
    """
    用 Marching Cubes 从体素网格提取三角网格。

    修复：
    1. Padding：在体素网格四周各加一层 0，防止位于边界的实体块产生开口面。
    2. 法向量方向：Marching Cubes 对 1=实体/0=空 的网格，法向量默认朝内
       （梯度方向 0→1），调用 mesh.invert() 翻转为朝外。
    3. 平滑：使用 Taubin（λ/μ）平滑，去抖动且几乎不收缩体积，
       优于 Laplacian（后者会收缩并放大薄壳褶皱）。

    Args:
        voxel_grid: [R, R, R] float/bool 体素值（1/True=实体，0/False=空）
        level: iso-surface 阈值
        smooth_iters: Taubin 平滑迭代次数（0=不平滑）

    Returns:
        trimesh.Trimesh（法向量朝外）
    """
    resolution = voxel_grid.shape[0]
    grid = voxel_grid.astype(np.float32)

    # Padding：各方向加 1 层空体素，确保边界处的实体面可以封闭
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=0.0)

    try:
        verts, faces, normals, _ = measure.marching_cubes(padded, level=level)
    except Exception as e:
        print(f'[WARN] Marching Cubes 失败: {e}，返回空网格')
        return trimesh.Trimesh()

    # 顶点坐标修正：减去 padding 偏移，并归一化到 [0,1]^3
    verts = (verts - 1.0) / (resolution - 1)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                            vertex_normals=normals, process=False)

    # 翻转法向量：使其指向实体外部（Marching Cubes 默认朝内）
    mesh.invert()

    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)

    return mesh


def sample_to_voxel_grid(
    split_f: torch.Tensor, xyz_f: torch.Tensor,
    finest_depth: int, resolution: int = None,
) -> np.ndarray:
    """
    将最细占据层（finest_depth）的占据节点重建为体素网格。

    ─────────────────────────────────────────────────────────────────────
    关键语义：
      表面八叉树中，split_d == 1 表示该 depth-d 节点含表面（继续向下细分），
      split_d == 0 表示该 cell 自身不含表面，是表面周围的空兄弟节点，不填充。

      最细层 finest_depth（如 depth-6）上 split==1 的节点即真实表面体素。
      在 2^finest_depth 分辨率（depth-6 → 64³）下，每个节点对应一个体素。
    ─────────────────────────────────────────────────────────────────────

    坐标归一化（与训练 _keys_to_xyz 一致）：xyz = int / (2^depth - 1)

    Args:
        split_f: [N]     最细层节点占据决策（1=含表面）
        xyz_f:   [N, 3]  最细层节点归一化坐标
        finest_depth: 最细占据层深度
        resolution: 体素网格分辨率（默认 2^finest_depth）

    Returns:
        [R, R, R] bool 体素网格
    """
    grid_dim = 2 ** finest_depth
    if resolution is None:
        resolution = grid_dim
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    if len(split_f) == 0:
        return grid

    block = max(1, resolution // grid_dim)
    occ_mask = (split_f.cpu() == 1).numpy()
    xyz = xyz_f.cpu().numpy()[occ_mask]
    if len(xyz) == 0:
        return grid

    max_int = float(2 ** finest_depth - 1)
    int_coords = np.round(xyz * max_int).astype(int)
    corners = np.clip(int_coords * block, 0, resolution - block)
    for c in corners:
        x, y, z = c
        grid[x:x + block, y:y + block, z:z + block] = True

    return grid


def sample_to_field_grid(
    field_f: torch.Tensor, xyz_f: torch.Tensor,
    finest_depth: int, resolution: int = None,
    default_outside: float = 1.0,
) -> np.ndarray:
    """把最细层的连续场节点值重建为稠密标量场。

    约定：field 外部为正、内部为负（标准 SDF），未知区域默认填
    default_outside（默认 +1，表示外部），避免 Marching Cubes 在空白处
    把 0 误判为表面。

    当 resolution == 2^finest_depth 时，每个节点对应唯一体素；当分辨率
    更高时，采用三线性插值散射，使 MC 能在更细网格上提取光滑表面。
    """
    grid_dim = 2 ** finest_depth
    if resolution is None:
        resolution = grid_dim

    grid = np.full((resolution, resolution, resolution),
                   default_outside, dtype=np.float32)
    counts = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    if len(field_f) == 0:
        return grid

    xyz = xyz_f.cpu().numpy()
    vals = field_f.detach().cpu().numpy()

    if resolution == grid_dim:
        max_int = float(2 ** finest_depth - 1)
        int_coords = np.round(xyz * max_int).astype(int)
        int_coords = np.clip(int_coords, 0, resolution - 1)
        for (x, y, z), value in zip(int_coords, vals):
            grid[x, y, z] += value
            counts[x, y, z] += 1.0
        mask = counts > 0
        grid[mask] /= counts[mask]
    else:
        grid_coords = xyz * (resolution - 1)
        for c, value in zip(grid_coords, vals):
            x0 = int(np.floor(c[0]))
            y0 = int(np.floor(c[1]))
            z0 = int(np.floor(c[2]))
            x1 = min(x0 + 1, resolution - 1)
            y1 = min(y0 + 1, resolution - 1)
            z1 = min(z0 + 1, resolution - 1)
            dx = c[0] - x0
            dy = c[1] - y0
            dz = c[2] - z0

            weights = [
                ((1 - dx) * (1 - dy) * (1 - dz), x0, y0, z0),
                ((1 - dx) * (1 - dy) * dz,     x0, y0, z1),
                ((1 - dx) * dy     * (1 - dz), x0, y1, z0),
                ((1 - dx) * dy     * dz,     x0, y1, z1),
                (dx     * (1 - dy) * (1 - dz), x1, y0, z0),
                (dx     * (1 - dy) * dz,     x1, y0, z1),
                (dx     * dy     * (1 - dz), x1, y1, z0),
                (dx     * dy     * dz,     x1, y1, z1),
            ]
            for w, ix, iy, iz in weights:
                grid[ix, iy, iz] += w * value
                counts[ix, iy, iz] += w
        mask = counts > 0
        grid[mask] /= counts[mask]

    return grid


def field_to_mesh(field_grid: np.ndarray, level: float = 0.0,
                  smooth_iters: int = 10,
                  field_smooth_sigma: float = 0.0) -> trimesh.Trimesh:
    """从连续 SDF 标量场用 Marching Cubes 提取水密网格。

    约定：field 外部为正、内部为负，level=0 为表面。MC 沿梯度方向
    （由负到正）提取的表面法向量自然朝外，因此不需要 invert()。

    Args:
        field_grid: [R, R, R] 连续 SDF 场。
        level: iso-surface 阈值，默认 0。
        smooth_iters: Taubin 平滑迭代次数。
        field_smooth_sigma: 在 MC 前对 SDF 做高斯平滑的 sigma（0=不平滑）。

    Returns:
        trimesh.Trimesh（法向量朝外）。
    """
    from scipy.ndimage import gaussian_filter

    resolution = field_grid.shape[0]
    grid = field_grid.astype(np.float32)

    if field_smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=field_smooth_sigma)

    # 四周补一层“外部”，保证边界处表面封闭
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=1.0)

    try:
        verts, faces, normals, _ = measure.marching_cubes(padded, level=level)
    except Exception as e:
        print(f'[WARN] 连续场 Marching Cubes 失败: {e}，返回空网格')
        return trimesh.Trimesh()

    verts = (verts - 1.0) / (resolution - 1)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                            vertex_normals=normals, process=False)

    # 标准 SDF 法向量已朝外，无需 invert
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)

    return mesh


# ─── 可视化（ASCII art 横截面，方便终端预览）──────────────────────────────────

def print_voxel_slice(voxel_grid: np.ndarray, axis: int = 2,
                       slice_idx: int = None) -> None:
    """打印体素网格某层的 ASCII 横截面（用于终端快速预览）。"""
    R = voxel_grid.shape[0]
    if slice_idx is None:
        slice_idx = R // 2

    if axis == 0:
        slc = voxel_grid[slice_idx, :, :]
    elif axis == 1:
        slc = voxel_grid[:, slice_idx, :]
    else:
        slc = voxel_grid[:, :, slice_idx]

    chars = {True: '█', False: ' '}
    print(f'  ── 横截面 (axis={axis}, idx={slice_idx}) ──')
    for row in slc:
        print('  ' + ''.join(chars[v] for v in row))


# ─── 主生成函数 ──────────────────────────────────────────────────────────────

def generate_shapes(
    model,
    num_samples: int,
    device: torch.device,
    temperature_l0: float = 0.9,
    temperature_l1: float = 0.9,
    temperature_l2: float = 0.0,
    batch_size: int = 4,
    resolution: int = None,
    output_dir: Path = None,
    export_mesh: bool = True,
    export_voxel: bool = True,
    solidify: bool = True,
    keep_largest: bool = True,
    closing_iters: int = 1,
    smooth_iters: int = 10,
    field_level: float = 0.0,
    field_smooth_sigma: float = 0.0,
    verbose: bool = True,
) -> list:
    """
    批量生成形状并导出。

    网格提取策略：优先使用最细层预测的连续 SDF 场（field_level=0 处等值面）
    提取水密表面；若场无有效零交叉，则回退到二值体素网格 + Marching Cubes。

    Args:
        model: FractalOctGen（已加载权重）
        num_samples: 生成形状数量
        device: 推理设备
        temperature_*: 各层采样温度
        batch_size: 每批生成数量
        resolution: 体素网格分辨率
        output_dir: 输出目录（None 则不保存文件）
        export_mesh: 是否导出 .obj 网格
        export_voxel: 是否导出 .npy 体素网格
        solidify/keep_largest/closing_iters: 仅作用于导出的二值体素网格
        smooth_iters: Taubin 平滑迭代次数
        field_level: 连续场 Marching Cubes 的 iso 阈值
        field_smooth_sigma: MC 前对 SDF 做高斯平滑的 sigma（0=不平滑）

    Returns:
        list of dict（每个形状的结果）
    """
    model.eval()
    results = []
    n_generated = 0

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    while n_generated < num_samples:
        cur_batch = min(batch_size, num_samples - n_generated)
        t0 = time.time()

        with torch.no_grad():
            out = model.sample(
                batch_size=cur_batch,
                class_id=0,
                temperature_l0=temperature_l0,
                temperature_l1=temperature_l1,
                temperature_l2=temperature_l2,
                device=device,
            )

        gen_time = time.time() - t0
        if verbose:
            print(f'生成 {cur_batch} 个形状，耗时 {gen_time:.1f}s')

        depths = out['depths']
        finest_depth = out['finest_depth']
        res = resolution if resolution is not None else 2 ** finest_depth

        # 逐形状处理
        for i in range(cur_batch):
            shape_id = n_generated + i
            split_f = out['split'][finest_depth][i]
            field_f = out['field'][finest_depth][i]
            xyz_f   = out['xyz'][finest_depth][i]

            # 统计信息：各深度节点数 / 占据数
            if verbose:
                stats = []
                for d in depths:
                    sp = out['split'][d][i]
                    n_total = len(sp)
                    n_occ = (sp == 1).sum().item() if n_total > 0 else 0
                    stats.append(f'd{d}:{n_occ}/{n_total}')
                print(f'  形状 {shape_id}: ' + ' '.join(stats))

            # 转换为体素网格（最细层占据 → 2^finest_depth 分辨率）
            voxel_grid = sample_to_voxel_grid(
                split_f, xyz_f, finest_depth=finest_depth, resolution=res
            )

            # 连续 SDF 场：默认未知区域为外部（+1），保证 MC 只提取有效表面
            field_grid = sample_to_field_grid(
                field_f, xyz_f, finest_depth=finest_depth, resolution=res,
                default_outside=1.0,
            )

            # 后处理：薄壳 → 实体（封孔 + 填充 + 去碎块）
            # 该体素网格用于统计与导出；连续场网格直接用于提面。
            voxel_grid = postprocess_voxel_grid(
                voxel_grid, solidify=solidify, keep_largest=keep_largest,
                closing_iters=closing_iters,
            )

            if verbose:
                print_voxel_slice(voxel_grid, axis=1, slice_idx=res // 2)

            n_voxels = int(voxel_grid.sum())
            result = {
                'shape_id': shape_id,
                'voxel_grid': voxel_grid,
                'field_grid': field_grid,
                'n_occupied': n_voxels,
            }

            # 导出文件
            if output_dir and export_voxel:
                np.save(output_dir / f'shape_{shape_id:04d}.npy', voxel_grid)

            if output_dir:
                np.save(output_dir / f'shape_{shape_id:04d}_field.npy', field_grid)

            if output_dir and export_mesh:
                # 优先从连续 SDF 场提取水密表面；若场无有效零交叉则回退到二值体素
                field_min = float(field_grid.min())
                field_max = float(field_grid.max())
                if field_min < field_level < field_max:
                    mesh = field_to_mesh(field_grid, level=field_level,
                                         smooth_iters=smooth_iters,
                                         field_smooth_sigma=field_smooth_sigma)
                else:
                    print(f'[WARN] 形状 {shape_id} 连续场无有效零交叉，'
                          f'回退到二值体素网格（field range=[{field_min:.3f}, {field_max:.3f}]）')
                    mesh = voxel_to_mesh(voxel_grid.astype(np.float32),
                                         level=0.5,
                                         smooth_iters=smooth_iters)
                if len(mesh.vertices) > 0:
                    mesh.export(str(output_dir / f'shape_{shape_id:04d}.obj'))
                    result['mesh'] = mesh

            results.append(result)

        n_generated += cur_batch

    return results


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FractalOctGen 形状生成')

    parser.add_argument('--checkpoint', type=str, default='',
                        help='模型 checkpoint 路径')
    parser.add_argument('--random_model', action='store_true',
                        help='使用随机初始化的模型（测试用）')
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'])

    parser.add_argument('--num_samples', type=int, default=16,
                        help='生成形状数量')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='每批生成形状数')
    parser.add_argument('--output_dir', type=str, default='outputs/shapes')
    parser.add_argument('--resolution', type=int, default=0,
                        help='体素网格分辨率（0=自动，取最细占据层 2^(depth-1)，如 64）')

    parser.add_argument('--temperature_l0', type=float, default=0.9)
    parser.add_argument('--temperature_l1', type=float, default=0.9)
    parser.add_argument('--temperature_l2', type=float, default=0.0)

    parser.add_argument('--no_mesh', action='store_true', help='不导出网格文件')
    parser.add_argument('--no_voxel', action='store_true', help='不导出体素文件')

    # 体素后处理 / 表面提取
    parser.add_argument('--no_solidify', action='store_true',
                        help='关闭实体化填充（保留空心薄壳）。仅影响导出的 .npy 体素，'
                             '连续场提面不受影响。')
    parser.add_argument('--no_keep_largest', action='store_true',
                        help='关闭"只保留最大连通分量"（保留漂浮碎块）。仅影响 .npy 体素。')
    parser.add_argument('--closing_iters', type=int, default=1,
                        help='形态学闭运算迭代次数（封孔强度，过大会粘连薄机翼）。仅影响 .npy 体素。')
    parser.add_argument('--smooth_iters', type=int, default=10,
                        help='Taubin 网格平滑迭代次数（0=不平滑；对连续场与二值回退均生效）')
    parser.add_argument('--field_level', type=float, default=0.0,
                        help='连续场 Marching Cubes 的 iso 阈值（默认 0.0，对应 SDF 表面）')
    parser.add_argument('--field_smooth_sigma', type=float, default=0.0,
                        help='MC 前对 SDF 场做高斯平滑的 sigma（0=不平滑，建议 0.3~0.8）')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    if args.model_size == 'small':
        model = fractal_oct_gen_small()
    elif args.model_size == 'large':
        model = fractal_oct_gen_large()
    else:
        model = fractal_oct_gen_base()

    if args.checkpoint and not args.random_model:
        if not os.path.exists(args.checkpoint):
            print(f'[错误] Checkpoint 不存在: {args.checkpoint}')
            return
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'已加载 checkpoint: {args.checkpoint}')
        print(f'（Epoch {ckpt.get("epoch", "?")}，'
              f'val_loss={ckpt.get("best_val_loss", "?"):.4f}）')
    elif not args.random_model:
        print('[WARN] 未指定 checkpoint，使用随机初始化模型')

    model = model.to(device)
    model.eval()

    # ── 生成 ──────────────────────────────────────────────────────────────────
    print(f'\n开始生成 {args.num_samples} 个形状...\n')

    results = generate_shapes(
        model=model,
        num_samples=args.num_samples,
        device=device,
        temperature_l0=args.temperature_l0,
        temperature_l1=args.temperature_l1,
        temperature_l2=args.temperature_l2,
        batch_size=args.batch_size,
        resolution=(args.resolution if args.resolution > 0 else None),
        output_dir=args.output_dir,
        export_mesh=not args.no_mesh,
        export_voxel=not args.no_voxel,
        solidify=not args.no_solidify,
        keep_largest=not args.no_keep_largest,
        closing_iters=args.closing_iters,
        smooth_iters=args.smooth_iters,
        field_level=args.field_level,
        field_smooth_sigma=args.field_smooth_sigma,
        verbose=True,
    )

    # ── 统计 ──────────────────────────────────────────────────────────────────
    n_occupied_all = [r['n_occupied'] for r in results]
    print(f'\n生成完成！')
    print(f'  平均占据节点数: {np.mean(n_occupied_all):.1f} ± {np.std(n_occupied_all):.1f}')
    print(f'  文件保存于: {args.output_dir}')


if __name__ == '__main__':
    main()
