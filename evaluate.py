"""
FractalOctGen 定量评估脚本

指标：
  - Reconstruction Loss：在测试集上计算平均 BCE loss（分类交叉熵）
  - Coverage（COV）：生成形状集中与测试集最近邻的比例
  - Minimum Matching Distance（MMD）：生成与测试集之间的点云 CD 距离
  - 占据率统计：生成形状的平均体素占据率，反映形状的饱满度

用法：
  # 仅计算重建 loss（快速）
  python evaluate.py --checkpoint runs/default/checkpoint_best.pt \
                     --data_dir data/shapenet_airplane_processed \
                     --mode recon

  # 生成形状 + 计算 COV/MMD（需要较长时间）
  python evaluate.py --checkpoint runs/default/checkpoint_best.pt \
                     --data_dir data/shapenet_airplane_processed \
                     --mode gen --num_gen 200
  
  # 全部指标
  python evaluate.py --checkpoint runs/default/checkpoint_best.pt \
                     --data_dir data/shapenet_airplane_processed \
                     --mode all --num_gen 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── 项目内导入 ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models.fractal_oct_gen import (
    fractal_oct_gen_base, fractal_oct_gen_small, fractal_oct_gen_large)
from datasets.shapenet_octree import ShapeNetOctreeDataset, collate_fn
from utils.octree_utils import OCTREE_DEPTH, FULL_DEPTH, ALL_PRED_DEPTHS, OCC_DEPTHS
from generate import generate_shapes, sample_to_voxel_grid

_FINEST_DEPTH = OCC_DEPTHS[-1] if OCC_DEPTHS else ALL_PRED_DEPTHS[-1]


# ─── 重建 Loss ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_reconstruction(model, data_dir: str, split: str = "test",
                         batch_size: int = 16, device: torch.device = None):
    """
    在 test split 上计算平均重建 BCE loss（逐层）。
    与训练时的 forward() 完全相同的损失计算。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ShapeNetOctreeDataset(data_dir, split=split)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)

    model.eval()
    model.to(device)

    total_loss = 0.0
    n_batches = 0
    level_losses: dict = {}

    print(f"[重建评估] split={split}, {len(dataset)} 个样本")
    for batch in tqdm(loader, desc="重建 loss"):
        # 迁移到设备
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        loss_dict = model(batch)
        total_loss += loss_dict["loss"].item()
        for k, v in loss_dict.items():
            if k.startswith("loss_d"):
                level_losses[k] = level_losses.get(k, 0.0) + v.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_level_losses = {k: l / max(n_batches, 1) for k, l in level_losses.items()}

    print(f"\n重建评估结果（{split} 集）：")
    print(f"  总 loss：{avg_loss:.4f}")
    for k in sorted(avg_level_losses):
        print(f"  {k}：{avg_level_losses[k]:.4f}")

    return {"loss": avg_loss, "level_losses": avg_level_losses}


# ─── 点云 Chamfer Distance ─────────────────────────────────────────────────────

def voxel_to_pointcloud(voxel_grid: np.ndarray, n_points: int = 2048) -> np.ndarray:
    """从体素网格均匀采样点云（通过占据体素中心）。"""
    occupied = np.argwhere(voxel_grid)  # [N, 3]
    if len(occupied) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    # 归一化到 [0, 1]^3
    res = voxel_grid.shape[0]
    pts = occupied.astype(np.float32) / (res - 1)

    # 有放回采样到固定点数
    idx = np.random.choice(len(pts), n_points, replace=(len(pts) < n_points))
    return pts[idx]


def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    计算两个点云之间的 Chamfer Distance（双向平均最近邻距离）。

    Args:
        pc1, pc2: [N, 3] float32 点云

    Returns:
        CD 值（越小越好）
    """
    # [N, 1, 3] vs [1, M, 3] → [N, M]
    diff = pc1[:, None, :] - pc2[None, :, :]        # [N, M, 3]
    dist2 = (diff ** 2).sum(axis=2)                  # [N, M]

    cd = dist2.min(axis=1).mean() + dist2.min(axis=0).mean()
    return float(cd)


# ─── COV + MMD 评估 ───────────────────────────────────────────────────────────

def eval_generation(model, data_dir: str, num_gen: int = 200,
                    n_points: int = 2048, device: torch.device = None):
    """
    生成 num_gen 个形状，与测试集计算：
      - COV（Coverage）：生成集对测试集的覆盖率
      - MMD（Minimum Matching Distance）：平均最小 CD
      - 平均占据率
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 生成形状 ────────────────────────────────────────────────────────────────
    print(f"\n[生成评估] 生成 {num_gen} 个形状...")
    results = generate_shapes(
        model=model,
        num_samples=num_gen,
        device=device,
        batch_size=4,
        resolution=None,           # 自动取 2^finest_depth（如 64）
        output_dir=None,
        export_mesh=False,
        export_voxel=False,
        verbose=False,
    )

    gen_voxels = [r["voxel_grid"] for r in results]
    gen_pcs = np.stack([voxel_to_pointcloud(v, n_points) for v in gen_voxels])  # [G, N, 3]

    avg_occ = np.mean([v.mean() for v in gen_voxels])
    print(f"  平均体素占据率：{avg_occ*100:.1f}%")

    # ── 加载测试集体素 ──────────────────────────────────────────────────────────
    from datasets.shapenet_octree import ShapeNetOctreeDataset
    test_dataset = ShapeNetOctreeDataset(data_dir, split="test")
    print(f"  加载 {len(test_dataset)} 个测试形状...")

    test_pcs = []
    for i in tqdm(range(len(test_dataset)), desc="加载测试集"):
        sample = test_dataset[i]
        # 用最细占据层（finest_depth）重建体素
        split_f = sample[f"split_{_FINEST_DEPTH}"].long()
        xyz_f = sample[f"xyz_{_FINEST_DEPTH}"]
        voxel = sample_to_voxel_grid(split_f, xyz_f, finest_depth=_FINEST_DEPTH)
        test_pcs.append(voxel_to_pointcloud(voxel, n_points))

    test_pcs = np.stack(test_pcs)  # [T, N, 3]
    G, T = len(gen_pcs), len(test_pcs)

    # ── 计算 CD 距离矩阵 ────────────────────────────────────────────────────────
    print(f"\n  计算 CD 距离矩阵 ({G}×{T})，请稍候...")
    cd_matrix = np.zeros((G, T), dtype=np.float32)
    for i in tqdm(range(G), desc="计算 CD"):
        for j in range(T):
            cd_matrix[i, j] = chamfer_distance(gen_pcs[i], test_pcs[j])

    # MMD：每个生成形状到测试集的最小 CD 的均值
    mmd = cd_matrix.min(axis=1).mean()

    # COV：测试集中被覆盖（至少一个生成形状最近）的比例
    matched_test = set(cd_matrix.argmin(axis=1).tolist())
    cov = len(matched_test) / T

    print(f"\n生成评估结果：")
    print(f"  平均体素占据率：{avg_occ*100:.2f}%")
    print(f"  COV（覆盖率）  ：{cov*100:.1f}%  （测试集 {len(matched_test)}/{T} 被覆盖）")
    print(f"  MMD（最小匹配距离）：{mmd:.6f}  （越小越好）")

    return {"cov": cov, "mmd": mmd, "avg_occupancy": avg_occ}


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FractalOctGen 定量评估")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--data_dir", type=str,
                        default="data/shapenet_airplane_processed",
                        help="预处理数据目录")
    parser.add_argument("--mode", choices=["recon", "gen", "all"],
                        default="all", help="评估模式")
    parser.add_argument("--num_gen", type=int, default=200,
                        help="生成形状数量（gen/all 模式）")
    parser.add_argument("--model_size", choices=["small", "base", "large"], default="base")
    parser.add_argument("--split", type=str, default="test",
                        help="重建评估使用的 split（test/val）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 加载模型
    if args.model_size == "small":
        model = fractal_oct_gen_small()
    elif args.model_size == "large":
        model = fractal_oct_gen_large()
    else:
        model = fractal_oct_gen_base()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    print(f"已加载 checkpoint（epoch={epoch}, val_loss={val_loss:.4f}）")
    model.to(device).eval()

    results = {}

    if args.mode in ("recon", "all"):
        recon_res = eval_reconstruction(model, args.data_dir,
                                         split=args.split, device=device)
        results.update(recon_res)

    if args.mode in ("gen", "all"):
        gen_res = eval_generation(model, args.data_dir,
                                   num_gen=args.num_gen, device=device)
        results.update(gen_res)

    print("\n" + "=" * 50)
    print("最终评估摘要：")
    for k, v in results.items():
        if isinstance(v, list):
            print(f"  {k}: {[f'{x:.4f}' for x in v]}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
