"""
数据预处理脚本：ShapeNetCore.v2 mesh（OBJ）→ ocnn 表面八叉树 → .npz

目录结构（ShapeNetCore.v2）：
  shapenet_dir/
    02691156/<model_id>/models/model_normalized.obj   ← airplane
    02958343/...                                        ← car
    ...

用法：
  python preprocess.py \
      --shapenet_dir data/ShapeNetCore.v2 \
      --category airplane \
      --num_points 100000 \
      --output_dir data/shapenet_airplane_processed \
      --num_workers 16

输出：
  output_dir/<model_id>.npz          每个形状一份八叉树数据
  output_dir/{train,val,test}.txt    8:1:1 随机划分的模型 ID 列表
"""

import argparse
import os
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.octree_utils import (
    build_octree_from_mesh,
    save_octree_data,
    OCTREE_DEPTH,
)

CATEGORY_SYNSET = {
    "airplane": "02691156",
    "car": "02958343",
    "chair": "03001627",
    "rifle": "04090263",
    "table": "04379243",
}


def find_mesh_file(model_dir: str) -> str:
    """在模型目录中查找网格文件（OBJ / OFF / PLY），优先 model_normalized。"""
    import glob
    for ext in ["*.obj", "*.off", "*.ply"]:
        files = glob.glob(os.path.join(model_dir, "**", ext), recursive=True)
        if files:
            for f in files:
                if "model_normalized" in os.path.basename(f):
                    return f
            return files[0]
    return None


def _process_one_shapenet(task):
    """单个模型的预处理（供多进程池调用，须为模块级函数以便 pickle）。

    Args:
        task: (model_id, model_dir, output_path, depth, num_points)
    Returns:
        (model_id, ok: bool, msg: str)
    """
    model_id, model_dir, output_path, depth, num_points = task

    # 限制每个 worker 的线程数，避免 N 进程 × M 线程导致 CPU 超额订阅
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    import trimesh

    if os.path.exists(output_path):
        return (model_id, True, "已存在")

    mesh_path = find_mesh_file(model_dir)
    if mesh_path is None:
        return (model_id, False, "找不到网格文件")

    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return (model_id, False, "非三角网格或无面片")
        mesh.fix_normals()
        octree = build_octree_from_mesh(mesh, depth=depth, num_points=num_points)
        if octree.nnum[depth].item() < 4:
            return (model_id, False, f"八叉树节点数过少（{octree.nnum[depth].item()}）")
        save_octree_data(octree, output_path)
        return (model_id, True, "ok")
    except Exception as e:
        return (model_id, False, f"{type(e).__name__}: {e}")


def process_shapenet_mesh(shapenet_dir: str, category: str, output_dir: str,
                          depth: int = OCTREE_DEPTH, num_points: int = 100000,
                          max_models: int = 0, num_workers: int = 0):
    """从 ShapeNetCore.v2 mesh（OBJ）预处理为八叉树 .npz（多进程并行）。

    Args:
        shapenet_dir: ShapeNetCore.v2 根目录（含 synset 子目录）
        category:     类别名（见 CATEGORY_SYNSET）
        output_dir:   输出 .npz 目录
        depth:        八叉树深度
        num_points:   网格表面采样点数（深八叉树建议 ≥ 100000）
        max_models:   调试用，限制模型数（0 = 全部）
        num_workers:  并行进程数（0 = 自动取 os.cpu_count()）
    """
    synset_id = CATEGORY_SYNSET[category]
    category_dir = os.path.join(shapenet_dir, synset_id)
    if not os.path.exists(category_dir):
        raise FileNotFoundError(f"[错误] 目录不存在：{category_dir}")

    model_ids = [d for d in os.listdir(category_dir)
                 if os.path.isdir(os.path.join(category_dir, d))]
    if max_models > 0:
        model_ids = model_ids[:max_models]
    print(f"[INFO] 找到 {len(model_ids)} 个 {category} 模型")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (model_id, os.path.join(category_dir, model_id),
         str(output_dir / f"{model_id}.npz"), depth, num_points)
        for model_id in model_ids
    ]

    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, min(num_workers, len(tasks)))
    print(f"[INFO] 使用 {num_workers} 个进程并行预处理")

    success, failed = 0, []
    filelist = []

    if num_workers == 1:
        results_iter = (_process_one_shapenet(t) for t in tasks)
        pool = None
    else:
        # chunksize=1：每个任务耗时较大且不均，细粒度分发可保持负载均衡
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=num_workers)
        results_iter = pool.imap_unordered(_process_one_shapenet, tasks, chunksize=1)

    try:
        for model_id, ok, msg in tqdm(
                results_iter, total=len(tasks),
                desc=f"预处理 ShapeNet {category}"):
            if ok:
                filelist.append(model_id)
                success += 1
            else:
                failed.append(f"{model_id}: {msg}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\n[INFO] 成功：{success}，失败：{len(failed)}")
    if failed:
        print("[失败样例]（前 10 条）：")
        for msg in failed[:10]:
            print(f"  - {msg}")

    # 随机划分 8:1:1
    np.random.seed(42)
    perm = np.random.permutation(len(filelist)).tolist()
    n = len(filelist)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    for split_name, indices in [
        ("train", perm[:n_train]),
        ("val",   perm[n_train:n_train + n_val]),
        ("test",  perm[n_train + n_val:]),
    ]:
        ids = [filelist[i] for i in indices]
        list_path = output_dir / f"{split_name}.txt"
        list_path.write_text("\n".join(ids))
        print(f"  {split_name}: {len(ids)} 个样本 → {list_path}")


def main():
    parser = argparse.ArgumentParser(
        description="预处理 ShapeNetCore.v2 mesh 为八叉树 .npz")
    parser.add_argument("--shapenet_dir", type=str, required=True,
                        help="ShapeNetCore.v2 根目录（含 02691156/ 等 synset 子目录）")
    parser.add_argument("--category", type=str, default="airplane",
                        choices=list(CATEGORY_SYNSET.keys()),
                        help="形状类别")
    parser.add_argument("--num_points", type=int, default=100000,
                        help="网格表面采样点数（深八叉树建议 ≥ 100000）")
    parser.add_argument("--output_dir", type=str,
                        default="data/shapenet_airplane_processed",
                        help="输出 .npz 文件目录")
    parser.add_argument("--depth", type=int, default=OCTREE_DEPTH,
                        help=f"八叉树深度（默认 {OCTREE_DEPTH}）")
    parser.add_argument("--max_models", type=int, default=0,
                        help="调试：限制最大处理模型数（0=全部）")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="并行进程数（0=自动取 CPU 核数）")
    args = parser.parse_args()

    process_shapenet_mesh(
        shapenet_dir=args.shapenet_dir,
        category=args.category,
        output_dir=args.output_dir,
        depth=args.depth,
        num_points=args.num_points,
        max_models=args.max_models,
        num_workers=args.num_workers,
    )

    print("\n[完成] 预处理完毕！")
    print(f"  下一步训练：python train.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
