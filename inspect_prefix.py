"""Export coarse prefix octree levels for visual inspection.

This script intentionally avoids importing project modules that depend on ocnn.
It reads preprocessed .npz files and exports the split==1 cells at selected
depths as box meshes in the same normalized [0, 1]^3 coordinate frame.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def _level_boxes(split: np.ndarray, xyz: np.ndarray, depth: int) -> trimesh.Trimesh:
    occ = np.asarray(split).astype(bool)
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.shape[0] != occ.shape[0]:
        raise ValueError(f'd{depth}: xyz length {xyz.shape[0]} != split length {occ.shape[0]}')

    grid = 2 ** depth
    max_coord = float(grid - 1)
    ijk = np.rint(xyz[occ] * max_coord).astype(np.int64)
    if len(ijk) == 0:
        return trimesh.Trimesh()

    cell_size = 1.0 / float(grid)
    meshes = []
    for c in ijk:
        center = (c.astype(np.float64) + 0.5) * cell_size
        box = trimesh.creation.box(extents=(cell_size, cell_size, cell_size))
        box.apply_translation(center)
        meshes.append(box)
    return trimesh.util.concatenate(meshes)


def export_prefix_levels(npz_path: Path, out_dir: Path, depths: list[int]) -> None:
    data = np.load(npz_path, allow_pickle=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'prefix: {npz_path}')
    for d in depths:
        split_key = f'split_{d}'
        xyz_key = f'xyz_{d}'
        if split_key not in data.files or xyz_key not in data.files:
            print(f'[skip] d{d}: missing {split_key} or {xyz_key}')
            continue

        split = data[split_key]
        xyz = data[xyz_key]
        n_total = int(len(split))
        n_occ = int(np.asarray(split).sum())
        cell_size = 1.0 / float(2 ** d)
        print(f'd{d}: split==1 {n_occ}/{n_total}, cell_size={cell_size:.6f}')

        mesh = _level_boxes(split, xyz, d)
        out_path = out_dir / f'{npz_path.stem}_d{d}_prefix.obj'
        mesh.export(out_path)
        print(f'  exported: {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Export d2/d3 prefix cells as OBJ boxes.')
    parser.add_argument('npz', type=str, help='Prefix .npz path')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 3],
                        help='Depths to export, default: 2 3')
    parser.add_argument('--out_dir', type=str, default='outputs/prefix_inspect',
                        help='Output directory')
    args = parser.parse_args()

    export_prefix_levels(Path(args.npz), Path(args.out_dir), args.depths)


if __name__ == '__main__':
    main()
