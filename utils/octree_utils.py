"""
八叉树序列化工具
将 ocnn.octree.Octree 对象转换为适合自回归模型训练的分裂标签序列，
以及从分裂标签序列重建八叉树。

主要函数：
  - build_octree_from_mesh: 从三角网格构建八叉树
  - extract_octree_data: 从 Octree 对象提取各层分裂标签和坐标
  - compute_parent_indices: 计算子节点到父节点的映射索引
  - reconstruct_octree: 从预测的分裂标签序列重建八叉树
"""

import numpy as np
import torch
import ocnn
from ocnn.octree import Octree, Points, key2xyz


# ─── 全局参数 ──────────────────────────────────────────────────────────────────
OCTREE_DEPTH = 7       # 八叉树最大深度（分辨率 2^7 = 128，有效占据分辨率 2^6 = 64³）
FULL_DEPTH = 2         # 全量展开深度（depth ≤ full_depth 的层全部展开）
SAMPLE_POINTS = 100000  # 从网格采样点数（深八叉树需要高密度点云才能填满细节）

# ─── 分层架构配置 ────────────────────────────────────────────────────────────────
# FractalOctGen 把八叉树的每一层分裂预测拆分为若干"分形层级"：
#   * 前 NUM_AR_LEVELS 层（较粗）使用自回归 Transformer（OctAR）——逐节点生成，
#     能建模兄弟节点之间的强相关（决定整体轮廓：机身/机翼/尾翼）。
#   * 其余更细的层使用并行占据 MLP（OccupancyMLP）——给定父层条件 + 位置一次性
#     预测所有节点占据，速度快，负责把表面逐级细化到 64³。
#
# 对 OCTREE_DEPTH=7、FULL_DEPTH=2、NUM_AR_LEVELS=2：
#   AR_DEPTHS  = [2, 3]        （depth-2→3、depth-3→4 的分裂，8³ 粗结构）
#   OCC_DEPTHS = [4, 5, 6]     （depth-4/5/6 的占据，逐级细化到 64³）
#   ALL_PRED_DEPTHS = [2,3,4,5,6]
NUM_AR_LEVELS = 2

def _derive_level_depths(full_depth: int = FULL_DEPTH,
                         octree_depth: int = OCTREE_DEPTH,
                         num_ar_levels: int = NUM_AR_LEVELS):
    """返回 (ar_depths, occ_depths, all_pred_depths)。

    预测层覆盖 depth ∈ [full_depth, octree_depth-1]（每层预测它是否向下分裂）。
    其中前 num_ar_levels 个为自回归层，其余为占据 MLP 层。
    """
    all_pred = list(range(full_depth, octree_depth))      # 例如 [2,3,4,5,6]
    ar = all_pred[:num_ar_levels]                          # [2,3]
    occ = all_pred[num_ar_levels:]                         # [4,5,6]
    return ar, occ, all_pred

AR_DEPTHS, OCC_DEPTHS, ALL_PRED_DEPTHS = _derive_level_depths()

# 旧常量保留以兼容（每层处理的 (父深度, 子深度)）
LEVEL_DEPTHS = [(d, d + 1) for d in ALL_PRED_DEPTHS]


def build_octree_from_mesh(mesh, depth: int = OCTREE_DEPTH,
                           full_depth: int = FULL_DEPTH,
                           num_points: int = SAMPLE_POINTS) -> Octree:
    """
    从 trimesh.Trimesh 网格构建 ocnn.Octree。

    流程：采样表面点 + 法向 → 归一化到单位球 → 构建八叉树

    Args:
        mesh: trimesh.Trimesh 对象
        depth: 八叉树最大深度
        full_depth: 全量展开深度
        num_points: 采样点数

    Returns:
        ocnn.octree.Octree 对象
    """
    import trimesh

    # 采样表面点和法向量
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    pts = pts.astype(np.float32)
    normals = mesh.face_normals[face_idx].astype(np.float32)

    # 归一化到 [-1, 1]^3（单位球内）
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    pts -= center
    scale = np.abs(pts).max()
    if scale > 0:
        pts = pts / scale * 0.9  # 稍微缩小，留边距

    # 法向量归一化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    normals = normals / norms

    # 构建 ocnn.Points
    points_obj = Points(
        points=torch.from_numpy(pts),
        normals=torch.from_numpy(normals),
    )
    points_obj.check_input()

    # 构建八叉树
    octree = Octree(depth=depth, full_depth=full_depth)
    octree.build_octree(points_obj)
    return octree


def extract_octree_data(octree: Octree) -> dict:
    """
    从 ocnn.Octree 提取训练所需的数据：各层的分裂标签、z-order 坐标。

    Returns:
        dict，包含：
          - 'nnum': 各层节点数（长度 = depth+1）
          - 'split_{d}': depth d 的分裂标签（0=叶节点，1=分裂），shape [nnum[d]]
          - 'keys_{d}': depth d 的 z-order Morton 码，shape [nnum[d]]
          - 'xyz_{d}': depth d 节点的归一化 3D 坐标 [0,1]^3，shape [nnum[d}, 3]
    """
    data = {}
    depth = octree.depth
    full_depth = octree.full_depth

    # 节点计数
    data['nnum'] = octree.nnum.numpy().copy()
    data['full_depth'] = full_depth
    data['depth'] = depth

    # 提取各层数据（从 full_depth 到 depth-1 提取分裂标签）
    for d in range(full_depth, depth):
        n_d = int(octree.nnum[d].item())
        if n_d == 0:
            data[f'split_{d}'] = np.zeros(0, dtype=np.int8)
            data[f'keys_{d}'] = np.zeros(0, dtype=np.int64)
            data[f'xyz_{d}'] = np.zeros((0, 3), dtype=np.float32)
            continue

        # 分裂标签
        children_d = octree.children[d]  # [n_d], int32
        split_d = (children_d >= 0).numpy().astype(np.int8)  # 1=分裂, 0=叶
        data[f'split_{d}'] = split_d

        # z-order keys
        keys_d = octree.keys[d]  # [n_d], int64 Morton codes
        data[f'keys_{d}'] = keys_d.numpy().astype(np.int64)

        # 3D 坐标（归一化）
        xyz_d = _keys_to_xyz(keys_d, depth=d)  # [n_d, 3] float32
        data[f'xyz_{d}'] = xyz_d

    return data


def _keys_to_xyz(keys: torch.Tensor, depth: int) -> np.ndarray:
    """
    将 Morton code keys 解码为归一化 [0,1]^3 坐标。

    key2xyz 返回 (x, y, z, batch_id) 四个分量，各元素为整数坐标（范围 [0, 2^depth-1]）。
    我们对返回结果做 stacking 并归一化。
    """
    x, y, z, _ = key2xyz(keys, depth=depth)
    xyz = torch.stack([x, y, z], dim=-1).float()
    # 归一化：整数坐标 [0, 2^depth - 1] → [0, 1]
    max_coord = float(2 ** depth - 1) if depth > 0 else 1.0
    xyz = xyz / max_coord
    return xyz.numpy()


def compute_parent_indices(octree: Octree, parent_depth: int) -> np.ndarray:
    """
    为 depth=(parent_depth+1) 的每个节点，计算其在 depth=parent_depth 的父节点索引。

    在 ocnn 中，若 children[d][i] = c（c >= 0），则 depth-(d+1) 的第 c 到 c+7 个节点
    是 depth-d 第 i 个节点的 8 个子节点（z-order 排列）。

    Args:
        octree: ocnn.Octree
        parent_depth: 父节点深度

    Returns:
        parent_idx: np.ndarray, shape [nnum[parent_depth+1]], dtype int32
                    parent_idx[j] = 父节点在 depth=parent_depth 序列中的索引
    """
    children_d = octree.children[parent_depth]  # [N_d]
    n_child_depth = int(octree.nnum[parent_depth + 1].item())

    if n_child_depth == 0:
        return np.zeros(0, dtype=np.int32)

    parent_idx = np.empty(n_child_depth, dtype=np.int32)

    valid_mask = (children_d >= 0).numpy()
    split_node_indices = np.where(valid_mask)[0]  # [n_split]
    child_starts = children_d[valid_mask].numpy()  # [n_split], first child index

    # 每个分裂节点贡献连续的 8 个子节点
    for local_i, (parent_i, c_start) in enumerate(zip(split_node_indices, child_starts)):
        parent_idx[c_start: c_start + 8] = parent_i

    return parent_idx


def save_octree_data(octree: Octree, filepath: str) -> None:
    """提取并保存八叉树数据到 .npz 文件。"""
    data = extract_octree_data(octree)

    # 额外保存各层的父节点索引（方便 Dataset 直接加载）
    depth = int(data['depth'])
    full_depth = int(data['full_depth'])
    for d in range(full_depth, depth - 1):
        parent_idx = compute_parent_indices(octree, parent_depth=d)
        data[f'parent_idx_{d+1}'] = parent_idx  # depth-(d+1) 节点的父索引

    np.savez_compressed(filepath, **data)


def load_octree_data(filepath: str) -> dict:
    """从 .npz 文件加载八叉树数据，返回纯 Python/numpy dict。"""
    npz = np.load(filepath, allow_pickle=False)
    data = {k: npz[k] for k in npz.files}
    return data


# ─── 推理用：从预测分裂标签重建八叉树 ─────────────────────────────────────────────

def reconstruct_octree(split_labels: dict, depth: int = OCTREE_DEPTH,
                       full_depth: int = FULL_DEPTH,
                       threshold: float = 0.5) -> Octree:
    """
    从各层的预测分裂标签（logits 或概率）重建 ocnn.Octree。

    Args:
        split_labels: dict，键为 'split_{d}'（d = full_depth..depth-1），
                      值为对应层的 logits/概率 tensor 或 numpy array，shape [N_d]。
                      如果传入 logits（未经 sigmoid），threshold 在 logit 空间比较（>0 为 split）。
                      也支持传入 binary 0/1 标签。
        depth: 目标八叉树深度
        full_depth: 全量展开深度
        threshold: 分裂决策阈值（对 logit 使用 0.0，对概率使用 0.5）

    Returns:
        重建的 ocnn.Octree
    """
    octree = Octree(depth=depth, full_depth=full_depth)
    octree.octree_grow_full(depth=full_depth)

    for d in range(full_depth, depth):
        key = f'split_{d}'
        if key not in split_labels:
            break

        raw = split_labels[key]
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu()
            labels = (raw > threshold).long()
        else:
            raw = np.asarray(raw)
            labels = torch.from_numpy((raw > threshold).astype(np.int64))

        n_expected = int(octree.nnum[d].item())
        if len(labels) < n_expected:
            # 用 0（叶节点）填充不足部分
            pad = torch.zeros(n_expected - len(labels), dtype=torch.long)
            labels = torch.cat([labels, pad])
        elif len(labels) > n_expected:
            labels = labels[:n_expected]

        octree.octree_split(labels.int(), depth=d)
        octree.octree_grow(depth=d + 1)

    return octree


def octree_to_voxel_grid(octree: Octree, resolution: int = None) -> np.ndarray:
    """
    将八叉树叶节点转换为体素网格（占据 0/1 矩阵）。

    Args:
        octree: 已构建的 ocnn.Octree
        resolution: 输出分辨率（默认 2^depth）

    Returns:
        voxel_grid: shape [res, res, res], dtype bool
    """
    depth = int(octree.depth)
    if resolution is None:
        resolution = 2 ** depth

    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    # 遍历最深层的所有节点
    keys_leaf = octree.keys[depth]
    if len(keys_leaf) == 0:
        return voxel_grid

    x, y, z, _ = key2xyz(keys_leaf, depth=depth)
    x = x.numpy().astype(int)
    y = y.numpy().astype(int)
    z = z.numpy().astype(int)

    # 裁剪到 [0, resolution-1]
    mask = (x >= 0) & (x < resolution) & (y >= 0) & (y < resolution) & (z >= 0) & (z < resolution)
    voxel_grid[x[mask], y[mask], z[mask]] = True
    return voxel_grid
