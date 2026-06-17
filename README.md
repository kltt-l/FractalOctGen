# FractalOctGen：三维分形自回归生成模型

## 项目结构

```
Final_project/
├── models/
│   ├── octree_ar.py         # OctAR Transformer + OccupancyMLP + 窗口注意力 + 3D 位置编码
│   └── fractal_oct_gen.py   # FractalOctGen 主模型（small/base/large 配置）
├── datasets/
│   └── shapenet_octree.py   # 八叉树数据集加载 + collate + 合成数据
├── utils/
│   └── octree_utils.py      # 八叉树构建/序列化/重建工具 + 全局常量
├── preprocess.py            # ShapeNetCore.v2 mesh → 八叉树 .npz（多进程并行）
├── train.py                 # 训练脚本（AMP + 梯度累积 + 余弦调度）
├── generate.py              # 生成 + 体素后处理 + Marching Cubes 导出
├── evaluate.py              # 定量评估（重建 loss / COV / MMD）
├── test_pipeline.py         # 端到端管线自测（无需外部数据）
├── milestone/               # Milestone 报告（.tex / .pdf）
├── requirements.txt
└── README.md
```

## 环境配置

```bash
conda create -n fractal_oct python=3.10 -y
conda activate fractal_oct

# 按 CUDA 版本安装 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 其余依赖
pip install -r requirements.txt
```

> 全局分辨率由 `utils/octree_utils.py` 中的 `OCTREE_DEPTH=7` 决定（占据分辨率 64³）。如需更改分辨率，改这一个常量即可，模型/数据/生成会自动适配。

## 1. 数据准备

从 [shapenet.org](https://shapenet.org) 获取 `ShapeNetCore.v2`，目录形如 `ShapeNetCore.v2/02691156/<model_id>/models/model_normalized.obj`。运行预处理（多进程并行）：

```bash
python preprocess.py \
    --shapenet_dir data/ShapeNetCore.v2 \
    --category airplane \
    --num_points 100000 \
    --output_dir data/shapenet_airplane_processed \
    --num_workers 16
```

输出：`data/shapenet_airplane_processed/` 下的 `*.npz`（每个形状一份八叉树数据）和 `train.txt / val.txt / test.txt`（8:1:1 随机划分）。已存在的 `.npz` 会自动跳过，可安全断点续跑。

**常用参数**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--shapenet_dir` | （必填） | ShapeNetCore.v2 根目录（含 synset 子目录） |
| `--category` | `airplane` | 类别：`airplane/car/chair/rifle/table` |
| `--num_points` | `100000` | 网格表面采样点数；深八叉树建议 ≥ 100000，过低会导致细节缺失 |
| `--num_workers` | `0`（自动） | 并行进程数，建议设为物理核数；显著加速（10–25×） |
| `--depth` | `7` | 八叉树深度（一般保持与 `OCTREE_DEPTH` 一致） |
| `--max_models` | `0` | 仅处理前 N 个模型（调试用，`0`=全部） |

## 2. 训练

```bash
# 正式训练（ShapeNet airplane，base 配置）
python train.py \
    --data_dir data/shapenet_airplane_processed \
    --model_size base \
    --batch_size 4 --grad_accum 4 \
    --output_dir runs/airplane_v2

# 无数据时用合成数据快速验证管线（small 配置）
python train.py --synthetic --model_size small \
    --epochs 50 --batch_size 4 --output_dir runs/debug
```

- **指定 GPU**：代码用默认 `cuda` 设备，通过环境变量选择卡，例如 Linux 下 `CUDA_VISIBLE_DEVICES=7 python train.py ...`。
- **显存不足（OOM）**：减小 `--batch_size`、增大 `--grad_accum`（二者乘积=有效 batch），并设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。例如 `--batch_size 2 --grad_accum 8`。
- **断点续训**：加 `--resume runs/airplane_v2/checkpoint_last.pt`。

**常用参数**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data_dir` | — | 预处理数据目录；不指定或加 `--synthetic` 则用合成数据 |
| `--model_size` | `base` | `small`(~1M，调试) / `base`(~132M) / `large`(~413M)；**生成/评估时须与训练一致** |
| `--batch_size` | `4` | 每步 micro-batch 大小（受显存限制时调小） |
| `--grad_accum` | `4` | 梯度累积步数；**有效 batch = batch_size × grad_accum** |
| `--epochs` | `400` | 训练轮数 |
| `--lr` | `3e-4` | 学习率（余弦调度 + warmup） |
| `--warmup_epochs` | `15` | warmup 轮数 |
| `--weight_decay` | `0.05` | AdamW 权重衰减（对 LN/bias 不施加） |
| `--grad_clip` | `1.0` | 梯度裁剪范数 |
| `--amp` / `--no_amp` | 开启 | 混合精度训练 |
| `--num_workers` | `4` | DataLoader 进程数 |
| `--save_every` | `25` | 每 N 轮存一次带轮次的 checkpoint |

训练在 `--output_dir` 下保存 `checkpoint_last.pt`、`checkpoint_best.pt`（按验证损失）、`history.json`、`args.json`。日志逐层打印 `d2…d6` 的 BCE 损失，其中 `loss_d6`（最细层）最难收敛，可重点观察。

## 3. 生成

```bash
python generate.py \
    --checkpoint runs/airplane_v2/checkpoint_best.pt \
    --model_size base \
    --num_samples 16 --batch_size 4 \
    --output_dir outputs/airplane_v2 \
    --temperature_l0 0.9 --temperature_l1 0.9 --temperature_l2 0.0
```

每个形状输出 `shape_XXXX.npy`（64³ 体素）和 `shape_XXXX.obj`（三角网格）。

**采样温度**（控制随机性与表面质量）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--temperature_l0` | `1.0` | 第一层 AR（depth-2，整体轮廓）；建议 `0.8~1.0` 保多样性 |
| `--temperature_l1` | `1.0` | 其余 AR 层（depth-3）；建议 `0.8~1.0` |
| `--temperature_l2` | `1.0` | **占据层（depth 4/5/6）**；**强烈建议 `0.0`**（确定性阈值），否则逐体素掷骰子会产生大量孔洞 |

**体素后处理 / 表面提取**（缓解薄壳导致的双壁、孔洞、抖动）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--closing_iters` | `1` | 形态学闭运算迭代次数（封针孔）；残洞多可设 2，过大会粘连薄机翼 |
| `--smooth_iters` | `10` | Taubin 网格平滑迭代（不收缩体积）；`0` 关闭 |
| `--no_solidify` | 关 | 加上则**关闭**实体化填充（保留空心薄壳） |
| `--no_keep_largest` | 关 | 加上则**保留**漂浮碎块（默认只保留最大连通分量） |
| `--resolution` | `0`（自动） | 体素网格分辨率，默认取 `2^finest_depth=64` |

> 快速冒烟测试（随机权重、不需 checkpoint）：`python generate.py --random_model --num_samples 4 --output_dir outputs/test`

## 4. 定量评估

```bash
# 重建损失（快）+ 生成质量 COV/MMD（较慢）
python evaluate.py \
    --checkpoint runs/airplane_v2/checkpoint_best.pt \
    --model_size base \
    --data_dir data/shapenet_airplane_processed \
    --mode all --num_gen 200
```

`--mode` 可选 `recon`（仅测试集逐层 BCE）/ `gen`（仅生成并算 COV、MMD、平均占据率）/ `all`。`--model_size` 须与训练一致。

## 5. 管线自测

```bash
python test_pipeline.py
```

不依赖外部数据，依次验证：八叉树构建、位置编码、OctAR、OccupancyMLP、完整前向/反向、采样、体素转网格。用于改动代码后的快速回归检查。