"""
FractalOctGen 训练脚本

支持：
  - 使用 ShapeNet airplane 预处理数据（--data_dir 指定）
  - 无 ShapeNet 时使用合成数据（--synthetic）
  - AdamW + 余弦学习率 + warmup
  - 混合精度（AMP）
  - 定期 checkpoint 保存

用法：
  # 使用合成数据快速验证
  python train.py --synthetic --epochs 50 --batch_size 8 --output_dir runs/debug

  # 使用 ShapeNet 数据
  python train.py --data_dir data/shapenet_airplane --epochs 300 \
                  --batch_size 16 --output_dir runs/airplane
"""

import argparse
import os
import time
import math
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE = None
from torch.utils.data import DataLoader

from models.fractal_oct_gen import (
    fractal_oct_gen_base, fractal_oct_gen_small, fractal_oct_gen_large,
    count_parameters)
from datasets.shapenet_octree import ShapeNetOctreeDataset, SyntheticOctreeDataset, collate_fn


# ─── 学习率调度（余弦 + warmup）──────────────────────────────────────────────

def cosine_lr_lambda(step: int, total_steps: int, warmup_steps: int,
                     min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return max(min_lr_ratio, cosine)


# ─── 训练单个 epoch ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args,
) -> dict:
    model.train()
    total_loss = 0.0
    level_sums: dict = {}
    n_batches = 0
    t0 = time.time()

    accum = max(1, args.grad_accum)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        # 移到 GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        amp_ctx = autocast(_AMP_DEVICE, enabled=args.amp) if _AMP_DEVICE else autocast(enabled=args.amp)
        with amp_ctx:
            out = model(batch)
            raw_loss = out['loss']
            loss = raw_loss / accum     # 梯度累积：缩放 loss

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 每 accum 个 micro-batch（或 epoch 末尾）更新一次参数
        do_step = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == len(loader))
        if do_step:
            optimizer_stepped = True
            if args.amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # 若本步因 inf/nan 梯度被 scaler 跳过，scale 会被调小；
                # 此时 optimizer 实际未更新，不应推进 LR 调度（也避免警告）
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and optimizer_stepped:
                scheduler.step()

        total_loss += raw_loss.item()
        for k, v in out.items():
            if k.startswith('loss_d'):
                level_sums[k] = level_sums.get(k, 0.0) + v.item()
        n_batches += 1

        if (batch_idx + 1) % args.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0
            level_str = ' '.join(
                f'{k.replace("loss_", "")}={level_sums[k]/n_batches:.3f}'
                for k in sorted(level_sums))
            print(
                f'  Epoch {epoch} [{batch_idx+1}/{len(loader)}] '
                f'loss={total_loss/n_batches:.4f} '
                f'({level_str}) '
                f'lr={lr:.2e} '
                f't={elapsed:.1f}s'
            )

    n = max(n_batches, 1)
    result = {'loss': total_loss / n}
    for k, s in level_sums.items():
        result[k] = s / n
    return result


# ─── 验证 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
) -> dict:
    model.eval()
    total_loss = 0.0
    level_sums: dict = {}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        amp_ctx = autocast(_AMP_DEVICE, enabled=args.amp) if _AMP_DEVICE else autocast(enabled=args.amp)
        with amp_ctx:
            out = model(batch)

        total_loss += out['loss'].item()
        for k, v in out.items():
            if k.startswith('loss_d'):
                level_sums[k] = level_sums.get(k, 0.0) + v.item()
        n_batches += 1

    n = max(n_batches, 1)
    result = {'val_loss': total_loss / n}
    for k, s in level_sums.items():
        result[f'val_{k}'] = s / n
    return result


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FractalOctGen 训练')

    # 数据
    parser.add_argument('--data_dir', type=str, default='',
                        help='预处理数据目录（含 train.txt, *.npz）')
    parser.add_argument('--synthetic', action='store_true',
                        help='使用合成数据（无需 ShapeNet）')
    parser.add_argument('--synthetic_size', type=int, default=512,
                        help='合成数据集大小')
    parser.add_argument('--num_workers', type=int, default=4)

    # 模型
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='模型规模（small=快速调试，base=标准，large=显存充足追求质量）')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='每步 micro-batch 大小（受显存限制时调小）')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='梯度累积步数（有效 batch = batch_size × grad_accum）')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=float, default=15.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true', default=True,
                        help='启用混合精度（AMP）')
    parser.add_argument('--no_amp', dest='amp', action='store_false')

    # 输出
    parser.add_argument('--output_dir', type=str, default='runs/default')
    parser.add_argument('--save_every', type=int, default=25,
                        help='每 N epoch 保存 checkpoint')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default='',
                        help='从 checkpoint 恢复训练')

    args = parser.parse_args()

    # ── 设备 ──────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── 数据集 ────────────────────────────────────────────────────────────────
    if args.synthetic or not args.data_dir:
        print(f'[INFO] 使用合成数据（{args.synthetic_size} 样本）')
        train_dataset = SyntheticOctreeDataset(size=args.synthetic_size)
        val_dataset = SyntheticOctreeDataset(size=max(32, args.synthetic_size // 8),
                                              seed=99)
    else:
        train_dataset = ShapeNetOctreeDataset(args.data_dir, split='train')
        val_dataset = ShapeNetOctreeDataset(args.data_dir, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    print(f'训练集: {len(train_dataset)} 样本，验证集: {len(val_dataset)} 样本')

    # ── 模型 ──────────────────────────────────────────────────────────────────
    if args.model_size == 'small':
        model = fractal_oct_gen_small()
    elif args.model_size == 'large':
        model = fractal_oct_gen_large()
    else:
        model = fractal_oct_gen_base()

    model = model.to(device)
    n_params = count_parameters(model)
    print(f'模型参数量: {n_params:,}（{n_params/1e6:.2f}M）')

    # ── 优化器 + 调度器 ────────────────────────────────────────────────────────
    # AdamW（对 LN/偏置不加 weight decay）
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'norm' in name or 'bias' in name or 'class_emb' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    # 调度器按"优化器更新次数"计步（每 grad_accum 个 micro-batch 更新一次）
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, total_steps, warmup_steps),
    )
    scaler = GradScaler(_AMP_DEVICE, enabled=args.amp) if _AMP_DEVICE else GradScaler(enabled=args.amp)

    # ── 恢复训练 ──────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float('inf')
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        history = ckpt.get('history', [])
        print(f'从 epoch {start_epoch} 恢复训练')

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    print(f'\n开始训练（共 {args.epochs} epochs）...\n')

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch, args
        )

        # 验证
        val_metrics = evaluate(model, val_loader, device, args)

        epoch_time = time.time() - t_epoch
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']

        # 日志
        record = {'epoch': epoch, **train_metrics, **val_metrics,
                  'epoch_time': epoch_time}
        history.append(record)

        print(
            f'Epoch {epoch:03d}/{args.epochs} '
            f'train={train_metrics["loss"]:.4f} '
            f'val={val_metrics["val_loss"]:.4f} '
            f'{"[BEST]" if is_best else ""} '
            f'({epoch_time:.1f}s)'
        )

        # 保存 checkpoint
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
            'args': vars(args),
        }

        torch.save(ckpt, output_dir / 'checkpoint_last.pt')
        if is_best:
            torch.save(ckpt, output_dir / 'checkpoint_best.pt')
            print(f'  → 保存最优模型（val_loss={best_val_loss:.4f}）')

        if epoch % args.save_every == 0:
            torch.save(ckpt, output_dir / f'checkpoint_epoch{epoch:04d}.pt')

        # 保存训练历史
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f'\n训练完成！最优验证损失: {best_val_loss:.4f}')
    print(f'模型保存于: {output_dir}')


if __name__ == '__main__':
    main()
