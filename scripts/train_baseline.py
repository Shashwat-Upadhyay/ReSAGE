"""
ReSAGE Baseline Training Script
Usage:
    python scripts/train_baseline.py --config configs/baseline.yaml
    python scripts/train_baseline.py --config configs/baseline.yaml --resume
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

# ── project imports ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets.dataset    import ReSAGEDataset
from src.losses.losses       import ReSAGELoss
from src.models.baseline     import ReSAGEBaseline
from src.utils.metrics       import MetricsTracker
from src.utils.visualization import save_validation_samples


# ── helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_train_val_split(dataset_len: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_len)
    rng.shuffle(indices)
    n_val = max(1, int(dataset_len * val_frac))
    val_idx   = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return train_idx, val_idx


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", 0.0)


def log_json(path: str, entry: dict):
    """Append one JSON object per line (jsonl)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_l1   = 0.0
    total_ssim = 0.0

    for noisy, gt in loader:
        noisy = noisy.to(device, non_blocking=True)
        gt    = gt.to(device,    non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred             = model(noisy)
            loss, l1, ssim_l = criterion(pred, gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_l1   += l1.item()
        total_ssim += ssim_l.item()

    n = len(loader)
    return total_loss / n, total_l1 / n, total_ssim / n


@torch.no_grad()
def validate(model, loader, criterion, metrics_tracker, device, vis_dir, epoch, num_vis=4):
    model.eval()
    metrics_tracker.reset()
    total_loss = 0.0
    vis_saved  = False

    for noisy, gt in loader:
        noisy = noisy.to(device, non_blocking=True)
        gt    = gt.to(device,    non_blocking=True)

        pred             = model(noisy)
        loss, _, _       = criterion(pred, gt)
        total_loss      += loss.item()

        metrics_tracker.update(pred, gt)

        # Save first batch for visualization
        if not vis_saved and vis_dir:
            save_validation_samples(noisy, pred, gt, vis_dir, epoch, num_samples=num_vis)
            vis_saved = True

    n      = len(loader)
    result = metrics_tracker.result()
    return total_loss / n, result["psnr"], result["ssim"]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── dataset ───────────────────────────────────────────────────────────────
    full_dataset = ReSAGEDataset(
        root_dir=cfg["dataset"]["root_dir"],
        split=cfg["dataset"]["train_split"],
    )
    train_idx, val_idx = build_train_val_split(
        len(full_dataset),
        cfg["dataset"]["validation_split"],
        cfg["training"]["seed"],
    )

    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    num_workers = cfg["dataset"]["num_workers"]
    train_loader = DataLoader(
        train_set, batch_size=cfg["training"]["batch_size"],
        shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    mcfg  = cfg["model"]
    model = ReSAGEBaseline(
        in_channels=mcfg["in_channels"],
        out_channels=mcfg["out_channels"],
        mid_channels=mcfg["mid_channels"],
        num_residual_blocks=mcfg["num_residual_blocks"],
        upscale_factor=mcfg["upscale_factor"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── loss / optim / scheduler ──────────────────────────────────────────────
    lcfg      = cfg["loss"]
    criterion = ReSAGELoss(l1_weight=lcfg["l1_weight"], ssim_weight=lcfg["ssim_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── resume ─────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_psnr   = 0.0
    latest_ckpt = cfg["training"]["latest_checkpoint_path"]
    best_ckpt   = cfg["training"]["best_checkpoint_path"]

    if args.resume and os.path.exists(latest_ckpt):
        start_epoch, best_psnr = load_checkpoint(latest_ckpt, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.4f} dB")

    # ── directories ───────────────────────────────────────────────────────────
    vis_dir  = os.path.join(ROOT, "outputs", "baseline")
    log_path = os.path.join(ROOT, "experiments", "baseline", "training_log.jsonl")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ── training loop ─────────────────────────────────────────────────────────
    metrics_tracker = MetricsTracker()

    for epoch in range(start_epoch + 1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_l1, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, metrics_tracker, device, vis_dir, epoch
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{cfg['training']['epochs']}] "
            f"train_loss={train_loss:.4f} (L1={train_l1:.4f} SSIM={train_ssim:.4f}) | "
            f"val_loss={val_loss:.4f} PSNR={val_psnr:.2f}dB SSIM={val_ssim:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        # ── checkpoint ────────────────────────────────────────────────────────
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
        }
        save_checkpoint(state, latest_ckpt)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(state, best_ckpt)
            print(f"  [*] New best PSNR: {best_psnr:.4f} dB -- checkpoint saved.")

        # ── log ───────────────────────────────────────────────────────────────
        log_json(log_path, {
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_l1":   train_l1,
            "train_ssim": train_ssim,
            "val_loss":   val_loss,
            "val_psnr":   val_psnr,
            "val_ssim":   val_ssim,
            "lr":         scheduler.get_last_lr()[0],
            "elapsed_s":  elapsed,
        })

    print(f"\nTraining complete. Best Val PSNR: {best_psnr:.4f} dB")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
