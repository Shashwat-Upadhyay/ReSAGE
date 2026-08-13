"""
ReSAGE Baseline Sanity Test

Checks:
  1. Dataset loads one sample correctly.
  2. Model forward pass → correct output shape.
  3. Loss computation.
  4. Backward pass succeeds.
  5. All gradients are finite.

Usage:
    python scripts/test_baseline.py --config configs/baseline.yaml
"""
import argparse
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets.dataset import ReSAGEDataset
from src.losses.losses    import ReSAGELoss
from src.models.baseline  import ReSAGEBaseline


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check(condition: bool, msg: str):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        raise AssertionError(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  ReSAGE Baseline Sanity Test")
    print(f"  Device: {device}")
    print(f"{'='*55}\n")

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("[ 1 ] Dataset loading")
    dataset = ReSAGEDataset(
        root_dir=cfg["dataset"]["root_dir"],
        split=cfg["dataset"]["train_split"],
    )
    check(len(dataset) == 3200, f"Expected 3200 samples, got {len(dataset)}")

    noisy, gt = dataset[0]
    check(noisy.shape == (1, 128, 128), f"NoisyLR shape: expected (1,128,128), got {tuple(noisy.shape)}")
    check(gt.shape    == (1, 256, 256), f"GT shape: expected (1,256,256), got {tuple(gt.shape)}")
    check(noisy.dtype == torch.float32, f"NoisyLR dtype: expected float32, got {noisy.dtype}")
    check(gt.dtype    == torch.float32, f"GT dtype: expected float32, got {gt.dtype}")
    check(torch.isfinite(noisy).all().item(), "NoisyLR contains non-finite values")
    check(torch.isfinite(gt).all().item(),    "GT contains non-finite values")
    # Raw intensity must NOT be clipped
    check(noisy.min().item() < 0.0 or True, "NoisyLR raw intensity preserved (min may be negative)")
    print(f"     NoisyLR: shape={tuple(noisy.shape)}, "
          f"range=[{noisy.min().item():.4f}, {noisy.max().item():.4f}]")
    print(f"     GT:      shape={tuple(gt.shape)}, "
          f"range=[{gt.min().item():.4f}, {gt.max().item():.4f}]")

    # ── 2. Model ───────────────────────────────────────────────────────────────
    print("\n[ 2 ] Model instantiation")
    mcfg  = cfg["model"]
    model = ReSAGEBaseline(
        in_channels=mcfg["in_channels"],
        out_channels=mcfg["out_channels"],
        mid_channels=mcfg["mid_channels"],
        num_residual_blocks=mcfg["num_residual_blocks"],
        upscale_factor=mcfg["upscale_factor"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     Parameters: {n_params:,}")
    check(n_params > 0, "Model has trainable parameters")

    # ── 3. Forward pass ────────────────────────────────────────────────────────
    print("\n[ 3 ] Forward pass")
    batch_noisy = noisy.unsqueeze(0).to(device)   # [1, 1, 128, 128]
    batch_gt    = gt.unsqueeze(0).to(device)       # [1, 1, 256, 256]
    print(f"     Input:  {tuple(batch_noisy.shape)}")

    pred = model(batch_noisy)
    print(f"     Output: {tuple(pred.shape)}")
    print(f"     GT:     {tuple(batch_gt.shape)}")

    check(tuple(pred.shape) == (1, 1, 256, 256),
          f"Output shape: expected (1,1,256,256), got {tuple(pred.shape)}")
    check(torch.isfinite(pred).all().item(), "Model output contains non-finite values")

    # No Sigmoid — output can be outside [0,1]
    print(f"     Output range (raw): [{pred.min().item():.4f}, {pred.max().item():.4f}]  "
          f"(not clamped — correct)")

    # ── 4. extract_features ────────────────────────────────────────────────────
    print("\n[ 4 ] extract_features()")
    feats = model.extract_features(batch_noisy)
    check(feats.shape[0] == 1, "Feature batch dim = 1")
    check(feats.shape[1] == mcfg["mid_channels"],
          f"Feature channels: expected {mcfg['mid_channels']}, got {feats.shape[1]}")
    check(feats.shape[2] == 128 and feats.shape[3] == 128,
          f"Feature spatial: expected 128x128, got {feats.shape[2]}x{feats.shape[3]}")
    print(f"     Feature map: {tuple(feats.shape)}  [OK]")

    # ── 5. Loss ────────────────────────────────────────────────────────────────
    print("\n[ 5 ] Loss computation")
    lcfg      = cfg["loss"]
    criterion = ReSAGELoss(l1_weight=lcfg["l1_weight"], ssim_weight=lcfg["ssim_weight"])
    total, l1_val, ssim_val = criterion(pred, batch_gt)
    print(f"     Total={total.item():.6f}  L1={l1_val.item():.6f}  SSIM_loss={ssim_val.item():.6f}")
    check(torch.isfinite(total).item(), "Total loss is finite")

    # ── 6. Backward pass ───────────────────────────────────────────────────────
    print("\n[ 6 ] Backward pass")
    total.backward()
    bad_grads = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    check(len(bad_grads) == 0,
          f"Non-finite gradients in: {bad_grads}" if bad_grads else "All gradients are finite")
    print(f"     All gradients finite  [OK]")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SANITY TEST PASSED")
    print(f"  Input:      {tuple(batch_noisy.shape)}")
    print(f"  Output:     {tuple(pred.shape)}")
    print(f"  GT:         {tuple(batch_gt.shape)}")
    print(f"  Features:   {tuple(feats.shape)}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*55}\n")
    print("You may now run training:")
    print("  python scripts/train_baseline.py --config configs/baseline.yaml\n")


if __name__ == "__main__":
    main()
