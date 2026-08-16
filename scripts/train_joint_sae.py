
import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "/kaggle/working/ReSAGE")

from src.datasets.dataset import ReSAGEDataset
from src.models.sae_restoration import ReSAGESAERestoration
from src.losses.losses import combined_loss
from src.utils.metrics import compute_psnr, compute_ssim


# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = (
    "/kaggle/input/datasets/shashwatupadhyay47/"
    "resage-data/ReSAGE_data/data - Copy"
)

BASELINE_CHECKPOINT = (
    "/kaggle/input/datasets/shashwatupadhyay47/"
    "baseline/baseline_best.pth"
)

SAE_CHECKPOINT = (
    "/kaggle/working/ReSAGE/checkpoints/sae_best.pth"
)

CHECKPOINT_DIR = (
    "/kaggle/working/ReSAGE/checkpoints"
)

JOINT_BEST = os.path.join(
    CHECKPOINT_DIR,
    "joint_sae_best.pth"
)

JOINT_LATEST = os.path.join(
    CHECKPOINT_DIR,
    "joint_sae_latest.pth"
)

SEED = 42
VAL_FRACTION = 0.10

BATCH_SIZE = 16
EPOCHS = 20

# Lower LR because both models are already pretrained
LEARNING_RATE = 1e-4

NUM_WORKERS = 2


# =========================================================
# REPRODUCIBILITY
# =========================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# EXACT SAME TRAIN/VAL SPLIT
# =========================================================

def build_train_val_split(
    dataset_len,
    val_frac,
    seed
):

    rng = np.random.default_rng(seed)

    indices = np.arange(dataset_len)

    rng.shuffle(indices)

    n_val = max(
        1,
        int(dataset_len * val_frac)
    )

    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    return train_idx, val_idx


# =========================================================
# DATASET
# =========================================================

dataset = ReSAGEDataset(
    root_dir=DATA_ROOT,
    split="train"
)

train_idx, val_idx = build_train_val_split(
    len(dataset),
    VAL_FRACTION,
    SEED
)

train_dataset = Subset(
    dataset,
    train_idx
)

val_dataset = Subset(
    dataset,
    val_idx
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nDataset:", len(dataset))
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))


# =========================================================
# MODEL
# =========================================================

model = ReSAGESAERestoration(
    in_channels=1,
    out_channels=1,
    mid_channels=64,
    num_residual_blocks=4,
    upscale_factor=2,
    latent_dim=128,
    freeze_sae=False
).to(device)


# =========================================================
# LOAD PRETRAINED BASELINE
# =========================================================

print("\nLoading baseline checkpoint...")

baseline_ckpt = torch.load(
    BASELINE_CHECKPOINT,
    map_location=device
)

model.baseline.load_state_dict(
    baseline_ckpt["model"]
)

print(
    "Baseline checkpoint loaded."
)

print(
    "Previous best PSNR:",
    baseline_ckpt.get(
        "best_psnr",
        "N/A"
    )
)


# =========================================================
# LOAD PRETRAINED SAE
# =========================================================

print("\nLoading SAE checkpoint...")

sae_ckpt = torch.load(
    SAE_CHECKPOINT,
    map_location=device
)

model.sae.load_state_dict(
    sae_ckpt["model"]
)

print(
    "SAE checkpoint loaded."
)


# =========================================================
# PARAMETER CHECK
# =========================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\nTotal parameters:", total_params)
print("Trainable parameters:", trainable_params)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# =========================================================
# TRAINING
# =========================================================

best_psnr = -float("inf")

print("\nStarting joint SAE training...")
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)


for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss = 0.0

    for noisy, gt in train_loader:

        noisy = noisy.to(
            device,
            non_blocking=True
        )

        gt = gt.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        output = model(noisy)

        # Use the same restoration loss as baseline
        loss = combined_loss(
            output,
            gt
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    psnr_values = []
    ssim_values = []
    val_loss = 0.0

    with torch.no_grad():

        for noisy, gt in val_loader:

            noisy = noisy.to(
                device,
                non_blocking=True
            )

            gt = gt.to(
                device,
                non_blocking=True
            )

            output = model(noisy)

            loss = combined_loss(
                output,
                gt
            )

            val_loss += loss.item()

            output_metric = torch.clamp(
                output,
                0.0,
                1.0
            )

            gt_metric = torch.clamp(
                gt,
                0.0,
                1.0
            )

            for i in range(output.shape[0]):

                psnr_values.append(
                    float(
                        compute_psnr(
                            output_metric[i:i+1],
                            gt_metric[i:i+1]
                        )
                    )
                )

                ssim_values.append(
                    float(
                        compute_ssim(
                            output_metric[i:i+1],
                            gt_metric[i:i+1]
                        )
                    )
                )

    val_loss /= len(val_loader)

    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]


    # =====================================================
    # LOG
    # =====================================================

    print(
        f"\nEpoch [{epoch:02d}/{EPOCHS}] "
        f"train_loss={train_loss:.6f} "
        f"val_loss={val_loss:.6f} "
        f"PSNR={mean_psnr:.4f} dB "
        f"SSIM={mean_ssim:.4f} "
        f"lr={current_lr:.2e}"
    )


    # =====================================================
    # SAVE LATEST
    # =====================================================

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_psnr": best_psnr
    }

    torch.save(
        checkpoint,
        JOINT_LATEST
    )


    # =====================================================
    # SAVE BEST
    # =====================================================

    if mean_psnr > best_psnr:

        best_psnr = mean_psnr

        checkpoint["best_psnr"] = best_psnr

        torch.save(
            checkpoint,
            JOINT_BEST
        )

        print(
            f"  Best checkpoint saved "
            f"(PSNR={best_psnr:.4f} dB)"
        )


print("\n" + "=" * 60)
print("JOINT SAE TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best validation PSNR: "
    f"{best_psnr:.4f} dB"
)

print(
    f"Best checkpoint: {JOINT_BEST}"
)
