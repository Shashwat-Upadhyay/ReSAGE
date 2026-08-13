import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a single CHW float32 tensor to an HW numpy array for display.
    Clamps to [0, 1] for visualization only — does not modify the tensor.
    """
    arr = tensor.detach().cpu().float()
    if arr.shape[0] == 1:
        arr = arr.squeeze(0)          # [H, W]
    else:
        arr = arr.permute(1, 2, 0)    # [H, W, C]
    return arr.clamp(0.0, 1.0).numpy()


def save_validation_samples(
    noisy_batch:   torch.Tensor,
    output_batch:  torch.Tensor,
    gt_batch:      torch.Tensor,
    save_dir:      str,
    epoch:         int,
    num_samples:   int = 4,
):
    """
    Save a grid of sample comparisons during validation.

    Each row: NoisyLR (128x128) | Restored output (256x256) | GT (256x256)

    Args:
        noisy_batch  : [B, 1, 128, 128] — raw NoisyLR input
        output_batch : [B, 1, 256, 256] — model output (raw, not clamped)
        gt_batch     : [B, 1, 256, 256] — ground truth
        save_dir     : directory to save PNGs
        epoch        : current epoch number (used in filename)
        num_samples  : number of rows to visualise
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(num_samples, noisy_batch.shape[0])

    fig = plt.figure(figsize=(12, 4 * n))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.3, wspace=0.1)

    for i in range(n):
        noisy_img  = _to_numpy_image(noisy_batch[i])
        output_img = _to_numpy_image(output_batch[i])
        gt_img     = _to_numpy_image(gt_batch[i])

        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(noisy_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax0.set_title(f"NoisyLR 128×128", fontsize=9)
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(output_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax1.set_title(f"Restored 256×256", fontsize=9)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[i, 2])
        ax2.imshow(gt_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax2.set_title(f"GT 256×256", fontsize=9)
        ax2.axis("off")

    save_path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
