import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a 2D Gaussian kernel for SSIM computation."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.outer(gauss)
    # Shape: [channels, 1, k, k]
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Compute mean SSIM between pred and target.

    Both tensors must be in [0, 1] for canonical SSIM values.
    For loss computation the caller should clamp the prediction.

    Args:
        pred   : [B, C, H, W] float32
        target : [B, C, H, W] float32

    Returns:
        Scalar SSIM value (mean over batch and channels).
    """
    channels = pred.shape[1]
    kernel = _gaussian_kernel(kernel_size, sigma, channels).to(pred.device)
    padding = kernel_size // 2

    mu_x = F.conv2d(pred,   kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_x_sq  = mu_x * mu_x
    mu_y_sq  = mu_y * mu_y
    mu_xy    = mu_x * mu_y

    sigma_x_sq  = F.conv2d(pred   * pred,   kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq  = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy    = F.conv2d(pred   * target, kernel, padding=padding, groups=channels) - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    """1 - SSIM loss. Clamps prediction to [0, 1] for stable metric computation."""

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp only for SSIM computation — does NOT alter the training graph in a
        # destructive way; the gradient still flows through `pred`.
        pred_clamped = pred.clamp(0.0, 1.0)
        return 1.0 - ssim(pred_clamped, target, self.kernel_size, self.sigma)


class ReSAGELoss(nn.Module):
    """
    Total loss: L1 + lambda_ssim * (1 - SSIM)

    Args:
        l1_weight   (float): Weight for the L1 term. Default 1.0.
        ssim_weight (float): Weight for the SSIM term. Default 0.1.
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.1):
        super().__init__()
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss     = nn.L1Loss()
        self.ssim_loss   = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1   = self.l1_loss(pred, target)
        ssim_l = self.ssim_loss(pred, target)
        total  = self.l1_weight * l1 + self.ssim_weight * ssim_l
        return total, l1, ssim_l
