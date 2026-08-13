import torch
import torch.nn.functional as F


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int, device) -> torch.Tensor:
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.outer(gauss)
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Prediction is clamped to [0, data_range] before computation to match GT range.
    Both tensors must be [B, C, H, W] float32.

    Returns:
        float: Mean PSNR (dB) over the batch.
    """
    pred_c = pred.clamp(0.0, data_range)
    mse = F.mse_loss(pred_c, target, reduction="none")
    # Mean over C, H, W; then mean over batch
    mse_per_image = mse.mean(dim=[1, 2, 3])
    psnr_per_image = 10.0 * torch.log10((data_range ** 2) / (mse_per_image + 1e-8))
    return psnr_per_image.mean().item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Structural Similarity Index (SSIM).

    Prediction is clamped to [0, data_range] before computation.
    Both tensors must be [B, C, H, W] float32.

    Returns:
        float: Mean SSIM over the batch.
    """
    pred_c   = pred.clamp(0.0, data_range)
    channels = pred_c.shape[1]
    kernel   = _gaussian_kernel(kernel_size, sigma, channels, pred_c.device)
    padding  = kernel_size // 2

    mu_x = F.conv2d(pred_c, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sg_x_sq = F.conv2d(pred_c * pred_c, kernel, padding=padding, groups=channels) - mu_x_sq
    sg_y_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y_sq
    sg_xy   = F.conv2d(pred_c * target, kernel, padding=padding, groups=channels) - mu_xy

    numer = (2 * mu_xy + C1) * (2 * sg_xy + C2)
    denom = (mu_x_sq + mu_y_sq + C1) * (sg_x_sq + sg_y_sq + C2)

    ssim_map = numer / (denom + 1e-8)
    return ssim_map.mean().item()


class MetricsTracker:
    """Accumulates PSNR and SSIM values over a validation epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._psnr_sum  = 0.0
        self._ssim_sum  = 0.0
        self._count     = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update with one batch. Detach from graph before calling."""
        with torch.no_grad():
            self._psnr_sum += compute_psnr(pred, target)
            self._ssim_sum += compute_ssim(pred, target)
            self._count    += 1

    def result(self) -> dict:
        if self._count == 0:
            return {"psnr": 0.0, "ssim": 0.0}
        return {
            "psnr": self._psnr_sum / self._count,
            "ssim": self._ssim_sum / self._count,
        }
