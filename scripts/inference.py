"""
ReSAGE Baseline Inference Script

Usage:
    python scripts/inference.py \
        --input_dir  data/test/NoisyLR \
        --output_dir outputs/inference \
        --checkpoint checkpoints/baseline_best.pth \
        [--save_png]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.baseline import ReSAGEBaseline


def load_model(checkpoint_path: str, device: torch.device) -> ReSAGEBaseline:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)  # support both plain state-dict and full checkpoint

    # Infer mid_channels from the checkpoint weights
    # key: "encoder.0.weight" → shape [mid_channels, in_channels, k, k]
    mid_channels = state["encoder.0.weight"].shape[0]
    # key: "backbone.0.conv1.weight" exists when num_residual_blocks >= 1
    num_blocks = sum(1 for k in state if k.startswith("backbone.") and k.endswith(".conv1.weight"))

    model = ReSAGEBaseline(
        in_channels=1,
        out_channels=1,
        mid_channels=mid_channels,
        num_residual_blocks=num_blocks,
        upscale_factor=2,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_inference(model, npy_path: str, device: torch.device):
    """Load one .npy file, run inference, return output as numpy [H, W] float32."""
    raw = np.load(npy_path).astype(np.float32)

    # Preserve raw intensity — no clipping, no per-image normalization
    tensor = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    out    = model(tensor)                                                 # [1,1,2H,2W]
    return out.squeeze().cpu().numpy()                                     # [2H, 2W]


def save_outputs(out_arr: np.ndarray, stem: str, output_dir: str, save_png: bool):
    os.makedirs(output_dir, exist_ok=True)

    # Always save .npy
    npy_path = os.path.join(output_dir, stem + ".npy")
    np.save(npy_path, out_arr)

    if save_png:
        import matplotlib.pyplot as plt
        png_path = os.path.join(output_dir, stem + ".png")
        fig, ax  = plt.subplots(figsize=(4, 4))
        ax.imshow(np.clip(out_arr, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ReSAGE Baseline Inference")
    parser.add_argument("--input_dir",  required=True,  help="Directory containing .npy NoisyLR files")
    parser.add_argument("--output_dir", required=True,  help="Directory to save restored .npy outputs")
    parser.add_argument("--checkpoint", required=True,  help="Path to model checkpoint")
    parser.add_argument("--save_png",   action="store_true", help="Also save PNG visualizations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    npy_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not npy_files:
        print(f"No .npy files found in {args.input_dir}")
        return

    print(f"Found {len(npy_files)} file(s). Running inference...")
    times = []
    for fname in npy_files:
        src_path = os.path.join(args.input_dir, fname)
        t0       = time.perf_counter()
        out_arr  = run_inference(model, src_path, device)
        elapsed  = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed)

        stem = os.path.splitext(fname)[0]
        save_outputs(out_arr, stem, args.output_dir, args.save_png)

    avg_ms = sum(times) / len(times)
    print(f"Done. Outputs saved to: {args.output_dir}")
    print(f"Average inference time: {avg_ms:.1f} ms/image")


if __name__ == "__main__":
    main()
