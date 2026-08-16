import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.sae import SparseAutoencoder, sae_loss


# =========================
# Configuration
# =========================

FEATURE_DIR = "outputs/features"
CHECKPOINT_DIR = "checkpoints"

LATENT_DIM = 128
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 1e-3
SPARSITY_WEIGHT = 1e-3

# Number of spatial feature vectors sampled from each image
SAMPLES_PER_IMAGE = 512

SEED = 42


# =========================
# Reproducibility
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# Dataset
# =========================

class FeatureDataset(Dataset):

    def __init__(
        self,
        feature_dir,
        samples_per_image=512
    ):

        self.feature_files = sorted([
            os.path.join(feature_dir, f)
            for f in os.listdir(feature_dir)
            if f.endswith(".npy")
        ])

        self.samples_per_image = samples_per_image

        print(f"Feature files found: {len(self.feature_files)}")

        if len(self.feature_files) == 0:
            raise RuntimeError(
                f"No .npy feature files found in {feature_dir}"
            )

        # Total number of training vectors
        self.total_samples = (
            len(self.feature_files) * samples_per_image
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):

        # Which feature image
        image_index = index // self.samples_per_image

        feature_path = self.feature_files[image_index]

        # Load feature map
        feature_map = np.load(feature_path)

        # Expected shape:
        # (64, 128, 128)

        if feature_map.ndim != 3:
            raise ValueError(
                f"Unexpected feature shape: {feature_map.shape}"
            )

        channels, height, width = feature_map.shape

        if channels != 64:
            raise ValueError(
                f"Expected 64 channels, got {channels}"
            )

        # Deterministic spatial sampling
        sample_index = index % self.samples_per_image

        # Generate deterministic pseudo-random coordinates
        rng = np.random.default_rng(
            SEED + index
        )

        y = rng.integers(0, height)
        x = rng.integers(0, width)

        # Extract 64-D feature vector
        feature_vector = feature_map[:, y, x]

        return torch.tensor(
            feature_vector,
            dtype=torch.float32
        )


# =========================
# Validation
# =========================

def validate_dataset(dataset):

    print("\nChecking dataset...")

    x = dataset[0]

    print("Sample shape:", x.shape)
    print("Sample dtype:", x.dtype)
    print("Sample min:", x.min().item())
    print("Sample max:", x.max().item())
    print("Sample mean:", x.mean().item())
    print("Sample std:", x.std().item())

    assert x.shape == (64,)

    print("Dataset check passed.\n")


# =========================
# Training
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoder on ReSAGE features"
    )

    parser.add_argument(
        "--feature_dir",
        default="outputs/features",
        help="Directory containing extracted feature .npy files"
    )

    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory for SAE checkpoints"
    )

    return parser.parse_args()

def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Create dataset
    dataset = FeatureDataset(
        args.feature_dir,
        SAMPLES_PER_IMAGE
    )

    validate_dataset(dataset)

    print(
        f"Training vectors: {len(dataset):,}"
    )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # SAE
    model = SparseAutoencoder(
        input_dim=64,
        latent_dim=LATENT_DIM
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        f"SAE parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_loss = float("inf")

    # =========================
    # Training loop
    # =========================

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        total_reconstruction = 0.0
        total_sparsity = 0.0

        for batch in loader:

            batch = batch.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            reconstructed, latent = model(batch)

            loss, reconstruction_loss, sparsity_loss = sae_loss(
                batch,
                reconstructed,
                latent,
                SPARSITY_WEIGHT
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_sparsity += sparsity_loss.item()

        num_batches = len(loader)

        avg_loss = total_loss / num_batches
        avg_reconstruction = (
            total_reconstruction / num_batches
        )
        avg_sparsity = (
            total_sparsity / num_batches
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Loss={avg_loss:.6f} "
            f"Recon={avg_reconstruction:.6f} "
            f"Sparsity={avg_sparsity:.6f}"
        )

        # Save best model
        if avg_loss < best_loss:

            best_loss = avg_loss

            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                "sae_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": best_loss,
                },
                checkpoint_path
            )

            print(
                f"  ✓ Saved best checkpoint: "
                f"{checkpoint_path}"
            )

    print("\nSAE training completed.")
    print(
        f"Best loss: {best_loss:.6f}"
    )


if __name__ == "__main__":
    main()