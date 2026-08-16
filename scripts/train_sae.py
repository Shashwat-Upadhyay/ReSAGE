import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.sae import SparseAutoencoder, sae_loss


LATENT_DIM = 128
BATCH_SIZE = 4096
EPOCHS = 30
LEARNING_RATE = 1e-3
SPARSITY_WEIGHT = 1e-3
SAMPLES_PER_IMAGE = 512
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_dataset(feature_dir):
    """
    Load each feature map once, sample 512 spatial locations,
    and create a tensor dataset of 64-D feature vectors.
    """

    files = sorted(
        f for f in os.listdir(feature_dir)
        if f.endswith(".npy")
    )

    print(f"Feature files found: {len(files)}")

    all_samples = []

    for idx, filename in enumerate(files):

        path = os.path.join(feature_dir, filename)

        feature_map = np.load(path).astype(np.float32)

        # Expected shape: (64, 128, 128)
        if feature_map.shape != (64, 128, 128):
            raise ValueError(
                f"Unexpected feature shape {feature_map.shape} "
                f"in {filename}"
            )

        # Convert:
        # (64, 128, 128)
        # →
        # (128*128, 64)
        vectors = feature_map.reshape(64, -1).T

        # Deterministic sampling
        rng = np.random.default_rng(SEED + idx)

        selected_indices = rng.choice(
            vectors.shape[0],
            size=SAMPLES_PER_IMAGE,
            replace=False
        )

        sampled = vectors[selected_indices]

        all_samples.append(sampled)

        if (idx + 1) % 200 == 0:
            print(
                f"Processed {idx + 1}/{len(files)} feature files"
            )

    # Shape:
    # (3200, 512, 64)
    all_samples = np.concatenate(all_samples, axis=0)

    print("\nDataset construction complete.")
    print(f"Training vectors: {all_samples.shape[0]}")
    print(f"Feature dimension: {all_samples.shape[1]}")

    tensor = torch.from_numpy(all_samples)

    print(f"Tensor size: {tensor.shape}")
    print(
        f"Memory usage: "
        f"{tensor.numel() * tensor.element_size() / (1024**2):.1f} MB"
    )

    return tensor


def main():

    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoder on ReSAGE features"
    )

    parser.add_argument(
        "--feature_dir",
        type=str,
        default="outputs/features"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints"
    )

    args = parser.parse_args()

    set_seed(SEED)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------

    print("\nBuilding training dataset...")

    features = build_training_dataset(
        args.feature_dir
    )

    dataset = TensorDataset(features)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = SparseAutoencoder(
        input_dim=64,
        latent_dim=LATENT_DIM
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        f"\nSAE parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        total_reconstruction = 0.0
        total_sparsity = 0.0

        for batch in loader:

            x = batch[0].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            reconstructed, z = model(x)

            loss, reconstruction_loss, sparsity_loss = sae_loss(
                x,
                reconstructed,
                z,
                sparsity_weight=SPARSITY_WEIGHT
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch {epoch}"
                )

            loss.backward()

            optimizer.step()

            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            total_reconstruction += (
                reconstruction_loss.item() * batch_size
            )
            total_sparsity += (
                sparsity_loss.item() * batch_size
            )

        dataset_size = len(dataset)

        avg_loss = total_loss / dataset_size
        avg_reconstruction = (
            total_reconstruction / dataset_size
        )
        avg_sparsity = (
            total_sparsity / dataset_size
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Loss: {avg_loss:.6f} | "
            f"Recon: {avg_reconstruction:.6f} | "
            f"Sparsity: {avg_sparsity:.6f}"
        )

        # -----------------------------------------------------
        # Save latest checkpoint
        # -----------------------------------------------------

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
            "reconstruction_loss": avg_reconstruction,
            "sparsity_loss": avg_sparsity,
            "latent_dim": LATENT_DIM
        }

        torch.save(
            checkpoint,
            os.path.join(
                args.checkpoint_dir,
                "sae_latest.pth"
            )
        )

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                checkpoint,
                os.path.join(
                    args.checkpoint_dir,
                    "sae_best.pth"
                )
            )

            print(
                f"  Best checkpoint saved "
                f"(loss={best_loss:.6f})"
            )

    print("\nSAE training completed.")
    print(f"Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()