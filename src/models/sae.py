import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=128):
        super().__init__()

        # Encoder: 64-D baseline feature → 128-D sparse latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder: 128-D latent → reconstructed 64-D feature
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)

        return x_reconstructed, z


def sae_loss(x, x_reconstructed, z, sparsity_weight=1e-3):
    # Reconstruction error
    reconstruction_loss = nn.functional.mse_loss(
        x_reconstructed,
        x
    )

    # L1 sparsity penalty
    sparsity_loss = torch.mean(torch.abs(z))

    # Total SAE loss
    total_loss = (
        reconstruction_loss
        + sparsity_weight * sparsity_loss
    )

    return total_loss, reconstruction_loss, sparsity_loss