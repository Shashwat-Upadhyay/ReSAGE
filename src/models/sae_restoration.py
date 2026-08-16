
import torch
import torch.nn as nn

from src.models.baseline import ReSAGEBaseline
from src.models.sae import SparseAutoencoder


class ReSAGESAERestoration(nn.Module):
    """
    ReSAGE baseline with a trained Sparse Autoencoder inserted
    between feature extraction and image reconstruction.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        mid_channels=64,
        num_residual_blocks=4,
        upscale_factor=2,
        latent_dim=128,
        freeze_sae=True
    ):
        super().__init__()

        self.baseline = ReSAGEBaseline(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            num_residual_blocks=num_residual_blocks,
            upscale_factor=upscale_factor
        )

        self.sae = SparseAutoencoder(
            input_dim=mid_channels,
            latent_dim=latent_dim
        )

        if freeze_sae:
            for param in self.sae.parameters():
                param.requires_grad = False

    def forward(self, x):

        # Extract baseline features
        features = self.baseline.extract_features(x)

        B, C, H, W = features.shape

        # [B,C,H,W] -> [B*H*W,C]
        feature_vectors = features.permute(
            0, 2, 3, 1
        ).reshape(-1, C)

        # SAE: 64 -> 128 -> 64
        reconstructed_vectors, latent = self.sae(
            feature_vectors
        )

        # [B*H*W,C] -> [B,C,H,W]
        reconstructed_features = reconstructed_vectors.reshape(
            B, H, W, C
        )

        reconstructed_features = reconstructed_features.permute(
            0, 3, 1, 2
        ).contiguous()

        # Original baseline reconstruction path
        upsampled = self.baseline.upsampler(
            reconstructed_features
        )

        output = self.baseline.reconstruct(
            upsampled
        )

        return output

    def extract_latent(self, x):

        features = self.baseline.extract_features(x)

        B, C, H, W = features.shape

        feature_vectors = features.permute(
            0, 2, 3, 1
        ).reshape(-1, C)

        latent = self.sae.encode(
            feature_vectors
        )

        latent = latent.reshape(
            B, H, W, -1
        )

        latent = latent.permute(
            0, 3, 1, 2
        ).contiguous()

        return latent
