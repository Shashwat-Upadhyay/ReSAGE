import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard residual block without normalization (e.g. BatchNorm),
    preserving absolute intensity information and preventing degradation corruptions.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class ReSAGEBaseline(nn.Module):
    """
    ReSAGE working baseline model for image restoration and 2x super-resolution.
    Designed with a modular architecture to expose intermediate features.
    """
    def __init__(self, in_channels=1, out_channels=1, mid_channels=64, num_residual_blocks=4, upscale_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.upscale_factor = upscale_factor
        
        # 1. Encoder / Feature Extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 2. Restoration Backbone (Residual blocks)
        backbone_layers = []
        for _ in range(num_residual_blocks):
            backbone_layers.append(ResidualBlock(mid_channels))
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 3. 2x Upsampling Module (PixelShuffle)
        # Upsamples from H x W to 2H x 2W
        self.upsampler = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * (upscale_factor ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 4. Reconstruction Head
        # Outputs 1 grayscale channel (No final Sigmoid/Tanh, as raw intensities must be preserved)
        self.reconstruct = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        
    def extract_features(self, x):
        """
        Extracts intermediate features before upsampling and reconstruction.
        Useful for downstream feature regulation and SAE coupling.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channels, H, W]
            
        Returns:
            torch.Tensor: Intermediate feature map of shape [batch, mid_channels, H, W]
        """
        features = self.encoder(x)
        features = self.backbone(features)
        return features
        
    def forward(self, x):
        """
        Full forward pass mapping input [batch, 1, 128, 128] to output [batch, 1, 256, 256].
        """
        # Step 1 & 2: Extract intermediate features
        features = self.extract_features(x)
        
        # Step 3: Learned 2x upsampling
        upsampled = self.upsampler(features)
        
        # Step 4: Final reconstruction
        out = self.reconstruct(upsampled)
        
        return out
