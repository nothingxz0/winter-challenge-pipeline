"""
Compact U-Net Feature Extractor for Stable-Baselines3.

This implements a U-Net architecture optimized for ~90k parameters
targeting spatial understanding for the snakebot game.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for parameter efficiency."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CompactUNetBlock(nn.Module):
    """Efficient U-Net block with depthwise separable convolutions."""

    def __init__(self, in_channels, out_channels, use_separable=True):
        super().__init__()

        if use_separable and in_channels > 1:
            self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, 3, 1, 1)
            self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, 3, 1, 1)
        else:
            # Regular conv for first layer or when separable not beneficial
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return x


class CompactUNetExtractor(BaseFeaturesExtractor):
    """
    Compact U-Net feature extractor optimized for ~90k parameters.

    Architecture:
    - Encoder: 4 levels with channels 7→12→24→48→96
    - Decoder: Skip connections, 96→48→24→12
    - Output: Adaptive pooling to fixed feature size
    - Target: ~90k parameters (vs 112k with larger channels)
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        # Extract spatial dimensions
        if len(observation_space.shape) != 3:
            raise ValueError(f"Expected 3D observation space (C, H, W), got {observation_space.shape}")

        channels, height, width = observation_space.shape
        super().__init__(observation_space, features_dim)

        self.input_channels = channels

        # Encoder levels - Leaner channels to fit 100k char limit
        self.enc1 = CompactUNetBlock(channels, 8, use_separable=False)  # Input layer
        self.enc2 = CompactUNetBlock(8, 16, use_separable=True)
        self.enc3 = CompactUNetBlock(16, 32, use_separable=True)
        self.enc4 = CompactUNetBlock(32, 64, use_separable=True)

        # Max pooling for downsampling
        self.pool = nn.MaxPool2d(2)

        # Decoder with skip connections - Leaner channels
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec4 = CompactUNetBlock(64, 32, use_separable=True)  # 32 + 32 skip

        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec3 = CompactUNetBlock(32, 16, use_separable=True)   # 16 + 16 skip

        self.up2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec2 = CompactUNetBlock(16, 8, use_separable=True)   # 8 + 8 skip

        # Final feature extraction - Smaller for budget
        self.final_conv = nn.Conv2d(8, 4, 1)  # Reduce channels
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed output size
        self.feature_proj = nn.Linear(4 * 4 * 4, self.features_dim)
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self._initialize_weights()

        # Print parameter count for verification
        param_count = count_parameters(self)
        print(f"CompactUNetExtractor parameters: {param_count:,}")

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Handle both batched and single observations
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)  # Add batch dimension
            remove_batch = True
        else:
            remove_batch = False

        batch_size = observations.shape[0]

        # Encoder path with skip connections
        enc1 = self.enc1(observations)  # channels → 16
        pool1 = self.pool(enc1)         # H/2, W/2

        enc2 = self.enc2(pool1)         # 16 → 32
        pool2 = self.pool(enc2)         # H/4, W/4

        enc3 = self.enc3(pool2)         # 32 → 64
        pool3 = self.pool(enc3)         # H/8, W/8

        enc4 = self.enc4(pool3)         # 64 → 128

        # Decoder path with skip connections
        up4 = self.up4(enc4)            # Upsample to H/4, W/4

        # Handle size mismatches due to padding
        if up4.shape != enc3.shape:
            up4 = F.interpolate(up4, size=enc3.shape[2:], mode='nearest')

        dec4 = self.dec4(torch.cat([up4, enc3], dim=1))  # 48 + 48 → 48

        up3 = self.up3(dec4)            # Upsample to H/2, W/2
        if up3.shape != enc2.shape:
            up3 = F.interpolate(up3, size=enc2.shape[2:], mode='nearest')

        dec3 = self.dec3(torch.cat([up3, enc2], dim=1))  # 24 + 24 → 24

        up2 = self.up2(dec3)            # Upsample to H, W
        if up2.shape != enc1.shape:
            up2 = F.interpolate(up2, size=enc1.shape[2:], mode='nearest')

        dec2 = self.dec2(torch.cat([up2, enc1], dim=1))  # 12 + 12 → 12

        # Final feature extraction
        final = self.final_conv(dec2)   # 12 → 6 channels
        pooled = self.adaptive_pool(final)  # → (6, 4, 4)
        flattened = pooled.view(batch_size, -1)  # → (6*4*4,)

        features = self.dropout(self.feature_proj(flattened))  # → self.features_dim

        if remove_batch:
            features = features.squeeze(0)

        return features


def test_unet_extractor():
    """Test the U-Net extractor with different input sizes."""
    print("Testing CompactUNetExtractor...")

    # Test with different spatial sizes to verify variable size handling
    test_cases = [
        (7, 32, 32),   # Small grid
        (7, 48, 48),   # Medium grid
        (7, 64, 64),   # Large grid (padded)
    ]

    for channels, height, width in test_cases:
        print(f"\nTesting with input shape: ({channels}, {height}, {width})")

        # Create observation space and model
        obs_space = spaces.Box(low=0, high=1, shape=(channels, height, width), dtype=np.float32)
        model = CompactUNetExtractor(obs_space, features_dim=64)

        # Test forward pass
        test_input = torch.randn(2, channels, height, width)  # Batch of 2
        output = model(test_input)

        print(f"Output shape: {output.shape}")
        assert output.shape == (2, 64), f"Expected (2, 64), got {output.shape}"

        # Test single observation
        single_input = torch.randn(channels, height, width)
        single_output = model(single_input)
        print(f"Single obs output shape: {single_output.shape}")
        assert single_output.shape == (64,), f"Expected (64,), got {single_output.shape}"

    print("\n✅ All U-Net tests passed!")
    return model


if __name__ == "__main__":
    test_unet_extractor()