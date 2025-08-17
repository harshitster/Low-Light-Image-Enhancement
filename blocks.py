"""
Building blocks for MIRNet model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveKernelFeatureFusion(nn.Module):
    """Selective Kernel Feature Fusion (SKFF) module"""

    def __init__(self, channels):
        super(SelectiveKernelFeatureFusion, self).__init__()
        self.channels = channels

        # Global average pooling is handled in forward pass
        self.compact_layer = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.descriptor1 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.descriptor2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.descriptor3 = nn.Conv2d(channels // 8, channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature1, feature2, feature3):
        # Combine features
        combined = feature1 + feature2 + feature3

        # Global average pooling
        gap = F.adaptive_avg_pool2d(combined, 1)  # [B, C, 1, 1]

        # Compact feature representation
        compact = self.relu(self.compact_layer(gap))

        # Generate attention weights
        desc1 = self.softmax(self.descriptor1(compact))
        desc2 = self.softmax(self.descriptor2(compact))
        desc3 = self.softmax(self.descriptor3(compact))

        # Apply attention weights
        weighted1 = feature1 * desc1
        weighted2 = feature2 * desc2
        weighted3 = feature3 * desc3

        return weighted1 + weighted2 + weighted3


class ChannelAttentionBlock(nn.Module):
    """Channel Attention Block"""

    def __init__(self, channels):
        super(ChannelAttentionBlock, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        gap = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]

        # Channel attention
        attention = self.relu(self.conv1(gap))
        attention = self.sigmoid(self.conv2(attention))

        return x * attention


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention Block"""

    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Concatenate
        concat = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]

        # Generate spatial attention map
        attention = self.sigmoid(self.conv(concat))

        return x * attention


class DualAttentionUnitBlock(nn.Module):
    """Dual Attention Unit (DAU) Block"""

    def __init__(self, channels):
        super(DualAttentionUnitBlock, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels * 2, channels, kernel_size=1)

        self.channel_attention = ChannelAttentionBlock(channels)
        self.spatial_attention = SpatialAttentionBlock()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Feature extraction
        feature = self.relu(self.conv1(x))
        feature = self.conv2(feature)

        # Apply attention mechanisms
        channel_att = self.channel_attention(feature)
        spatial_att = self.spatial_attention(feature)

        # Concatenate and combine
        concat = torch.cat([channel_att, spatial_att], dim=1)
        combined = self.conv3(concat)

        # Residual connection
        return x + combined


class DownSamplingBlock(nn.Module):
    """Down-sampling block for multi-scale processing"""

    def __init__(self, channels):
        super(DownSamplingBlock, self).__init__()
        self.channels = channels

        # Main branch
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Skip branch
        self.conv_skip = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Main branch
        main = self.relu(self.conv1(x))
        main = self.relu(self.conv2(main))
        main = self.maxpool1(main)
        main = self.conv3(main)

        # Skip branch
        skip = self.maxpool2(x)
        skip = self.conv_skip(skip)

        return main + skip


class UpSamplingBlock(nn.Module):
    """Up-sampling block for multi-scale processing"""

    def __init__(self, channels):
        super(UpSamplingBlock, self).__init__()
        self.channels = channels

        # Main branch
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels // 2, kernel_size=1)

        # Skip branch
        self.conv_skip = nn.Conv2d(channels, channels // 2, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Main branch
        main = self.relu(self.conv1(x))
        main = self.relu(self.conv2(main))
        main = F.interpolate(main, scale_factor=2, mode="bilinear", align_corners=False)
        main = self.conv3(main)

        # Skip branch
        skip = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        skip = self.conv_skip(skip)

        return main + skip


class MultiScaleResidualBlock(nn.Module):
    """Multi-Scale Residual Block (MSRB)"""

    def __init__(self, channels):
        super(MultiScaleResidualBlock, self).__init__()
        self.channels = channels

        # Down-sampling blocks
        self.down1 = DownSamplingBlock(channels)  # 64 -> 128
        self.down2 = DownSamplingBlock(channels * 2)  # 128 -> 256

        # Up-sampling blocks with unique instances
        self.up_256_to_128_1 = UpSamplingBlock(channels * 4)  # 256 -> 128
        self.up_256_to_128_2 = UpSamplingBlock(channels * 4)  # 256 -> 128
        self.up_256_to_128_3 = UpSamplingBlock(channels * 4)  # 256 -> 128
        self.up_128_to_64_1 = UpSamplingBlock(channels * 2)  # 128 -> 64
        self.up_128_to_64_2 = UpSamplingBlock(channels * 2)  # 128 -> 64
        self.up_128_to_64_3 = UpSamplingBlock(channels * 2)  # 128 -> 64
        self.up_128_to_64_4 = UpSamplingBlock(channels * 2)  # 128 -> 64

        # Dual attention units
        self.dau1_1 = DualAttentionUnitBlock(channels)  # 64 channels
        self.dau1_2 = DualAttentionUnitBlock(channels * 2)  # 128 channels
        self.dau1_3 = DualAttentionUnitBlock(channels * 4)  # 256 channels

        self.dau2_1 = DualAttentionUnitBlock(channels)  # 64 channels
        self.dau2_2 = DualAttentionUnitBlock(channels * 2)  # 128 channels
        self.dau2_3 = DualAttentionUnitBlock(channels * 4)  # 256 channels

        # SKFF modules
        self.skff1 = SelectiveKernelFeatureFusion(channels)  # 64 channels
        self.skff2 = SelectiveKernelFeatureFusion(channels * 2)  # 128 channels
        self.skff3 = SelectiveKernelFeatureFusion(channels * 4)  # 256 channels
        self.skff_final = SelectiveKernelFeatureFusion(channels)  # 64 channels

        # Additional down-sampling for SKFF
        self.down_for_skff2 = DownSamplingBlock(channels)  # 64 -> 128
        self.down_for_skff3_1 = DownSamplingBlock(channels)  # 64 -> 128
        self.down_for_skff3_2 = DownSamplingBlock(channels * 2)  # 128 -> 256

        # Final convolution
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Multi-scale feature extraction
        feature1 = x  # channels (64)
        feature2 = self.down1(feature1)  # channels * 2 (128)
        feature3 = self.down2(feature2)  # channels * 4 (256)

        # First round of DAU
        feature1_dau1 = self.dau1_1(feature1)  # 64 channels
        feature2_dau1 = self.dau1_2(feature2)  # 128 channels
        feature3_dau1 = self.dau1_3(feature3)  # 256 channels

        # SKFF at different scales
        # For skff1: all inputs should be 64 channels
        skff1 = self.skff1(
            feature1_dau1,  # 64 channels
            self.up_128_to_64_1(feature2_dau1),  # 128 -> 64 channels
            self.up_128_to_64_2(
                self.up_256_to_128_1(feature3_dau1)
            ),  # 256 -> 128 -> 64 channels
        )

        # For skff2: all inputs should be 128 channels
        skff2 = self.skff2(
            self.down_for_skff2(feature1_dau1),  # 64 -> 128 channels
            feature2_dau1,  # 128 channels
            self.up_256_to_128_2(feature3_dau1),  # 256 -> 128 channels
        )

        # For skff3: all inputs should be 256 channels
        skff3 = self.skff3(
            self.down_for_skff3_2(self.down_for_skff3_1(feature1_dau1)),  # 64->128->256
            self.down2(feature2_dau1),  # 128 -> 256 channels
            feature3_dau1,  # 256 channels
        )

        # Second round of DAU
        feature1_dau2 = self.dau2_1(skff1)  # 64 channels
        feature2_dau2 = self.dau2_2(skff2)  # 128 channels
        feature3_dau2 = self.dau2_3(skff3)  # 256 channels

        # Final SKFF - bring all to 64 channels
        skff_final = self.skff_final(
            feature1_dau2,  # 64 channels
            self.up_128_to_64_3(feature2_dau2),  # 128 -> 64 channels
            self.up_128_to_64_4(
                self.up_256_to_128_3(feature3_dau2)
            ),  # 256 -> 128 -> 64 channels
        )

        # Final convolution and residual connection
        output = self.final_conv(skff_final)
        return x + output


class RecursiveResidualBlock(nn.Module):
    """Recursive Residual Block (RRB)"""

    def __init__(self, channels, msrb_count):
        super(RecursiveResidualBlock, self).__init__()
        self.channels = channels
        self.msrb_count = msrb_count

        self.conv_in = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Stack multiple MSRBs
        self.msrbs = nn.ModuleList(
            [MultiScaleResidualBlock(channels) for _ in range(msrb_count)]
        )

    def forward(self, x):
        residual = x
        x = self.conv_in(x)

        # Pass through multiple MSRBs
        for msrb in self.msrbs:
            x = msrb(x)

        x = self.conv_out(x)
        return residual + x
