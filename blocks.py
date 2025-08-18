import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveKernelFeatureFusion(nn.Module):
    """Selective Kernel Feature Fusion (SKFF) module"""

    def __init__(self, channels):
        super(SelectiveKernelFeatureFusion, self).__init__()
        self.channels = channels

        self.compact_layer = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.descriptor1 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.descriptor2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.descriptor3 = nn.Conv2d(channels // 8, channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature1, feature2, feature3):
        combined = feature1 + feature2 + feature3

        gap = F.adaptive_avg_pool2d(combined, 1)

        compact = self.relu(self.compact_layer(gap))

        desc1 = self.softmax(self.descriptor1(compact))
        desc2 = self.softmax(self.descriptor2(compact))
        desc3 = self.softmax(self.descriptor3(compact))

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
        gap = F.adaptive_avg_pool2d(x, 1)

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
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg_pool, max_pool], dim=1)

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
        feature = self.relu(self.conv1(x))
        feature = self.conv2(feature)

        channel_att = self.channel_attention(feature)
        spatial_att = self.spatial_attention(feature)

        concat = torch.cat([channel_att, spatial_att], dim=1)
        combined = self.conv3(concat)

        return x + combined


class DownSamplingBlock(nn.Module):
    """Down-sampling block for multi-scale processing"""

    def __init__(self, channels):
        super(DownSamplingBlock, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_skip = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main = self.relu(self.conv1(x))
        main = self.relu(self.conv2(main))
        main = self.maxpool1(main)
        main = self.conv3(main)

        skip = self.maxpool2(x)
        skip = self.conv_skip(skip)

        return main + skip


class UpSamplingBlock(nn.Module):
    """Up-sampling block for multi-scale processing"""

    def __init__(self, channels):
        super(UpSamplingBlock, self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels // 2, kernel_size=1)

        self.conv_skip = nn.Conv2d(channels, channels // 2, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main = self.relu(self.conv1(x))
        main = self.relu(self.conv2(main))
        main = F.interpolate(main, scale_factor=2, mode="bilinear", align_corners=False)
        main = self.conv3(main)

        skip = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        skip = self.conv_skip(skip)

        return main + skip


class MultiScaleResidualBlock(nn.Module):
    """Multi-Scale Residual Block (MSRB)"""

    def __init__(self, channels):
        super(MultiScaleResidualBlock, self).__init__()
        self.channels = channels

        self.down1 = DownSamplingBlock(channels)
        self.down2 = DownSamplingBlock(channels * 2)

        self.up_256_to_128_1 = UpSamplingBlock(channels * 4)
        self.up_256_to_128_2 = UpSamplingBlock(channels * 4)
        self.up_256_to_128_3 = UpSamplingBlock(channels * 4)
        self.up_128_to_64_1 = UpSamplingBlock(channels * 2)
        self.up_128_to_64_2 = UpSamplingBlock(channels * 2)
        self.up_128_to_64_3 = UpSamplingBlock(channels * 2)
        self.up_128_to_64_4 = UpSamplingBlock(channels * 2)

        self.dau1_1 = DualAttentionUnitBlock(channels)
        self.dau1_2 = DualAttentionUnitBlock(channels * 2)
        self.dau1_3 = DualAttentionUnitBlock(channels * 4)

        self.dau2_1 = DualAttentionUnitBlock(channels)
        self.dau2_2 = DualAttentionUnitBlock(channels * 2)
        self.dau2_3 = DualAttentionUnitBlock(channels * 4)

        self.skff1 = SelectiveKernelFeatureFusion(channels)
        self.skff2 = SelectiveKernelFeatureFusion(channels * 2)
        self.skff3 = SelectiveKernelFeatureFusion(channels * 4)
        self.skff_final = SelectiveKernelFeatureFusion(channels)

        self.down_for_skff2 = DownSamplingBlock(channels)
        self.down_for_skff3_1 = DownSamplingBlock(channels)
        self.down_for_skff3_2 = DownSamplingBlock(channels * 2)

        self.final_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        feature1 = x
        feature2 = self.down1(feature1)
        feature3 = self.down2(feature2)

        feature1_dau1 = self.dau1_1(feature1)
        feature2_dau1 = self.dau1_2(feature2)
        feature3_dau1 = self.dau1_3(feature3)

        skff1 = self.skff1(
            feature1_dau1,
            self.up_128_to_64_1(feature2_dau1),
            self.up_128_to_64_2(self.up_256_to_128_1(feature3_dau1)),
        )

        skff2 = self.skff2(
            self.down_for_skff2(feature1_dau1),
            feature2_dau1,
            self.up_256_to_128_2(feature3_dau1),
        )

        skff3 = self.skff3(
            self.down_for_skff3_2(self.down_for_skff3_1(feature1_dau1)),
            self.down2(feature2_dau1),
            feature3_dau1,
        )

        feature1_dau2 = self.dau2_1(skff1)
        feature2_dau2 = self.dau2_2(skff2)
        feature3_dau2 = self.dau2_3(skff3)

        skff_final = self.skff_final(
            feature1_dau2,
            self.up_128_to_64_3(feature2_dau2),
            self.up_128_to_64_4(self.up_256_to_128_3(feature3_dau2)),
        )

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

        self.msrbs = nn.ModuleList(
            [MultiScaleResidualBlock(channels) for _ in range(msrb_count)]
        )

    def forward(self, x):
        residual = x
        x = self.conv_in(x)

        for msrb in self.msrbs:
            x = msrb(x)

        x = self.conv_out(x)
        return residual + x
