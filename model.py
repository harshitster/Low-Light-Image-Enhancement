"""
MIRNet Model Implementation in PyTorch
"""

import torch
import torch.nn as nn
from blocks import RecursiveResidualBlock


class MIRNet(nn.Module):
    """
    MIRNet: Multi-scale Image Restoration Network

    Args:
        rrb_count (int): Number of Recursive Residual Blocks
        msrb_count (int): Number of Multi-Scale Residual Blocks per RRB
        channels (int): Number of channels for feature processing
        in_channels (int): Number of input channels (default: 3 for RGB)
        out_channels (int): Number of output channels (default: 3 for RGB)
    """

    def __init__(
        self, rrb_count=3, msrb_count=2, channels=64, in_channels=3, out_channels=3
    ):
        super(MIRNet, self).__init__()

        self.rrb_count = rrb_count
        self.msrb_count = msrb_count
        self.channels = channels

        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        self.rrbs = nn.ModuleList(
            [RecursiveResidualBlock(channels, msrb_count) for _ in range(rrb_count)]
        )

        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of MIRNet

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Enhanced image tensor of shape [B, C, H, W]
        """
        input_tensor = x

        x = self.conv_in(x)

        for rrb in self.rrbs:
            x = rrb(x)

        x = self.conv_out(x)

        output = input_tensor + x

        return output

    def get_model_size(self):
        """Calculate total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),
        }

    def print_model_summary(self):
        """Print model architecture summary"""
        print(f"MIRNet Model Summary:")
        print(f"- RRB Count: {self.rrb_count}")
        print(f"- MSRB Count per RRB: {self.msrb_count}")
        print(f"- Channels: {self.channels}")

        model_info = self.get_model_size()
        print(f"- Total Parameters: {model_info['total_parameters']:,}")
        print(f"- Trainable Parameters: {model_info['trainable_parameters']:,}")
        print(f"- Model Size: {model_info['model_size_mb']:.2f} MB")


def create_mirnet_model(config):
    """
    Factory function to create MIRNet model from config

    Args:
        config: Configuration object with model parameters

    Returns:
        MIRNet: Instantiated model
    """
    model = MIRNet(
        rrb_count=config.RRB_COUNT,
        msrb_count=config.MSRB_COUNT,
        channels=config.CHANNELS,
    )

    return model


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    import config

    model = create_mirnet_model(config)
    model.print_model_summary()

    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nTest Forward Pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
