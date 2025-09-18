import torch
import torch.nn as nn
from src.blocks import ConvBNAct, DepthwiseSeparableConv, SqueezeExcite


class SimpleBackbone(nn.Module):
    """
    A lightweight CNN backbone built from our custom blocks.

    Structure:
    - ConvBNAct stem
    - Several stages (each can downsample by stride=2)
    - Each stage = [DepthwiseSeparableConv + SqueezeExcite] * n
    """

    def __init__(self, in_channels=3, channels=[32, 64, 128, 256], num_blocks=[2, 2, 3, 2]):
        super().__init__()

        assert len(channels) == len(num_blocks), "channels and num_blocks must match in length"

        # Stem: first conv to reduce spatial size and increase channels
        self.stem = ConvBNAct(in_channels, channels[0], kernel_size=3, stride=2)

        # Build stages
        stages = []
        in_ch = channels[0]
        for stage_ch, n_blocks in zip(channels, num_blocks):
            blocks = []
            for i in range(n_blocks):
                stride = 2 if i == 0 else 1  # first block in stage downsamples
                blocks.append(DepthwiseSeparableConv(in_ch, stage_ch, stride))
                blocks.append(SqueezeExcite(stage_ch))
                in_ch = stage_ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.out_channels = in_ch

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
