import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv → BatchNorm → Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=nn.SiLU()):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv + Pointwise conv"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SqueezeExcite(nn.Module):
    """Channel attention block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)          # squeeze
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))           # excite
        return x * w
