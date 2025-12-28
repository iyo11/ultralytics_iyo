import torch
import torch.nn as nn

from build.lib.ultralytics.nn.modules import Conv


class ECA(nn.Module):
    """Efficient Channel Attention module."""

    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        # 根据通道数自适应计算卷积核大小 k
        import math
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class DualDownsampling(nn.Module):
    """
    双分支下采样模块:
    Left: Standard Conv (k=3, s=2)
    Right: DSConv(1x1) -> DWConv(3x3, s=2) -> DSConv(1x1)
    Fusion: Add + ECA
    """

    def __init__(self, c1, c2, mc=None):
        super().__init__()
        if mc is None:
            mc = c2  # 中间通道数，默认为输出通道数

        # 左分支: 标准下采样路径
        self.cv_left = Conv(c1, c2, k=3, s=2)

        # 右分支: 细节增强路径
        # 注意: 根据图示, 右侧中间层是 DW 卷积 (g=mc)
        self.cv_right = nn.Sequential(
            Conv(c1, mc, k=1, s=1),
            Conv(mc, mc, k=3, s=2, g=mc),  # Stride=2 进行空间压缩
            Conv(mc, c2, k=1, s=1)
        )

        # 融合后的注意力机制
        self.eca = ECA(c2)

    def forward(self, x):
        # 左右分支相加并经过 ECA
        return self.eca(self.cv_left(x) + self.cv_right(x))