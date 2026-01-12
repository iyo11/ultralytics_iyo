import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    """标准卷积: Conv + BN + SiLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class OrthoBlock(nn.Module):
    """正交/空间分离卷积 (1xK + Kx1)"""

    def __init__(self, c, kernel_size=3, dilation=1):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv_1xk = nn.Conv2d(c, c, kernel_size=(1, kernel_size), stride=1,
                                  padding=(0, pad), dilation=dilation, bias=False)
        self.conv_kx1 = nn.Conv2d(c, c, kernel_size=(kernel_size, 1), stride=1,
                                  padding=(pad, 0), dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv_1xk(x)
        x = self.conv_kx1(x)
        return self.act(self.bn(x))


class CBAM(nn.Module):
    """CBAM 注意力模块"""

    def __init__(self, c, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(c, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        hidden_planes = max(4, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ==========================================
#  BHFM (Split Version) - 低参数量版
# ==========================================
class BHFM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2

        # 1. 输入投影（如果通道数变了，或者单纯为了特征对齐）
        self.input_proj = None
        if c1 != c2:
            self.input_proj = BasicConv(c1, c2, kernel_size=1)

        # 2. 计算分组通道数
        # 我们把通道分成 3 份：[c_part, c_part, c_rest]
        # 确保除不尽的时候也能正常运行
        self.c_part = c2 // 3
        self.c_rest = c2 - (2 * self.c_part)

        # === 分支 1: 标准 3x3 (处理 1/3 通道) ===
        self.branch_std = BasicConv(self.c_part, self.c_part, kernel_size=3)

        # === 分支 2: 空洞正交 (处理 1/3 通道) ===
        self.branch_dilated = OrthoBlock(self.c_part, kernel_size=3, dilation=2)

        # === 分支 3: 标准正交 (处理剩余通道) ===
        self.branch_ortho = OrthoBlock(self.c_rest, kernel_size=3, dilation=1)

        # === 融合层 ===
        # Concat 后是 c2 通道，用 1x1 卷积做一次 Channel Mixing
        # 这里的参数量只有原来的 1/9 (因为输入只有 c 而不是 3c，且前面分支参数也减少了)
        self.fusion = BasicConv(c2, c2, kernel_size=1)

        # === CBAM 注意力 ===
        self.attention = CBAM(c2, ratio=16)

    def forward(self, x):
        # 0. 对齐通道
        if self.input_proj is not None:
            x = self.input_proj(x)

        # 1. Split (切分)
        # 按照初始化计算好的通道数切分
        x_std, x_dil, x_ort = torch.split(x, [self.c_part, self.c_part, self.c_rest], dim=1)

        # 2. Transform (并行处理)
        x_std = self.branch_std(x_std)
        x_dil = self.branch_dilated(x_dil)
        x_ort = self.branch_ortho(x_ort)

        # 3. Merge (拼接)
        x_cat = torch.cat([x_std, x_dil, x_ort], dim=1)

        # 4. Mix & Attention
        x_fused = self.fusion(x_cat)
        x_attn = self.attention(x_fused)

        # 5. Residual
        return x + x_attn