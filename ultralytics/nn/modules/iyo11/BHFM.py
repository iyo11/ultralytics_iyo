import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力机制 (保持不变)
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicConv(nn.Module):
    """
    基础卷积块：Conv + BN + SiLU
    完全替换之前的 OrthoConv，使用普通卷积 (groups=1)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        # 计算 padding 以保持特征图尺寸不变
        # Padding = (Kernel_size - 1) * dilation / 2
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False  # 有BN通常不需要bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FusionModule(nn.Module):
    """
    融合模块：Concat -> Conv1x1 -> BN -> SiLU
    用于融合 原始特征(A) 和 处理后特征(B)
    """

    def __init__(self, c):
        super().__init__()
        # 输入是 A(c) + B(c) = 2c -> 输出 c
        self.fusion_conv = nn.Conv2d(c * 2, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x_orig, x_processed):
        # Concat
        cat_feat = torch.cat([x_processed, x_orig], dim=1)
        # 融合 + 降维
        return self.act(self.bn(self.fusion_conv(cat_feat)))


class BHFM(nn.Module):
    """
    BHFM 改良版：使用普通卷积 (Standard Convolution)
    """

    def __init__(self, c1, c2):
        super().__init__()
        # 假设输入输出通道一致，如果需要在YOLO yaml中使用，通常 c1 != c2 时需要处理
        # 这里为了模块内部稳定性，统一映射到 c2
        self.input_proj = None
        if c1 != c2:
            self.input_proj = nn.Conv2d(c1, c2, 1, bias=False)

        c = c2

        # === 1. 特征提取分支 (改为普通卷积) ===

        # Step 1: 标准 3x3 卷积
        self.conv3 = BasicConv(c, c, kernel_size=3, dilation=1)
        self.fusion1 = FusionModule(c)

        # Step 2: 标准 3x3 膨胀卷积 (Dilation=2, 感受野类比 5x5)
        self.conv5 = BasicConv(c, c, kernel_size=3, dilation=2)
        self.fusion2 = FusionModule(c)

        # === 2. 聚合与注意力 ===

        # 最终整合特征的 1x1 卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )

        # 通道注意力
        self.channel_attn = ChannelAttention(c)

    def forward(self, x):
        # 通道对齐
        if self.input_proj is not None:
            x = self.input_proj(x)

        # --- 阶梯式处理流程 ---

        # Step 1: 3x3 分支
        feat3 = self.conv3(x)
        # 融合: 输入 x 和 feat3
        node_1 = self.fusion1(x_orig=x, x_processed=feat3)

        # Step 2: 5x5 (dilation) 分支
        # 注意：这里输入是上一级的输出 node_1
        feat5 = self.conv5(node_1)
        # 融合: 输入 node_1 和 feat5
        node_2 = self.fusion2(x_orig=node_1, x_processed=feat5)

        # --- 输出阶段 ---

        # 1. 整理特征
        refined_feat = self.final_conv(node_2)

        # 2. 计算注意力权重
        attn_out = self.channel_attn(refined_feat)

        # 3. 最终残差 (Scale + Add)
        # 使用注意力加权后的特征 + 原始输入
        out = x + (refined_feat * attn_out)

        return out