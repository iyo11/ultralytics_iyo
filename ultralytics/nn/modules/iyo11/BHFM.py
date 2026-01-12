import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    """标准卷积: Conv + BN + SiLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        # 自动计算 padding
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
    """
    正交/空间分离卷积块
    结构: 1xK (Conv) -> Kx1 (Conv) -> BN -> SiLU
    """

    def __init__(self, c, kernel_size=3, dilation=1):
        super().__init__()

        # 定义 padding
        # 1xK: 高度不变，宽度需 padding
        pad_w = ((kernel_size - 1) * dilation) // 2
        # Kx1: 宽度不变，高度需 padding
        pad_h = ((kernel_size - 1) * dilation) // 2

        # 1. 先进行 1xK 卷积 (处理宽度方向信息)
        self.conv_1xk = nn.Conv2d(
            c, c,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, pad_w),
            dilation=dilation,
            bias=False
        )

        # 2. 再进行 Kx1 卷积 (处理高度方向信息)
        self.conv_kx1 = nn.Conv2d(
            c, c,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(pad_h, 0),
            dilation=dilation,
            bias=False
        )

        # 3. 统一做一次 BN 和 激活，减少中间层碎片化
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 串行流过 1x3 -> 3x1
        x = self.conv_1xk(x)
        x = self.conv_kx1(x)
        return self.act(self.bn(x))


class CoordAtt(nn.Module):
    """
    Coordinate Attention (CoordAtt) - 比 SE 更强
    分别在 H 和 W 维度进行 Attention，保留位置信息
    """

    def __init__(self, inp, reduction=32):
        super().__init__()
        # 这里的 reduction 可以适当调大，减少参数
        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # H 方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # W 方向池化

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)  # 原论文用 h_swish，但在 YOLO 中 SiLU 更通用

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 1. 分解 Pooling
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, w, 1] -> 这里的 permute 是为了拼接

        # 2. 拼接 + 降维
        y = torch.cat([x_h, x_w], dim=2)  # [n, c, h+w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 3. 拆分 + 升维
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 转回 [n, c, 1, w]

        # 4. 生成 Attention Map
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 5. 双向加权
        out = identity * a_h * a_w
        return out


class BHFM(nn.Module):
    """
    BHFM Ultimate Version
    并行正交卷积 + 坐标注意力
    """

    def __init__(self, c1, c2):
        super().__init__()
        # 输入投影
        self.input_proj = None
        if c1 != c2:
            self.input_proj = BasicConv(c1, c2, kernel_size=1)

        c = c2

        # === 分支 1: 标准 3x3 卷积 ===
        # 扎实的局部特征
        self.branch_std = BasicConv(c, c, kernel_size=3)

        # === 分支 2: 空洞正交卷积 (Dilation=2) ===
        # 1x3(d=2) + 3x1(d=2) -> 等效感受野 5x5
        self.branch_dilated = OrthoBlock(c, kernel_size=3, dilation=2)

        # === 分支 3: 标准正交卷积 ===
        # 1x3 + 3x1 -> 轻量级 3x3 特征
        self.branch_ortho = OrthoBlock(c, kernel_size=3, dilation=1)

        # === 融合层 ===
        # 将 3 个分支 (3c) 融合回 c
        self.fusion = nn.Sequential(
            nn.Conv2d(c * 3, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )

        # === 强力注意力 (CoordAtt) ===
        self.attention = CoordAtt(c)

    def forward(self, x):
        # 0. 输入对齐
        if self.input_proj is not None:
            x = self.input_proj(x)

        # 1. 并行提取特征
        feat_std = self.branch_std(x)
        feat_dil = self.branch_dilated(x)
        feat_ort = self.branch_ortho(x)

        # 2. 拼接融合
        # shape: [b, 3c, h, w] -> [b, c, h, w]
        concat_feat = torch.cat([feat_std, feat_dil, feat_ort], dim=1)
        fused_feat = self.fusion(concat_feat)

        # 3. 强力注意力加权
        # CoordAtt 会同时在 H 和 W 方向打分
        attn_feat = self.attention(fused_feat)

        # 4. 残差连接 (分支 4)
        # 原始输入 x + 处理后的特征
        return x + attn_feat