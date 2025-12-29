import torch
import torch.nn as nn

__all__ = ['PoolFormerMSConvBlockV3']


# -------------------------
# 1. 极简高效的通道注意力 (ECA)
# -------------------------
class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


# -------------------------
# 2. 改进的多尺度异构并行分支 (Split-Parallel MS-Conv)
# -------------------------
class SplitParallelMSConv(nn.Module):
    def __init__(self, dim, k_large=5, k_strip=7):
        super().__init__()
        # 确保 dim 能够被 2 整除
        self.dim1 = dim // 2
        self.dim2 = dim - self.dim1

        # --- 分支1: 多尺度局部特征 (针对小目标) ---
        self.dw3 = nn.Conv2d(self.dim1, self.dim1, 3, padding=1, groups=self.dim1, bias=False)
        self.dw5 = nn.Conv2d(self.dim1, self.dim1, k_large, padding=k_large // 2, groups=self.dim1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim1)

        # --- 分支2: 条带卷积 (针对中等目标的结构特征) ---
        self.strip_h = nn.Conv2d(self.dim2, self.dim2, (1, k_strip), padding=(0, k_strip // 2), groups=self.dim2,
                                 bias=False)
        self.strip_v = nn.Conv2d(self.dim2, self.dim2, (k_strip, 1), padding=(k_strip // 2, 0), groups=self.dim2,
                                 bias=False)
        self.bn2 = nn.BatchNorm2d(self.dim2)

        # --- 融合层: 使用 1x1 卷积聚合 Shuffle 后的特征 ---
        self.fuse_conv = nn.Conv2d(dim, dim, 1, groups=1, bias=False)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)

        # 支路 1 融合
        x1 = self.dw3(x1) + self.dw5(x1)
        x1 = self.bn1(x1)

        # 支路 2 融合
        x2 = self.strip_h(x2) + self.strip_v(x2)
        x2 = self.bn2(x2)

        out = torch.cat([x1, x2], dim=1)

        # Channel Shuffle: 混合两路信息
        b, c, h, w = out.shape
        out = out.view(b, 2, c // 2, h, w).transpose(1, 2).contiguous().view(b, c, h, w)

        return self.fuse_conv(out)


# -------------------------
# 3. 增强型 MLP (MSConvStarV3)
# -------------------------
class MSConvStarV3(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        # 强制 hidden 为偶数防止 chunk 失败
        hidden = (int(dim * mlp_ratio) // 2) * 2
        self.fc1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        self.ms_conv = SplitParallelMSConv(hidden)
        self.act = nn.SiLU()
        self.eca = ECA(kernel_size=3)

        # 空间门控：改用 3x3 卷积增强对小目标的局部空间聚焦
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.fc2 = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        # 内部残差逻辑由 PoolFormerBlock 处理，这里做变换
        x = self.act(self.bn1(self.fc1(x)))
        x = x + self.ms_conv(x)
        x = x * self.spatial_gate(x)
        x = self.eca(x)
        x = self.bn2(self.fc2(x))
        return x


# -------------------------
# 4. 核心 Block V3
# -------------------------
class PoolFormerMSConvBlockV3(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, args=None):
        super().__init__()
        # args 默认格式: [pool_size, mlp_ratio]
        if args is None:
            args = [3, 4.0]

        pool_size = int(args[0])
        mlp_ratio = float(args[1])

        self.dim = c1
        pad = pool_size // 2
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pad, count_include_pad=False)

        # 可学习的缩放因子
        self.token_scale = nn.Parameter(torch.ones(1) * 1e-4)
        self.mlp_scale = nn.Parameter(torch.ones(1) * 1e-4)

        self.mlp = MSConvStarV3(self.dim, mlp_ratio=mlp_ratio)

    def forward(self, x):
        # 1. Token Mixing (Pooling)
        x = x + self.token_scale * (self.pool(x) - x)
        # 2. Channel Mixing (MSConvStarV3)
        x = x + self.mlp_scale * self.mlp(x)
        return x