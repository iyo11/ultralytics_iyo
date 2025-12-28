import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['PoolFormerMSConvBlockV2']


# -------------------------
# 1. 极简高效的通道注意力 (ECA)
# -------------------------
class ECA(nn.Module):
    """ECA-Net: Efficient Channel Attention, 避免降维，适合增强通道交互"""

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
# 2. 改进的多尺度卷积 (Parallel MS-Conv) - 小目标优化版
# -------------------------
class ParallelMSConv(nn.Module):
    def __init__(self, dim, k_large=5):
        super().__init__()
        # 分支1：标准 3x3 DW (捕捉微小细节)
        self.dw3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 分支2：大核 (优化点：移除 dilation)
        # 针对小目标，contiguous (连续) 的感受野比 dilated (空洞) 更好。
        # 使用 5x5 或 7x7 纯卷积，不使用 dilation=2，避免小目标落入空洞。
        padding = (k_large - 1) // 2
        self.dw_large = nn.Sequential(
            nn.Conv2d(dim, dim, k_large, padding=padding, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 分支3：全局特征 (保留，用于提供背景上下文，抑制误检)
        self.global_branch = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x3 = self.dw3(x)
        x_large = self.dw_large(x)
        # 全局分支广播回原尺寸
        x_global = self.global_branch(x)

        return x3 + x_large + x_global


# -------------------------
# 3. 增强型 MLP (接口微调)
# -------------------------
class MSConvStarV2(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        # 优化点：这里的大核可以根据需求调整，默认为5，适合中小目标
        self.ms_conv = ParallelMSConv(hidden, k_large=5)

        self.act = nn.SiLU()
        self.eca = ECA(kernel_size=3)  # ECA k=3 适合捕捉局部通道交互
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid()
        )
        self.fc2 = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = x + self.ms_conv(x)
        x = x * self.spatial_gate(x)  # Star 风格门控
        x = self.eca(x)  # 通道增强
        x = self.bn2(self.fc2(x))
        return x


# -------------------------
# 4. 核心 Block V2
# -------------------------
class PoolFormerMSConvBlockV2(nn.Module):
    # args对应 yaml 中的 [pool_size, mlp_ratio, drop_path]
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, args=None):
        super().__init__()
        # 兼容 YOLO 的参数解析
        # 假设 args = [pool_size, mlp_ratio, drop_path]
        # 如果 args 没传，给默认值
        if args is None:
            args = [3, 4.0, 0.0]

        pool_size = int(args[0])
        mlp_ratio = float(args[1])
        drop_path = float(args[2]) if len(args) > 2 else 0.0

        self.dim = c1  # 输入通道数 (YOLO 中 c1=c2)

        # Token Mixer: Pooling
        # 优化点：针对极小目标，pool_size=3 还是可能丢失信息，可以考虑设为 1 (即不做 pooling 混合，只做 identity)
        # 但为了保留 PoolFormer 特性，建议保持 3
        pad = pool_size // 2
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pad, count_include_pad=False)
        self.token_scale = nn.Parameter(torch.ones(1) * 1e-4)

        # Channel Mixer: MSConvStar
        self.mlp = MSConvStarV2(self.dim, mlp_ratio=mlp_ratio)

    def forward(self, x):
        # Stage 1: High-pass like filtering
        pool_feat = self.pool(x)
        x = x + self.token_scale * (pool_feat - x)

        # Stage 2: Feature Extraction
        x = x + self.mlp(x)
        return x