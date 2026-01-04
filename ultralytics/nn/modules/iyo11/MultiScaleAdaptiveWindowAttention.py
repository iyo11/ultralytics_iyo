import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MultiScaleAdaptiveWindowAttention']

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAdaptiveWindowAttention(nn.Module):
    def __init__(self, dim, window_sizes=(3, 5, 7), reduction=16):
        super().__init__()
        # 强制 dim 必须是 8 的倍数（YOLO 默认符合）
        self.dim = dim
        self.window_sizes = [window_sizes] if isinstance(window_sizes, int) else list(window_sizes)

        # 1. 核心卷积：全部使用 Depthwise 卷积，参数量极低
        self.attn_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            for k in self.window_sizes
        ])

        # 2. 尺度选择器：将中间通道数压缩到极小 (如 256 -> 16)
        mid_dim = max(dim // reduction, 8)
        self.scale_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, len(self.window_sizes), 1, bias=False)
        )

        # 3. 空间门控：使用通道池化代替卷积提取空间特征，仅用一个 7x7 DW 卷积
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 4. 最后投影：改为 Pointwise 卷积或直接省略。
        # 这里为了保持特征一致性，只使用一个最基本的 Scale 缩放
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 多尺度 DW 卷积提取
        feats = [conv(x) for conv in self.attn_convs]

        # 计算尺度权重并 Softmax
        # 结果形状: [B, 窗口数, 1, 1, 1]
        scale_weights = self.scale_fc(x).view(B, len(self.window_sizes), 1, 1, 1)
        scale_weights = torch.softmax(scale_weights, dim=1)

        # 融合特征
        fused = 0
        for i, feat in enumerate(feats):
            fused = fused + feat * scale_weights[:, i]

        # 空间注意力门控 (Channel Pool: Mean & Max)
        avg_out = torch.mean(fused, dim=1, keepdim=True)
        max_out, _ = torch.max(fused, dim=1, keepdim=True)
        spatial_mask = torch.cat([avg_out, max_out], dim=1)
        gate = self.sigmoid(self.spatial_conv(spatial_mask))

        # 最终输出 = 原始残差 + 增强特征
        return x + self.alpha * (fused * gate)
# 测试不同输入尺寸
if __name__ == "__main__":
    # 定义模块
    msa = MultiScaleAdaptiveWindowAttention(dim=256, window_sizes=(3,5,7), reduction=4)

    # 模拟输入特征图: B=1, C=256, H=32, W=32
    x = torch.randn(1, 256, 32, 32)
    y = msa(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    # 测试其他尺寸
    for H, W in [(64, 64), (33, 33), (128, 128)]:
        x = torch.randn(1, 256, H, W)
        y = msa(x)
        print(f"Input shape: {x.shape} -> Output shape: {y.shape}")