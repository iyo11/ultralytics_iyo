import torch
import torch.nn as nn
import torch.nn.functional as F


class StableDSU(nn.Module):
    """
    Stable Dynamic Semantic Upsampler (Stable-DSU)
    借鉴 DySample 思想，但去除了非确定性的 grid_sample。
    采用内容感知门控机制，确保在固定种子下 100% 可复现。
    """

    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. 语义路径：保持全局语义稳定
        self.semantic_path = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        # 2. 动态细节路径：学习边缘和细节的增强
        # 使用 Depthwise 卷积减少参数，同时捕捉局部特征
        self.detail_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Sigmoid()  # 输出 0-1 之间的权重系数
        )

        # 3. 特征细化
        self.refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)

    def forward(self, x):
        # 基础语义特征
        base_feat = self.semantic_path(x)

        # 生成内容自适应权重（类似 DySample 的采样意图，但以权重形式体现）
        detail_mask = self.detail_gate(x)

        # 动态融合：基础特征 * 权重增强
        # 这样做可以确保：如果模型学不到东西，detail_mask 趋近 0.5，退化为普通上采样，非常稳定
        out = base_feat * (1 + detail_mask)

        return self.refine(out)