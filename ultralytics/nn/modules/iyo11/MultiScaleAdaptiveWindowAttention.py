import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import autopad

__all__ = ['MultiScaleAdaptiveWindowAttention']


def channel_shuffle(x, groups):
    """通道重组：打乱分组卷积后的特征，增强信息流动"""
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    return x.view(batchsize, -1, height, width)


class MultiScaleAdaptiveWindowAttention(nn.Module):
    # YOLO 会自动传入 c1 (输入通道), c2 (输出通道)
    def __init__(self, c1, c2, window_sizes=(3, 5), reduction=8):
        super().__init__()

        # 1. 首先定义所有基础属性，防止 AttributeError
        self.groups = 8
        self.window_sizes = [window_sizes] if isinstance(window_sizes, int) else list(window_sizes)
        self.inter_dim = max(c2 // reduction, 16)

        # 2. 极致压缩计算量的投影层 (使用分组卷积)
        self.proj_in = nn.Sequential(
            nn.Conv2d(c1, self.inter_dim, 1, groups=self.groups, bias=False),
            nn.BatchNorm2d(self.inter_dim),
            nn.SiLU()
        )

        # 3. 轻量化多尺度 DW 卷积
        self.attn_convs = nn.ModuleList([
            nn.Conv2d(self.inter_dim, self.inter_dim, k, padding=autopad(k, None),
                      groups=self.inter_dim, bias=False)
            for k in self.window_sizes
        ])

        # 4. 尺度选择器
        self.scale_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inter_dim, len(self.window_sizes), 1, bias=False),
            nn.Softmax(dim=1)
        )

        # 5. 极致轻量化门控 (DW + PW 结构)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False),
            nn.Conv2d(2, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # 6. 输出投影 (c2 确保与 YOLO 下一层匹配)
        self.proj_out = nn.Sequential(
            nn.Conv2d(self.inter_dim, c2, 1, groups=self.groups, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.alpha = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        B, C, H, W = x.shape

        # 通道压缩 + Shuffle (计算量大幅下降)
        x_reduced = self.proj_in(x)
        if C >= self.groups:  # 确保可以 shuffle
            x_reduced = channel_shuffle(x_reduced, self.groups)

        # 多尺度特征提取 + 尺寸强制对齐 (解决 8 vs 9 RuntimeError)
        feats = []
        for conv in self.attn_convs:
            f = conv(x_reduced)
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode='nearest')
            feats.append(f)

        # 动态尺度融合
        scale_weights = self.scale_fc(x_reduced)
        fused = 0
        for i in range(len(self.window_sizes)):
            fused = fused + feats[i] * scale_weights[:, i:i + 1]

        # 空间注意力门控
        avg_out = torch.mean(fused, dim=1, keepdim=True)
        max_out, _ = torch.max(fused, dim=1, keepdim=True)
        spatial_mask = torch.cat([avg_out, max_out], dim=1)
        gate = self.spatial_gate(spatial_mask)

        # 还原维度 + 残差连接
        out = self.proj_out(fused * gate)

        # 如果输入输出通道不一致（通常在 YOLO 中 c1=c2），这里会自动处理
        return (x + self.alpha * out) if C == out.shape[1] else out