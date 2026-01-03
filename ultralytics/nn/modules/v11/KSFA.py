import torch
import torch.nn as nn


class KernelSelectiveFusionAttention(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        # 降维后的中间维度
        d = max(dim // r, L)

        # 1. 空间特征提取分支
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)

        # 2. 特征压缩层
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

        # 3. 融合与通道注意力权重生成
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, dim, _, _ = x.shape

        # 特征提取
        feat1 = self.conv0(x)
        feat2 = self.conv_spatial(feat1)

        # 降道
        attn1 = self.conv1(feat1)
        attn2 = self.conv2(feat2)

        # 空间特征汇聚 (这里简化了 agg 的逻辑以匹配 dim//2 的设计)
        attn_cat = torch.cat([attn1, attn2], dim=1)  # (B, dim, H, W)
        avg_attn = torch.mean(attn_cat, dim=1, keepdim=True)  # (B, 1, H, W)

        # 通道权重计算
        ch_pool = self.global_pool(attn_cat)
        z = self.fc1(ch_pool)
        a_b = self.fc2(z).reshape(batch_size, 2, dim // 2, 1, 1)
        a_b = self.softmax(a_b)

        # 拆分权重并融合
        a1, a2 = a_b[:, 0], a_b[:, 1]
        fused_attn = attn1 * a1 * avg_attn + attn2 * a2 * avg_attn

        # 生成最终注意力图并作用于输入
        final_attn = self.conv(fused_attn).sigmoid()
        return x * final_attn