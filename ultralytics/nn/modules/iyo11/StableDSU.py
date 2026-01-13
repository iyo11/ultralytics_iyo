# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#
#         self.semantic_path = nn.Sequential(
#             nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
#             nn.Conv2d(c1, c2, 1, bias=False)
#         )
#
#         self.detail_gate = nn.Sequential(
#             nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
#             nn.BatchNorm2d(c1),
#             nn.SiLU(),
#             nn.Conv2d(c1, c2, 1, bias=False),
#             nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
#             nn.Sigmoid()
#         )
#
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#     def forward(self, x):
#         base_feat = self.semantic_path(x)
#         detail_mask = self.detail_gate(x)
#         out = base_feat * (1 + detail_mask)
#         return self.refine(out)


import torch
import torch.nn as nn
import torch.nn.functional as F


class StableDSU(nn.Module):
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.scale = scale

        # 1. 预投影：在低分辨率（H, W）下完成通道变换
        # 原版在 (H*scale, W*scale) 下做 1x1 卷积，开销是现在的 scale^2 倍
        self.proj = nn.Conv2d(c1, c2, 1, bias=False)

        # 2. 细节分支优化：采用轻量化深度可分离结构
        self.detail_gate = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 1, bias=False),
            # 这里删除内部的 Upsample，移至 forward 中统一处理
            nn.Sigmoid()
        )

        # 3. 细化层：保持深度卷积，减少参数和计算量
        self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)

    def forward(self, x):
        # 首先降低通道维度 (c1 -> c2)，此时空间尺寸仍为 (H, W)
        x_proj = self.proj(x)

        # 在低分辨率下计算 Mask，进一步节省计算量
        gate = self.detail_gate(x_proj)

        # 统一进行上采样
        # 将 base 和 gate 合并在一起上采样，或者分别采样
        # 为了保证效果，我们先对基础特征和 Mask 分别上采样
        base_feat = F.interpolate(x_proj, scale_factor=self.scale, mode='bilinear', align_corners=False)
        detail_mask = F.interpolate(gate, scale_factor=self.scale, mode='bilinear', align_corners=False)

        # 融合与残差增强
        out = base_feat * (1 + detail_mask)
        return self.refine(out)