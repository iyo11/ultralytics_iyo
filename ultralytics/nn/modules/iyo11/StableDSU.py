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
        # 预先降维，减少后续计算量
        self.compress = nn.Conv2d(c1, c2, 1, bias=False)

        # 改进的轻量化门控：在低分辨率下完成卷积
        self.gate_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.Sigmoid()
        )

        self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)

    def forward(self, x):
        # 1. 先压缩通道
        x_low = self.compress(x)
        # 2. 在低分辨率下生成 mask (省显存/计算量)
        mask_low = self.gate_conv(x_low)
        # 3. 应用 mask
        out_low = x_low * (1 + mask_low)
        # 4. 统一进行一次上采样
        out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
        # 5. 最后平滑
        return self.refine(out)
