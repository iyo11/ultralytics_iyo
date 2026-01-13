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
    def __init__(self, c1, c2, scale=2, gn_groups=32):
        super().__init__()
        self.scale = scale

        self.semantic_path = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(c1, c2, 1, bias=False)
        )

        # GN
        g = min(gn_groups, c1)
        while c1 % g != 0 and g > 1:
            g -= 1
        self.norm = nn.GroupNorm(g, c1)

        # --- NEW: blur for high-frequency extraction (cheap DWConv) ---
        self.blur = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)

        # gate now takes high-frequency (x - blur(x))
        self.detail_gate = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            self.norm,
            nn.SiLU(),
            nn.Conv2d(c1, c2, 1, bias=True),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Tanh()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        # refine: spatial + channel mix
        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1, bias=False)
        )

    def forward(self, x):
        base_feat = self.semantic_path(x)

        # --- NEW: high-frequency residual for gate ---
        high = x - self.blur(x)
        gate = self.detail_gate(high)

        out = base_feat * (1.0 + self.gamma * gate)
        return self.refine(out)
