import torch
import torch.nn as nn
import torch.nn.functional as F


class StableDSU(nn.Module):
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.scale = scale

        self.semantic_path = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(c1, c2, 1, bias=False)
        )

        self.detail_gate = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )

        self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)

    def forward(self, x):
        base_feat = self.semantic_path(x)
        detail_mask = self.detail_gate(x)
        out = base_feat * (1 + detail_mask)
        return self.refine(out)
