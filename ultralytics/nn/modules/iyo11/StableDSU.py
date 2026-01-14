#################################################################################################
##E0
#################################################################################################
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




#################################################################################################
##E1
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#         # 预先降维，减少后续计算量
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         # 改进的轻量化门控：在低分辨率下完成卷积
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#     def forward(self, x):
#         # 1. 先压缩通道
#         x_low = self.compress(x)
#         # 2. 在低分辨率下生成 mask (省显存/计算量)
#         mask_low = self.gate_conv(x_low)
#         # 3. 应用 mask
#         out_low = x_low * (1 + mask_low)
#         # 4. 统一进行一次上采样
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#         # 5. 最后平滑
#         return self.refine(out)
#


#################################################################################################
##E2
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class StableDSU(nn.Module):
#     """
#     Version B (更实用的保守版):
#     - gamma_gate = 0.05 近 0-init：既稳又容易学出收益
#     - gate 用 sigmoid 并中心化：梯度更友好
#     - refine 继续 0-init：避免 RSOD 被细化卷积伤到
#     """
#     def __init__(self, c1, c2, scale=2, gn=False):
#         super().__init__()
#         self.scale = scale
#
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         Norm = (lambda c: nn.GroupNorm(32, c)) if gn else (lambda c: nn.BatchNorm2d(c))
#
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             Norm(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=True),
#             nn.Sigmoid()
#         )
#
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#         # 门控强度：给一个很小的初值，避免“学不起来”
#         self.gamma_gate = nn.Parameter(torch.ones(1, c2, 1, 1) * 0.05)
#         # 细化强度：继续 0-init，减少跨数据集副作用
#         self.gamma_ref = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#     def forward(self, x):
#         x_low = self.compress(x)
#
#         gate = self.gate_conv(x_low)          # (0, 1)
#         gate = gate - 0.5                     # 中心化到 (-0.5, 0.5)
#
#         out_low = x_low * (1.0 + self.gamma_gate * gate)
#
#         out = F.interpolate(out_low, scale_factor=self.scale,
#                             mode='bilinear', align_corners=False)
#
#         out = out + self.gamma_ref * self.refine(out)
#         return out


#################################################################################################
##E3
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# def make_gn(c, max_groups=32):
#     g = min(max_groups, c)
#     while c % g != 0 and g > 1:
#         g -= 1
#     return nn.GroupNorm(g, c)
#
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2, use_gn=True):
#         super().__init__()
#         self.scale = scale
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         Norm = (lambda c: make_gn(c)) if use_gn else (lambda c: nn.BatchNorm2d(c))
#
#         # gate in low-res (keep E1 spirit)
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             Norm(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=True),   # keep bias
#             nn.Sigmoid()
#         )
#
#         # refine: keep DWConv but make it residual+controllable
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#         self.alpha = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12
#
#         # init: make gate start neutral (sigmoid bias ~ 0 => 0.5)
#         nn.init.zeros_(self.gate_conv[-2].bias)
#
#     def forward(self, x):
#         x_low = self.compress(x)
#
#         mask = self.gate_conv(x_low)     # (0,1)
#
#         # KEY: allow suppress/enhance, starts neutral at ~1
#         gate = 2.0 * mask                # (0,2)
#         out_low = x_low * gate
#
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#
#         w = torch.sigmoid(self.alpha)    # (0,1)
#         out = out + w * self.refine(out)
#         return out


#################################################################################################
##E4
#################################################################################################\
import torch
import torch.nn as nn
import torch.nn.functional as F

class StableDSU(nn.Module):
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.scale = scale

        self.compress = nn.Conv2d(c1, c2, 1, bias=False)

        # gate: 生成“可增强可抑制”的mask（tanh）
        self.gate_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1, bias=False),
        )

        # 关键：稳定起步，避免一上来就把纹理噪声抬起来
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))

        self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)

    def forward(self, x):
        x_low = self.compress(x)
        mask_low = torch.tanh(self.gate_conv(x_low))          # [-1, 1]
        out_low = x_low * (1 + self.gamma * mask_low)         # 可增强可抑制，且初始≈x_low
        out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return self.refine(out)
