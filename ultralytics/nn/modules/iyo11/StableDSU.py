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
##E4  效果最好
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         # gate: 生成“可增强可抑制”的mask（tanh）
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=False),
#         )
#
#         # 关键：稳定起步，避免一上来就把纹理噪声抬起来
#         self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#     def forward(self, x):
#         x_low = self.compress(x)
#         mask_low = torch.tanh(self.gate_conv(x_low))          # [-1, 1]
#         out_low = x_low * (1 + self.gamma * mask_low)         # 可增强可抑制，且初始≈x_low
#
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#         return self.refine(out)

#################################################################################################
##E4 + DWConv → DWConv+PWConv+残差
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=False),
#         )
#
#         self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#         # ONLY change: refine upgraded
#         self.refine_dw = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#         self.refine_pw = nn.Conv2d(c2, c2, 1, bias=False)
#
#     def forward(self, x):
#         x_low = self.compress(x)
#         mask_low = torch.tanh(self.gate_conv(x_low))
#         out_low = x_low * (1 + self.gamma * mask_low)
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#
#         y = self.refine_pw(self.refine_dw(out))
#         return out + y   # residual refine


#################################################################################################
##E4 + 只改“gate 做 residual”（仍然 BN + refine=DW 不变）
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=False),
#         )
#
#         self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#     def forward(self, x):
#         x_low = self.compress(x)
#
#         # main
#         main = F.interpolate(x_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#
#         # residual gate (ONLY change vs E4)
#         mask_low = torch.tanh(self.gate_conv(x_low))  # [-1, 1]
#         res = F.interpolate(mask_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#
#         out = main + self.gamma * res
#         return self.refine(out)


#################################################################################################
##E6 + PixelShuffle
#################################################################################################
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#
#         # 1. 通道压缩
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         # 2. 动态门控生成 (保持低分辨率下的空间增强控制)
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=False),
#         )
#         self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#         # 3. 关键修改：为 PixelShuffle 准备的通道扩充层
#         # PixelShuffle 会将通道维度 (C * scale^2) 搬运到空间维度 (H*scale, W*scale)
#         # 为了保证输出通道依然是 c2，这里输入通道需要扩充到 c2 * (scale**2)
#         self.up_conv = nn.Sequential(
#             nn.Conv2d(c2, c2 * (scale ** 2), 3, padding=1, groups=1, bias=False),
#             nn.BatchNorm2d(c2 * (scale ** 2)),
#             nn.SiLU()
#         )
#         self.pixel_shuffle = nn.PixelShuffle(scale)
#
#         # 4. 细节修正层 (针对 PixelShuffle 可能产生的棋盘效应进行平滑)
#         # 建议这里使用正常的卷积而非深度卷积，或者深度卷积+逐点卷积，以增强特征融合
#         self.refine = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.Conv2d(c2, c2, 1, bias=False)
#         )
#
#     def forward(self, x):
#         # 低分辨特征提取与增强
#         x_low = self.compress(x)
#         mask_low = torch.tanh(self.gate_conv(x_low))
#         out_low = x_low * (1 + self.gamma * mask_low)
#
#         # 亚像素上采样
#         out_high_res = self.up_conv(out_low)
#         out = self.pixel_shuffle(out_high_res)
#
#         # 最终细节修复
#         return self.refine(out)

#################################################################################################
##E7
#################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

def icnr_init(weight, scale=2):
    # weight: [out_c, in_c, k, k], out_c should be c2 * scale^2
    out_c, in_c, k1, k2 = weight.shape
    r = scale ** 2
    if out_c % r != 0:
        return
    sub_c = out_c // r
    w = torch.randn(sub_c, in_c, k1, k2)
    w = w.repeat_interleave(r, dim=0)
    with torch.no_grad():
        weight.copy_(w)

class StableDSU(nn.Module):
    def __init__(self, c1, c2, scale=2, use_icnr=True):
        super().__init__()
        self.scale = scale

        self.compress = nn.Conv2d(c1, c2, 1, bias=False)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1, bias=False),
        )
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))

        # 方案C：DW + PW 生成 c2*s^2（更稳/更轻），你也可以换回你原来的普通3x3
        out_c = c2 * (scale ** 2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(),
        )

        # 如果你坚持用原始普通3x3版本，用这个替换上面 up_conv 即可：
        # self.up_conv = nn.Sequential(
        #     nn.Conv2d(c2, out_c, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_c),
        #     nn.SiLU(),
        # )

        self.pixel_shuffle = nn.PixelShuffle(scale)

        # 改动B：refine 用普通3x3 + 激活（更抹格栅）
        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1, bias=False),
        )

        if use_icnr:
            # 找到生成 out_c 的那一层 Conv 权重来 init
            # 这里是 up_conv 的最后一个 Conv2d(1x1)
            conv = self.up_conv[3]
            icnr_init(conv.weight, scale=scale)

    def forward(self, x):
        x_low = self.compress(x)
        mask_low = torch.tanh(self.gate_conv(x_low))

        # 保持你原始：乘法改写主干
        out_low = x_low * (1.0 + self.gamma * mask_low)

        out = self.pixel_shuffle(self.up_conv(out_low))
        return self.refine(out)
