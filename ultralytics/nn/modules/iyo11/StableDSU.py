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
##E4 nearest
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
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=False),
#         )
#
#         self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#     def forward(self, x):
#         x_low = self.compress(x)
#         mask_low = torch.tanh(self.gate_conv(x_low))
#         out_low = x_low * (1 + self.gamma * mask_low)
#
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='nearest')
#         return self.refine(out)

#################################################################################################
##E7  __gemini 设计
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class StableDSU(nn.Module):
#     """
#     E7: Strip-Gated DSU (针对小目标与长条目标的特化版)
#     继承 E4 的稳定性 (Tanh + Zero-init)，引入条形卷积解决几何劣势。
#     """
#
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#
#         # 1. 压缩通道 (保持高效)
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         # 2. 条形门控 (Strip Gating) - 解决长条目标问题
#         # 相比普通的 3x3，使用 5x1 和 1x5 并联
#         # k=5 可以看更远，且不会像 3x3 那样引入过多背景噪声
#         self.gate_h = nn.Conv2d(c2, c2, (1, 5), padding=(0, 2), groups=c2, bias=False)
#         self.gate_v = nn.Conv2d(c2, c2, (5, 1), padding=(2, 0), groups=c2, bias=False)
#
#         # 门控的后处理 (BN + SiLU + 1x1 融合)
#         self.gate_norm = nn.BatchNorm2d(c2)
#         self.gate_act = nn.SiLU()
#         self.gate_fusion = nn.Conv2d(c2, c2, 1, bias=False)
#
#         # E4 的核心 trick：零初始化 Gamma，保证训练初期极其稳定
#         self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#         # 3. 改进的 Refine：残差结构 - 解决小目标模糊问题
#         # 通过 res_refine 学习高频残差（锐化），而不是重构整个特征
#         self.refine_conv = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#         self.refine_gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#     def forward(self, x):
#         # [Step 1] 降维
#         x_low = self.compress(x)
#
#         # [Step 2] 条形特征提取 (并行提取横向和纵向上下文)
#         # 这样电线杆(纵向)和车辆(横向/块状)都能被捕捉，且互不干扰
#         g_h = self.gate_h(x_low)
#         g_v = self.gate_v(x_low)
#
#         # 融合特征生成 Mask
#         mask_feat = self.gate_fusion(self.gate_act(self.gate_norm(g_h + g_v)))
#
#         # 使用 Tanh 允许抑制背景 (-1) 和增强目标 (+1)
#         mask_low = torch.tanh(mask_feat)
#
#         # 应用门控 (E4 逻辑)
#         out_low = x_low * (1 + self.gamma * mask_low)
#
#         # [Step 3] 上采样 (Bilinear 最稳，不要动)
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#
#         # [Step 4] 细节修复 (Residual Refine)
#         # out 是模糊的，refine_conv 负责计算“锐化残差”
#         # 初始化为 0，让模型慢慢学着去锐化边缘
#         return out + self.refine_gamma * self.refine_conv(out)

#################################################################################################
##E8  ########2222
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class StableDSU(nn.Module):
#     """
#     E9: Serial-Strip DSU (串联版)
#     结构：Input -> [1x5 Conv] -> [5x1 Conv] -> ...
#     效果：
#     1. 感受野变成了实打实的满 5x5 矩形。
#     2. 相比并行版，背景信息会被充分融合（对于纹理型目标可能更好）。
#     """
#
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#
#         # 1. 压缩
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         # 2. 串联条形门控 (Sequential Strip Gating)
#         # 第一步：横向扫 (1x5)
#         self.gate_h = nn.Conv2d(c2, c2, (1, 5), padding=(0, 2), groups=c2, bias=False)
#         # 第二步：纵向扫 (5x1)
#         # 注意：这里把 gate_h 的输出作为输入，相当于把横向特征在纵向进行了扩散
#         self.gate_v = nn.Conv2d(c2, c2, (5, 1), padding=(2, 0), groups=c2, bias=False)
#
#         # 门控激活与融合
#         # 此时的特征已经具备了 5x5 的上下文
#         self.gate_norm = nn.BatchNorm2d(c2)
#         self.gate_act = nn.SiLU()
#         self.gate_fusion = nn.Conv2d(c2, c2, 1, bias=False)
#
#         # E4 Trick: 零初始化 Gamma
#         self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#         # 3. 细节 Refine (保持 E7 的残差结构，防止小目标丢失太严重)
#         self.refine_conv = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#         self.refine_gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#     def forward(self, x):
#         x_low = self.compress(x)
#
#         # === 串联处理 (Serial) ===
#         # 1. 先提取横向特征
#         feat_h = self.gate_h(x_low)
#         # 2. 基于横向特征，再提取纵向特征
#         # 这一步完成后，feat_v 中的每个点都包含了原始 x_low 中 5x5 区域的信息
#         feat_full = self.gate_v(feat_h)
#
#         # 生成 Mask
#         mask_raw = self.gate_fusion(self.gate_act(self.gate_norm(feat_full)))
#         mask_low = torch.tanh(mask_raw)
#
#         # 应用门控
#         out_low = x_low * (1 + self.gamma * mask_low)
#
#         # 上采样
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#
#         # 残差锐化
#         return out + self.refine_gamma * self.refine_conv(out)

#################################################################################################
##E9
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class StableDSU(nn.Module):
#     """
#     E10: Hybrid-Strip DSU (混合版)
#     结构：融合了三种不同的感受野提取方式，生成更鲁棒的 Mask。
#     分支 1: 3x3 DWConv (局部密集)
#     分支 2: 1x5 -> 5x1 (串行，模拟 5x5 满感受野)
#     分支 3: 1x5 + 5x1 (并行，十字形轴向特征)
#     """
#
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#
#         # 1. 压缩 (Input Projection)
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         # === Branch 1: 3x3 Local (局部特征) ===
#         self.branch_local = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#         # === Branch 2: Serial Strip (串行：模拟大核) ===
#         # 先扫横向，结果再扫纵向
#         self.branch_serial_h = nn.Conv2d(c2, c2, (1, 5), padding=(0, 2), groups=c2, bias=False)
#         self.branch_serial_v = nn.Conv2d(c2, c2, (5, 1), padding=(2, 0), groups=c2, bias=False)
#
#         # === Branch 3: Parallel Strip (并行：十字特征) ===
#         # 独立的横向和纵向，保留轴向信息
#         self.branch_parallel_h = nn.Conv2d(c2, c2, (1, 5), padding=(0, 2), groups=c2, bias=False)
#         self.branch_parallel_v = nn.Conv2d(c2, c2, (5, 1), padding=(2, 0), groups=c2, bias=False)
#
#         # === 门控生成 (Mask Generation) ===
#         self.gate_norm = nn.BatchNorm2d(c2)
#         self.gate_act = nn.SiLU()
#         self.gate_fusion = nn.Conv2d(c2, c2, 1, bias=False)
#
#         # E4 Trick: 零初始化 Gamma (控制门控初始权重)
#         self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#         # 3. 细节 Refine (残差锐化)
#         self.refine_conv = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#         self.refine_gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#     def forward(self, x):
#         # 1. 降维
#         x_low = self.compress(x)
#
#         # 2. 多分支特征提取
#         # --- 分支1: 3x3 ---
#         feat_local = self.branch_local(x_low)
#
#         # --- 分支2: 串行 (Serial) ---
#         # 逻辑：Input -> [1x5] -> [5x1]
#         feat_serial = self.branch_serial_v(self.branch_serial_h(x_low))
#
#         # --- 分支3: 并行 (Parallel) ---
#         # 逻辑：Input -> [1x5] + Input -> [5x1]
#         feat_parallel = self.branch_parallel_h(x_low) + self.branch_parallel_v(x_low)
#
#         # 3. 特征融合 (Sum Fusion)
#         # 将三种视角的上下文叠加
#         feat_combined = feat_local + feat_serial + feat_parallel
#
#         # 4. 生成 Mask
#         mask_raw = self.gate_fusion(self.gate_act(self.gate_norm(feat_combined)))
#         mask_low = torch.tanh(mask_raw)
#
#         # 5. 应用门控
#         out_low = x_low * (1 + self.gamma * mask_low)
#
#         # 6. 上采样
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#
#         # 7. 残差锐化
#         return out + self.refine_gamma * self.refine_conv(out)

#################################################################################################
##E9
##################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(c, gn_groups=32):
    # 让 groups 可整除通道数，避免报错
    g = min(gn_groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)


class StableDSU(nn.Module):
    """
    E10: Hybrid-Strip DSU (混合版)
    结构：融合三种感受野提取方式，生成更鲁棒的 Mask。
    分支1: 3x3 普通Conv（局部密集）
    分支2: 1x5 -> 5x1（串行，模拟 5x5 满感受野）
    分支3: 1x7 + 7x1（并行，十字形轴向特征）
    """

    def __init__(self, c1, c2, scale=2, norm="gn", gn_groups=32):
        super().__init__()
        self.scale = scale

        # 1) 压缩
        self.compress = nn.Conv2d(c1, c2, 1, bias=False)

        # Branch 1: 3x3 普通卷积（非 depthwise）
        self.branch_local = nn.Conv2d(c2, c2, 3, padding=1, groups=1, bias=False)

        # Branch 2: 串行 strip，1x5 -> 5x1（depthwise）
        self.branch_serial_h = nn.Conv2d(c2, c2, (1, 5), padding=(0, 2), groups=c2, bias=False)
        self.branch_serial_v = nn.Conv2d(c2, c2, (5, 1), padding=(2, 0), groups=c2, bias=False)

        # Branch 3: 并行 strip，1x7 + 7x1（depthwise）
        self.branch_parallel_h = nn.Conv2d(c2, c2, (1, 7), padding=(0, 3), groups=c2, bias=False)
        self.branch_parallel_v = nn.Conv2d(c2, c2, (7, 1), padding=(3, 0), groups=c2, bias=False)

        # gate norm
        if norm.lower() == "bn":
            self.gate_norm = nn.BatchNorm2d(c2)
        else:
            self.gate_norm = _make_gn(c2, gn_groups)

        self.gate_act = nn.SiLU()
        self.gate_fusion = nn.Conv2d(c2, c2, 1, bias=False)

        # E4 Trick: zero-init gamma
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))

        # refine：DWConv residual sharpen（zero-init）
        self.refine_conv = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.refine_gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))

    def forward(self, x):
        # 1) 压缩
        x_low = self.compress(x)

        # 2) 多分支
        feat_local = self.branch_local(x_low)
        feat_serial = self.branch_serial_v(self.branch_serial_h(x_low))
        feat_parallel = self.branch_parallel_h(x_low) + self.branch_parallel_v(x_low)

        # 3) 融合
        feat_combined = feat_local + feat_serial + feat_parallel

        # 4) 生成 mask（可增强可抑制）
        mask_raw = self.gate_fusion(self.gate_act(self.gate_norm(feat_combined)))
        mask_low = torch.tanh(mask_raw)

        # 5) 门控（稳定起步）
        out_low = x_low * (1 + self.gamma * mask_low)

        # 6) 上采样
        out = F.interpolate(out_low, scale_factor=self.scale, mode="bilinear", align_corners=False)

        # 7) 残差锐化（稳定起步）
        out = out + self.refine_gamma * self.refine_conv(out)
        return out
