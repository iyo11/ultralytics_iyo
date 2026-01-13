#################################################################################################
##E1
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
##E2
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



#################################################################################################
##E3
#################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class StableDSU(nn.Module):
    """
    StableDSU（版本 B：更通用、更保守）

    设计目标：
    1. 模块初始行为接近恒等映射（zero-init），避免跨数据集性能退化
    2. 在低分辨率特征上生成门控，降低计算量
    3. 门控为“中心化”调制（可增强也可抑制特征）
    """

    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.scale = scale

        # ---------------------------------------------------
        # 1. 通道压缩（通常 c1 > c2）
        #    作用：
        #    - 对齐通道数，便于后续融合
        #    - 降低后续门控和卷积的计算量
        # ---------------------------------------------------
        self.compress = nn.Conv2d(c1, c2, kernel_size=1, bias=False)

        # ---------------------------------------------------
        # 2. 低分辨率门控分支（Lightweight Gate）
        #    特点：
        #    - 深度可分离卷积（groups=c2），计算开销小
        #    - 在低分辨率下工作，进一步省算力
        #    - Tanh 输出在 [-1, 1]，是“中心化门控”
        #      可对特征进行增强或抑制
        # ---------------------------------------------------
        self.gate_conv = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=1, bias=True),
            nn.Tanh()
        )

        # ---------------------------------------------------
        # 3. 上采样后的轻量细化卷积
        #    - 深度卷积，仅做空间细化
        #    - 不改变通道数
        # ---------------------------------------------------
        self.refine = nn.Conv2d(
            c2, c2, kernel_size=3, padding=1, groups=c2, bias=False
        )

        # ---------------------------------------------------
        # 4. 残差强度系数 gamma（核心）
        #    - 通道级可学习参数
        #    - 初始化为 0
        #
        #    关键性质：
        #    gamma = 0 时：
        #      模块整体 ≈ 恒等映射（非常保守）
        #    gamma > 0 时：
        #      模块才逐渐发挥增强/抑制作用
        # ---------------------------------------------------
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))

    def forward(self, x):
        """
        前向传播流程：
        1. 通道压缩
        2. 低分辨率门控生成
        3. 残差式门控调制
        4. 双线性上采样
        5. 轻量细化（同样受 gamma 控制）
        """

        # 1. 通道压缩（仍在低分辨率）
        x_low = self.compress(x)

        # 2. 生成门控权重（范围 [-1, 1]）
        gate = self.gate_conv(x_low)

        # 3. 残差式调制
        #    gamma = 0 时：out_low == x_low
        out_low = x_low * (1.0 + self.gamma * gate)

        # 4. 统一进行一次上采样
        out = F.interpolate(
            out_low,
            scale_factor=self.scale,
            mode='bilinear',
            align_corners=False
        )

        # 5. 上采样后细化（同样用残差，保证稳定）
        out = out + self.gamma * self.refine(out)

        return out
