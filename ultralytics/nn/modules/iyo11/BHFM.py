import torch
import torch.nn as nn
import torch.nn.functional as F


class CPlus(nn.Module):
    """
    C+ 融合模块：实现 Concat -> Conv 1x1 -> Add 逻辑
    """

    def __init__(self, c):
        super().__init__()
        self.conv1x1 = nn.Conv2d(c * 2, c, kernel_size=1, bias=False)

    def forward(self, x_orig, x_proc):
        out = torch.cat([x_proc, x_orig], dim=1)
        out = self.conv1x1(out)
        return out + x_orig


class ParameterFreeChannelAttention(nn.Module):
    """
    无参通道注意力机制 (Parameter-free Channel Attention)
    利用全局平均池化和全局最大池化的组合，通过 Sigmoid 激活生成权重
    不需要任何训练参数 (bias, weight)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 使用全局平均池化 (GAP) 捕捉全局背景
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        # 使用全局最大池化 (GMP) 捕捉显著特征 (可选，增加鲁棒性)
        # 先算维度 2 的最大值，再算维度 3 的最大值
        max_out = torch.max(x, dim=2, keepdim=True)[0]
        max_out = torch.max(max_out, dim=3, keepdim=True)[0]

        # 融合两者（或只选其一）并通过 Sigmoid 归一化到 0-1
        # 这里采用均值和最大值的均值，能够更全面地反映通道重要性
        out = 0.5 * (avg_out + max_out)
        return torch.sigmoid(out)


class BHFM(nn.Module):
    """
    Bio-inspired Hierarchical Feature Modulation (BHFM)
    已将通道注意力更换为无参版本
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.project = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        c = c2

        # 左侧：空间特征提取器
        self.ortho_conv5 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, 5), padding=(0, 2), groups=c),
            nn.Conv2d(c, c, kernel_size=(5, 1), padding=(2, 0), groups=c)
        )
        self.ortho_conv19 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, 7), padding=(0, 9), dilation=3, groups=c),
            nn.Conv2d(c, c, kernel_size=(7, 1), padding=(9, 0), dilation=3, groups=c)
        )

        # C+ 融合模块
        self.cplus1 = CPlus(c)
        self.cplus2 = CPlus(c)

        # 右侧：更换为无参通道注意力
        self.channel_attn = ParameterFreeChannelAttention()

        # 底部组件
        self.conv1x1 = nn.Conv2d(c, c, 1, bias=False)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x):
        # 0. 初始输入对齐
        x = self.project(x)

        # 1. 顶部：并行进入 Channel Attention 和 OrthoConv 5x5
        ca_weight = self.channel_attn(x)
        feat5 = self.ortho_conv5(x)

        # 2. 第一个 C+：融合 OrthoConv 5x5 的输出与原始 Input
        feat_cplus1 = self.cplus1(x, feat5)

        # 3. 经过 OrthoConv 19x19
        feat19 = self.ortho_conv19(feat_cplus1)

        # 4. 第二个 C+
        feat_cplus2 = self.cplus2(feat_cplus1, feat19)

        # 5. 乘法节点 (×)
        feat_mul1 = feat_cplus2 * x

        # 6. Conv 1x1
        feat_conv = self.conv1x1(feat_mul1)

        # 7. 加法节点 (+)
        feat_add = feat_conv + x

        # 8. 乘法节点 (×)：使用无参生成的 ca_weight 进行调制
        feat_mod = feat_add * ca_weight

        # 9. Norm 归一化
        feat_norm = self.norm(feat_mod)

        # 10. 底部加法节点 (+)
        return feat_norm + x