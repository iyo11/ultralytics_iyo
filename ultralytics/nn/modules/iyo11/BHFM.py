import torch
import torch.nn as nn

class CPlus(nn.Module):
    """
    C+ 融合模块：实现 Concat -> Conv 1x1 -> Add 逻辑
    """
    def __init__(self, c):
        super().__init__()
        self.conv1x1 = nn.Conv2d(c * 2, c, kernel_size=1, bias=False)

    def forward(self, x_orig, x_proc):
        # x_orig: 原始特征A, x_proc: 处理后特征B
        # 1. Concat
        out = torch.cat([x_proc, x_orig], dim=1)
        # 2. Conv 1x1
        out = self.conv1x1(out)
        # 3. Residual Add (与原始特征A相加)
        return out + x_orig

class ChannelAttention(nn.Module):
    """BHFM 右分支：通道注意力机制"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y  # 注意：图中右分支输出后是与主路径相乘，这里只返回权重


class BHFM(nn.Module):
    """
    Bio-inspired Hierarchical Feature Modulation (BHFM) 最终版
    严格对照 image_031c8.png 的复杂残差与 C+ 逻辑实现
    """

    def __init__(self, c1, c2):
        super().__init__()
        # 统一通道数
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

        # 右侧：通道注意力
        self.channel_attn = ChannelAttention(c)

        # 底部组件
        self.conv1x1 = nn.Conv2d(c, c, 1, bias=False)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x):
        # 0. 初始输入对齐 (Input)
        x = self.project(x)

        # --- 流程开始 ---

        # 1. 顶部：并行进入 Channel Attention 和 OrthoConv 5x5
        ca_weight = self.channel_attn(x)
        feat5 = self.ortho_conv5(x)

        # 2. 第一个 C+：融合 OrthoConv 5x5 的输出与原始 Input
        # 根据图示：B 是 feat5，A 是 Input
        feat_cplus1 = self.cplus1(x, feat5)

        # 3. 经过 OrthoConv 19x19
        feat19 = self.ortho_conv19(feat_cplus1)

        # 4. 第二个 C+：融合 OrthoConv 19x19 的输出与上一个 C+ 的输出
        # 根据图示：B 是 feat19，A 是 feat_cplus1
        feat_cplus2 = self.cplus2(feat_cplus1, feat19)

        # 5. 乘法节点 (×)：feat_cplus2 与 原始 Input 相乘
        feat_mul1 = feat_cplus2 * x

        # 6. Conv 1x1
        feat_conv = self.conv1x1(feat_mul1)

        # 7. 加法节点 (+)：feat_conv 与 原始 Input 相加
        feat_add = feat_conv + x

        # 8. 乘法节点 (×)：与右侧 Channel Attention 输出调制
        feat_mod = feat_add * ca_weight

        # 9. Norm 归一化
        feat_norm = self.norm(feat_mod)

        # 10. 底部加法节点 (+)：最终残差，与原始 Input 相加
        return feat_norm + x