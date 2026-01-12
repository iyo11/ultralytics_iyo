import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """BHFM 右分支：通道注意力机制 [cite: 62, 439]"""

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
        return x * y


class BHFM(nn.Module):
    """
    Bio-inspired Hierarchical Feature Modulation (BHFM) [cite: 59, 361]
    特点：
    1. 阶梯式正交卷积将感受野从 5x5 扩展至 19x19 [cite: 10, 116, 434]。
    2. 利用逐元素点乘实现全局语义与局部细节的深度交互 [cite: 63, 69, 70]。
    """

    def __init__(self, c1, c2):  # c1, c2 用于兼容 YOLO 接口
        super().__init__()
        self.conv_1x1_pre = nn.Conv2d(c1, c1, 1)

        # 5x5 正交卷积 (1x5 + 5x1)
        self.ortho_conv5 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=(1, 5), padding=(0, 2), groups=c1),
            nn.Conv2d(c1, c1, kernel_size=(5, 1), padding=(2, 0), groups=c1)
        )

        # 19x19 正交空洞卷积 (k=7, d=3 -> 感受野 19) [cite: 10, 67, 434]
        self.ortho_conv19 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=(1, 7), padding=(0, 9), dilation=3, groups=c1),
            nn.Conv2d(c1, c1, kernel_size=(7, 1), padding=(9, 0), dilation=3, groups=c1)
        )

        self.channel_attn = ChannelAttention(c1)
        self.conv_1x1_post = nn.Conv2d(c1, c1, 1)
        self.norm = nn.BatchNorm2d(c1)

    def forward(self, x):
        # 左分支：捕捉局部高频细节 [cite: 61, 433]
        feat5 = self.ortho_conv5(x) + x
        feat19 = self.ortho_conv19(feat5) + feat5
        local_feat = feat5 * feat19  # 空间自适应选择 [cite: 69, 436]

        # 右分支：关注全局语义信息 [cite: 62, 438]
        global_feat = self.channel_attn(x)

        # 深度特征交互调制 [cite: 63, 70]
        out = self.conv_1x1_post(local_feat) * global_feat
        return self.norm(out) + x