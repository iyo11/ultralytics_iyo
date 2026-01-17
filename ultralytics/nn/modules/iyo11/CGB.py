import torch
import torch.nn as nn

__all__ = ['ContextGuidedBlock_Down']


class FGlo(nn.Module):
    """
    全局上下文模块 (Global Context Refinement)
    利用原生 Conv2d(1x1) 代替 Linear，省去 view/reshape 操作
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ContextGuidedBlock_Down(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        super().__init__()

        # 1. 下采样 + 通道扩张 (对应原 ConvBNPReLU)
        # 直接使用 nn.Sequential 组合
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nOut, eps=1e-3),
            nn.PReLU(nOut)
        )
        # 2. 局部特征提取 (对应原 ChannelWiseConv)
        # 使用原生 Conv2d, 设置 groups=nOut 实现深度可分离卷积
        self.F_loc = nn.Conv2d(nOut, nOut, kernel_size=3, padding=1,
                               groups=nOut, bias=False)
        # 3. 周围上下文提取 (对应原 ChannelWiseDilatedConv)
        # 原生 Conv2d, 同时设置 groups 和 dilation
        self.F_sur = nn.Conv2d(nOut, nOut, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate, groups=nOut, bias=False)
        # 4. 融合后的归一化与激活
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        # 5. 通道降维 (对应原 Conv)
        self.reduce = nn.Conv2d(2 * nOut, nOut, kernel_size=1, bias=False)
        # 6. 全局上下文优化
        self.F_glo = FGlo(nOut, reduction)
    def forward(self, input):
        # 下采样
        output = self.conv1x1(input)
        # 分支提取
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        # 拼接与融合
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.reduce(self.act(self.bn(joi_feat)))
        # 全局加权
        output = self.F_glo(joi_feat)
        return output