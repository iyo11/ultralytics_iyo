import torch
import torch.nn as nn
import timm

__all__ = (
    "LCNet075",
)


class LCNetBlock(nn.Module):
    """
    基础 DP-Conv Block（论文 LCBackbone 基础结构）
    使用: DWConv(5x5) + PWConv + SE + H-Swish
    """
    def __init__(self, c1, c2, stride=1, act=True):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(c1, c1, 5, stride, 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            h_swish()
        )
        self.pw = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.se = SE(c2)
        self.act = h_swish() if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.se(x)
        return self.act(x)


class LCNet075(nn.Module):
    """
    完整 LCBackbone (论文 Fig.3 + PP-LCNet_0.75)
    以你上传的 YAML 结构为准进行输出节点划分。
    """

    # PP-LCNet 0.75× 官方通道
    cfg = {
        "stage1": (16, 24, 1),   # out:24
        "stage2": (24, 32, 3),   # out:32
        "stage3": (32, 48, 4),   # out:48
        "stage4": (48, 96, 4),   # out:96
        "stage5": (96, 128, 4),  # out:128
    }

    def __init__(self):
        super().__init__()

        # stem
        self.stem = nn.Sequential(
            Conv(3, 16, k=3, s=2),  # stride 2
            h_swish()
        )

        # stage1
        c1, c2, n = self.cfg["stage1"]
        layers = []
        for i in range(n):
            layers.append(LCNetBlock(c1 if i == 0 else c2, c2, stride=1))
        self.s1 = nn.Sequential(*layers)

        # stage2 (stride=2)
        c1, c2, n = self.cfg["stage2"]
        layers = [LCNetBlock(24, c2, stride=2)]
        for i in range(n-1):
            layers.append(LCNetBlock(c2, c2, stride=1))
        self.s2 = nn.Sequential(*layers)

        # stage3
        c1, c2, n = self.cfg["stage3"]
        layers = [LCNetBlock(32, c2, stride=2)]
        for i in range(n-1):
            layers.append(LCNetBlock(c2, c2, stride=1))
        self.s3 = nn.Sequential(*layers)

        # stage4
        c1, c2, n = self.cfg["stage4"]
        layers = [LCNetBlock(48, c2, stride=2)]
        for i in range(n-1):
            layers.append(LCNetBlock(c2, c2, stride=1))
        self.s4 = nn.Sequential(*layers)

        # stage5
        c1, c2, n = self.cfg["stage5"]
        layers = [LCNetBlock(96, c2, stride=2)]
        for i in range(n-1):
            layers.append(LCNetBlock(c2, c2, stride=1))
        self.s5 = nn.Sequential(*layers)

        # 最终 1280 conv (论文 Fig.3)
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            h_swish()
        )

    def forward(self, x):
        x = self.stem(x)     # 16ch
        x = self.s1(x)       # 24ch     stride 2
        p2 = self.s2(x)      # 32ch     stride 4
        p3 = self.s3(p2)     # 48ch     stride 8
        p4 = self.s4(p3)     # 96ch     stride 16
        x  = self.s5(p4)     # 128ch    stride 32
        x  = self.final_conv(x) # 1280ch

        # 根据你 yaml 的 backbone 输出，我保持 P2/P3/P4 输出：
        return [p2, p3, p4]



if __name__ == '__main__':
    m = LCNet075()
    print(m.out_channels)
    print([x.shape for x in m(torch.randn(1, 3, 640, 640))])