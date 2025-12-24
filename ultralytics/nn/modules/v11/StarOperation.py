import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, Conv


# 你工程里已有
# from ultralytics.nn.modules.block import C2f, Bottleneck, C3k
# from ultralytics.nn.modules.conv import Conv

__all__ = ['StarOperation', 'StarBottleneck', 'StarC3k2', 'StarC2f']

class StarOperation(nn.Module):
    """Conv1 -> Conv2; x * gate; Conv3"""
    def __init__(self, c, e=0.5, Conv=None):
        super().__init__()
        assert Conv is not None, "Conv 不能为空"
        hidden = int(c * e)
        self.cv1 = Conv(c, hidden, 1, 1)
        self.cv2 = Conv(hidden, c, 1, 1, act=False)
        self.cv3 = Conv(c, c, 1, 1)

    def forward(self, x):
        g = self.cv2(self.cv1(x))
        x = x * g
        return self.cv3(x)


class StarBottleneck(nn.Module):
    """替代 Bottleneck，用在 C3k2 的 m 分支中"""
    def __init__(self, c, shortcut=True, e=0.5, Conv=None):
        super().__init__()
        self.add = shortcut
        self.star = StarOperation(c, e=e, Conv=Conv)

    def forward(self, x):
        y = self.star(x)
        return x + y if self.add else y


class StarC3k2(C2f):
    """
    完全对齐你给出的 C3k2 实现：
      - 继承 C2f
      - 参数签名一模一样
      - 仅替换 self.m 中的 block
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,   # 保留接口，占位但不再使用
        e: float = 0.5,
        g: int = 1,          # 占位，Star 不用 group conv
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        # self.c 是 C2f 中定义的分支通道
        self.m = nn.ModuleList(
            StarBottleneck(self.c, shortcut=shortcut, e=1.0, Conv=Conv)
            for _ in range(n)
        )

class StarC2f(C2f):
    """
    继承 C2f，只替换 self.m 中的 block 为 StarBottleneck
    签名与原 C2f 对齐： (c1, c2, n=1, shortcut=False, g=1, e=0.5)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        # self.c 是 C2f 内部的分支通道数（C2f 已经算好）
        self.m = nn.ModuleList(
            StarBottleneck(self.c, shortcut=shortcut, e=1.0, Conv=Conv)
            for _ in range(n)
        )

