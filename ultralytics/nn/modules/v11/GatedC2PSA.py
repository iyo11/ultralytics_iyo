import torch
import torch.nn as nn

__all__ = ['GatedC2PSA']

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import PSABlock


class GatedC2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # PSA 序列
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

        # 门控分支的额外处理（可选，用于增强门控信号的表达）
        self.gate_conv = nn.Sequential(
            Conv(self.c, self.c, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Split 输入
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # 2. b 分支进行 PSA 处理
        b_processed = self.m(b)

        # 3. 门控操作：利用 a 分支产生门控权重，调制 b 分支
        # 这里的逻辑是：a 决定了 b 中哪些信息是重要的
        g = self.gate_conv(a)
        b_gated = b_processed * g  # 逐元素相乘

        # 4. 融合返回
        return self.cv2(torch.cat((a, b_gated), 1))