import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f


def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    """Auto padding for 'same' output shape (odd kernels only)."""
    assert kernel_size % 2 == 1, "if use auto pad, kernel size must be odd"
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Round channel number to be divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute(0, 2, 3, 1)


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute(0, 3, 1, 2)


class ConvBNAct(nn.Module):
    """
    Minimal replacement for mmcv.cnn.ConvModule using pure PyTorch:
    Conv2d -> (optional) BatchNorm2d -> (optional) SiLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        norm: bool = True,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit"""

    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class CAA(nn.Module):
    """Context Anchor Attention (mmcv-free)"""

    def __init__(
        self,
        channels: int,
        h_kernel_size: int = 11,
        v_kernel_size: int = 11,
    ):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)

        self.conv1 = ConvBNAct(
            channels, channels, kernel_size=1, stride=1, padding=0, norm=True, act=True
        )

        # depthwise 1 x k
        self.h_conv = ConvBNAct(
            channels,
            channels,
            kernel_size=(1, h_kernel_size),
            stride=1,
            padding=(0, h_kernel_size // 2),
            groups=channels,
            norm=False,
            act=False,
        )

        # depthwise k x 1
        self.v_conv = ConvBNAct(
            channels,
            channels,
            kernel_size=(v_kernel_size, 1),
            stride=1,
            padding=(v_kernel_size // 2, 0),
            groups=channels,
            norm=False,
            act=False,
        )

        self.conv2 = ConvBNAct(
            channels, channels, kernel_size=1, stride=1, padding=0, norm=True, act=True
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.h_conv(attn)
        attn = self.v_conv(attn)
        attn = self.conv2(attn)
        return self.act(attn)


class C2f_CAA(C2f):
    """Ultralytics C2f block with CAA modules replacing the default 'm' blocks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CAA(self.c) for _ in range(n))



if __name__ == "__main__":
    from torchsummary import summary
    import torch
    from thop import profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = C2f_CAA(3, 64, n=1).to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    summary(model, (3, 224, 224))
    #输出模块计算量

    flops, params = profile(model, inputs=(x,), verbose=False)

    print(f"\nParams: {params / 1e6:.4f} M")
    print(f"FLOPs:  {flops / 1e9:.4f} G")
