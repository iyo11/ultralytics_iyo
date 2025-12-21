import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f, C3


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

class CAA_(nn.Module):
    """
    Context Anchor Attention (robust / Ultralytics-friendly)
    - Ignores constructor channel args (because parse_model may pass them inconsistently)
    - Builds internal convs on first forward using actual input channels
    - Keeps H/W unchanged
    - Returns feature map: x * attn (optionally residual)
    """

    def __init__(self, *args, h_kernel_size: int = 11, v_kernel_size: int = 11, **kwargs):
        super().__init__()
        self.h_kernel_size = h_kernel_size
        self.v_kernel_size = v_kernel_size

        # keeps spatial size
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()

        # will be created on first forward
        self._built = False
        self.conv1 = None
        self.h_conv = None
        self.v_conv = None
        self.conv2 = None

    def _build(self, channels: int):
        # ConvBNAct must be available in your project (you already have it)
        self.conv1 = ConvBNAct(
            channels, channels, kernel_size=1, stride=1, padding=0, norm=True, act=True
        )
        self.h_conv = ConvBNAct(
            channels,
            channels,
            kernel_size=(1, self.h_kernel_size),
            stride=1,
            padding=(0, self.h_kernel_size // 2),
            groups=channels,
            norm=False,
            act=False,
        )
        self.v_conv = ConvBNAct(
            channels,
            channels,
            kernel_size=(self.v_kernel_size, 1),
            stride=1,
            padding=(self.v_kernel_size // 2, 0),
            groups=channels,
            norm=False,
            act=False,
        )
        self.conv2 = ConvBNAct(
            channels, channels, kernel_size=1, stride=1, padding=0, norm=True, act=True
        )
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[1])

        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.h_conv(attn)
        attn = self.v_conv(attn)
        attn = self.conv2(attn)
        attn = self.act(attn)

        # safety for odd sizes
        if attn.shape[-2:] != x.shape[-2:]:
            attn = F.interpolate(attn, size=x.shape[-2:], mode="nearest")

        return x * attn
        # 更稳可用残差：
        # return x * attn + x


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck_CAA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = CAA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C3k_CAA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_CAA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_CAA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_CAA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CAA(self.c, self.c, shortcut, g,
                                                                               k=((3, 3), (3, 3)), e=1.0) for _ in
            range(n)
        )


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
