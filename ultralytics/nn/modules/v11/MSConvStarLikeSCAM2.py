import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# DropPath
# -------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = torch.floor(rnd)
        return x / keep_prob * mask


# -------------------------
# LayerNorm2d for NCHW
# -------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
        return x.permute(0, 3, 1, 2)


# -------------------------
# MSDWConv (NCHW)
# -------------------------
class MSDWConv(nn.Module):
    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.dw_sizes = list(dw_sizes)
        self.channels = []
        self.proj = nn.ModuleList()
        n = len(self.dw_sizes)
        for i, k in enumerate(self.dw_sizes):
            ch = dim - (dim // n) * (n - 1) if i == 0 else (dim // n)
            self.channels.append(ch)
            self.proj.append(nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=True))

    def forward(self, x):
        xs = torch.split(x, self.channels, dim=1)
        ys = [conv(t) for conv, t in zip(self.proj, xs)]
        return torch.cat(ys, dim=1)


# -------------------------
# MSConvStar (NCHW)
# -------------------------
class MSConvStar(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        if hidden % 2 != 0:
            raise ValueError(f"hidden_dim must be even for chunk(2), got {hidden}")
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dw = MSDWConv(hidden, dw_sizes=dw_sizes)
        # 多尺度输出后的融合 + 非线性
        self.fuse = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)
        self.fuse_act = nn.GELU()
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden // 2, dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        # 多尺度DWConv残差
        x = x + self.dw(x)
        # 1x1融合 + GELU（更贴近论文图里的“融合+非线性”位置）
        x = self.fuse_act(self.fuse(x))
        # Star
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        return self.fc2(x)


# -------------------------
# Cascaded Sparse Context Conv (DWConv cascade + dilation)
# -------------------------
class CascadedSparseContextConv(nn.Module):
    def __init__(self, dim, kernel_size=5, dilations=(1, 2, 3)):
        super().__init__()
        if isinstance(dilations, int):
            dilations = (dilations,)
        assert len(dilations) >= 1, "dilations must have at least one element"

        layers = []
        for d in dilations:
            d = int(d)
            padding = (kernel_size // 2) * d
            layers.append(
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=d,
                    groups=dim,
                    bias=True,
                )
            )
            layers.append(nn.GELU())
        self.dw = nn.Sequential(*layers)
        self.pw = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x):
        return self.pw(self.dw(x))


# -------------------------
# Minimal Conv wrappers (for "official" SCAM dependencies)
# -------------------------
class Conv(nn.Module):
    """Conv2d + BN + SiLU (common detection style)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv_withoutBN(nn.Module):
    """Conv2d (bias=True) + optional act; no BN."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=False):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=True)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


# ============================================================
# Official SCAM (your provided "authentic" version)
# ============================================================
class OfficialSCAM(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(OfficialSCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels  # keep as-is

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

        k = self.k(x).view(n, 1, -1, 1).softmax(2)
        v = self.v(x).view(n, 1, c, -1)

        y = torch.matmul(v, k).view(n, c, 1, 1)

        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)
        y_cat = torch.cat((y_avg, y_max), 1)

        y = self.m(y) * self.m2(y_cat).sigmoid()

        return x + y


# ============================================================
# SCAM Adapter (drop-in for MABL_SCAM)
# ============================================================
class SCAM(nn.Module):
    """
    Drop-in replacement for the previous SCAM API:
      - Accepts dim/qk_bias/use_channel/use_spatial but ignores them
      - Executes OfficialSCAM exactly
    """

    def __init__(self, dim, qk_bias=True, use_channel=True, use_spatial=True, reduction=1, **kwargs):
        super().__init__()
        self.core = OfficialSCAM(in_channels=dim, reduction=reduction)

    def forward(self, x):
        return self.core(x)


# ============================================================
# UMABL — Unified block (SCAM + MSConvStar + SMA + MSConvStar)
# ============================================================
class MABL_SCAM2(nn.Module):
    """
    One unified block to use at P2/P3/P4 with the same hyperparams by default.
    """

    def __init__(
        self,
        dim: int,
        # keep for API compatibility; SCAM doesn't use heads
        num_heads: int = 8,
        # SCAM (kept for compatibility; adapter ignores switches)
        qkv_bias: bool = True,
        use_spatial: bool = True,
        use_channel: bool = True,
        # MLP
        mlp_ratio: float = 1.5,
        dw_sizes=(1, 3, 5, 7),
        # SMA
        sma_conv_kernel: int = 3,
        sma_dilations=(1, 2),
        # regularization
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = LayerNorm2d(dim)
        self.ma = SCAM(dim=dim, qk_bias=qkv_bias, use_channel=use_channel, use_spatial=use_spatial)

        self.norm2 = LayerNorm2d(dim)
        self.mlp1 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

        self.norm3 = LayerNorm2d(dim)
        self.sma = CascadedSparseContextConv(dim=dim, kernel_size=sma_conv_kernel, dilations=sma_dilations)

        self.norm4 = LayerNorm2d(dim)
        self.mlp2 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

    def forward(self, x):
        x = x + self.drop_path(self.ma(self.norm1(x)))
        x = x + self.drop_path(self.mlp1(self.norm2(x)))
        x = x + self.drop_path(self.sma(self.norm3(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        return x


# -------------------------
# Quick sanity check (optional)
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 96, 64, 64)
    blk = MABL_SCAM2(dim=96, drop_path=0.1)
    y = blk(x)
    print("input:", x.shape, "output:", y.shape)
