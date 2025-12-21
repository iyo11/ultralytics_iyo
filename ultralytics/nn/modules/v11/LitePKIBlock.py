import math
import torch
import torch.nn as nn
from typing import Optional, Sequence


__all__ = [
    "LitePKIBlock"
]

# -------------------------
# Small utils
# -------------------------
def autopad(k: int, p: Optional[int] = None, d: int = 1):
    # same padding for odd k
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    """Conv2d -> BN -> SiLU (or Identity)"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------
# Attention modules (return ATTENTION MAP, not x*attn)
# -------------------------
class IdentityAttn(nn.Module):
    def forward(self, x):
        return 1.0


class SEAttn(nn.Module):
    """Squeeze-Excitation: returns (B,C,1,1)"""
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        hidden = max(channels // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.pool(x))


class ECAAttn(nn.Module):
    """Efficient Channel Attention: returns (B,C,1,1)"""
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)                   # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)  # (B,1,C)
        y = self.conv1d(y)                 # (B,1,C)
        y = y.transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
        return self.act(y)


class CAALite(nn.Module):
    """
    Simplified CAA-style directional depthwise conv attention.
    Returns (B,C,H,W).
    """
    def __init__(self, channels: int, k: int = 11):
        super().__init__()
        self.pool = nn.AvgPool2d(7, 1, 3)
        self.pre = ConvBNAct(channels, channels, 1, 1, act=True)
        self.h = nn.Conv2d(channels, channels, (1, k), 1, (0, k // 2), groups=channels, bias=False)
        self.v = nn.Conv2d(channels, channels, (k, 1), 1, (k // 2, 0), groups=channels, bias=False)
        self.post = ConvBNAct(channels, channels, 1, 1, act=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.pre(y)
        y = self.h(y)
        y = self.v(y)
        y = self.post(y)
        return self.act(y)


class CoordAttn(nn.Module):
    """
    Coordinate Attention (lightweight): returns (B,C,H,W).
    """
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(mip, channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = self.pool_h(x)                      # (B,C,H,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B,C,W,1)
        y = torch.cat([x_h, x_w], dim=2)          # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)             # (B,mip,1,W)
        a_h = self.sigmoid(self.conv_h(y_h))      # (B,C,H,1)
        a_w = self.sigmoid(self.conv_w(y_w))      # (B,C,1,W)
        return a_h * a_w


def build_attn(attn: str, channels: int, **kwargs) -> nn.Module:
    attn = (attn or "none").lower()
    if attn in ["none", "identity"]:
        return IdentityAttn()
    if attn == "se":
        return SEAttn(channels, r=kwargs.get("se_r", 16))
    if attn == "eca":
        return ECAAttn(channels, k_size=kwargs.get("eca_k", 3))
    if attn in ["caa", "caa_lite"]:
        return CAALite(channels, k=kwargs.get("caa_k", 11))
    if attn in ["coord", "coordatt", "coordinate"]:
        return CoordAttn(channels, reduction=kwargs.get("coord_r", 32))
    raise ValueError(f"Unknown attn type: {attn}")


# -------------------------
# FFN (conv-mlp style)
# -------------------------
class GSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.pool(x))


class ConvFFN(nn.Module):
    """LN (BHWC) + 1x1 + DWConv + gate + 1x1"""
    def __init__(self, channels: int, hidden_mult: float = 4.0, dw_k: int = 3, dropout: float = 0.0):
        super().__init__()
        hidden = int(channels * hidden_mult)
        self.ln = nn.LayerNorm(channels)
        self.pw1 = ConvBNAct(channels, hidden, 1, 1, act=True)
        self.dw = ConvBNAct(hidden, hidden, dw_k, 1, g=hidden, act=False)
        self.gate = GSiLU()
        self.drop = nn.Dropout(dropout)
        self.pw2 = ConvBNAct(hidden, channels, 1, 1, act=True)

    def forward(self, x):
        # LN in BHWC
        y = x.permute(0, 2, 3, 1)
        y = self.ln(y)
        y = y.permute(0, 3, 1, 2)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.gate(y)
        y = self.drop(y)
        y = self.pw2(y)
        y = self.drop(y)
        return y


# -------------------------
# Lite PKIBlock (swappable attention) -- LayerScale modes
# -------------------------
class LitePKIBlock(nn.Module):
    """
    A compact PKI-like block:
      - pre 1x1
      - multi-kernel depthwise (parallel) + sum
      - pw 1x1
      - attention map from "anchor" (pre features)
      - gated residual
      - FFN
      - (optional) LayerScale + DropPath

    LayerScale modes:
      - ls_mode="pre"  (DEFAULT, recommended): gamma1 on hidden, applied BEFORE post (post前)
      - ls_mode="post": gamma1 on c2, applied AFTER post  (post后)
      - ls_mode="dual": both (hidden pre + c2 post), output gamma uses layer_scale_out (default smaller)
    """
    def __init__(
        self,
        c1: int,                                  # input channels (Ultralytics injects this)
        c2: int,                                  # output channels (you provide this in YAML)
        kernels: Sequence[int] = (3, 5, 7, 9, 11),
        expansion: float = 1.0,
        attn: str = "caa",
        ffn_mult: float = 4.0,
        ffn_dw_k: int = 3,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float = 1.0,
        ls_mode: str = "pre",                     # "pre" | "post" | "dual"
        layer_scale_out: Optional[float] = None,  # only for dual: gamma on c2 after post (建议更小)
        **attn_kwargs,
    ):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.hidden = int(c2 * expansion)
        self.ls_mode = (ls_mode or "pre").lower()

        if self.ls_mode not in {"pre", "post", "dual"}:
            raise ValueError(f"LitePKIBlock: ls_mode must be one of ['pre','post','dual'], got {ls_mode}")

        # 1) pre projection: c1 -> hidden
        self.pre = ConvBNAct(c1, self.hidden, 1, 1, act=True)

        # 2) parallel multi-kernel depthwise convs
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(self.hidden, self.hidden, k, 1, autopad(k), groups=self.hidden, bias=False)
            for k in kernels
        ])
        self.dw_bn = nn.BatchNorm2d(self.hidden, eps=1e-3, momentum=0.03)
        self.dw_act = nn.SiLU()

        # 3) pointwise after DW sum
        self.pw = ConvBNAct(self.hidden, self.hidden, 1, 1, act=True)

        # 4) attention map from anchor y0 (hidden channels)
        self.attn = build_attn(attn, self.hidden, **attn_kwargs)

        # 5) project back: hidden -> c2
        self.post = ConvBNAct(self.hidden, c2, 1, 1, act=True)

        # 6) FFN on c2
        self.ffn = ConvFFN(c2, hidden_mult=ffn_mult, dw_k=ffn_dw_k, dropout=dropout)

        # DropPath
        self.dp = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

        # LayerScale setup
        self.use_ls = (layer_scale is not None) and (layer_scale != 0)

        # Defaults: if dual and user didn't set layer_scale_out, make it smaller for stability.
        if layer_scale_out is None:
            layer_scale_out = 0.1 * layer_scale if self.ls_mode == "dual" else layer_scale

        if self.use_ls:
            # gamma for main branch
            if self.ls_mode == "pre":
                # ✅ recommended: scale hidden BEFORE post
                self.gamma1 = nn.Parameter(layer_scale * torch.ones(self.hidden), requires_grad=True)
                self.gamma1_out = None
            elif self.ls_mode == "post":
                # scale output AFTER post
                self.gamma1 = nn.Parameter(layer_scale * torch.ones(c2), requires_grad=True)
                self.gamma1_out = None
            else:  # dual
                self.gamma1 = nn.Parameter(layer_scale * torch.ones(self.hidden), requires_grad=True)
                self.gamma1_out = nn.Parameter(layer_scale_out * torch.ones(c2), requires_grad=True)

            # gamma for ffn branch (always c2)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(c2), requires_grad=True)

    def forward(self, x):
        # Branch 1
        y0 = self.pre(x)  # (B,hidden,H,W)

        y = 0
        for conv in self.dw_convs:
            y = y + conv(y0)
        y = self.dw_act(self.dw_bn(y))
        y = self.pw(y)

        a = self.attn(y0)  # broadcastable to (B,hidden,H,W)
        y = y * a

        # LayerScale gamma1 placement
        if self.use_ls and self.ls_mode in {"pre", "dual"}:
            # gamma1 on hidden (post前)
            y = self.gamma1.view(1, -1, 1, 1) * y

        y = self.post(y)  # (B,c2,H,W)

        if self.use_ls and self.ls_mode == "post":
            # gamma1 on output (post后)
            y = self.gamma1.view(1, -1, 1, 1) * y
        elif self.use_ls and self.ls_mode == "dual":
            # additional output scale (建议更小)
            y = self.gamma1_out.view(1, -1, 1, 1) * y

        x = x + self.dp(y)

        # Branch 2 (FFN)
        f = self.ffn(x)
        if self.use_ls:
            f = self.gamma2.view(1, -1, 1, 1) * f
        x = x + self.dp(f)

        return x


# -------------------------
# Quick sanity test
# -------------------------
if __name__ == "__main__":
    # Example: pre mode (recommended)
    m = LitePKIBlock(128, 128, attn="caa", drop_path=0.1, layer_scale=1.0, ls_mode="pre")
    x = torch.randn(2, 128, 80, 80)
    y = m(x)
    print(y.shape)
