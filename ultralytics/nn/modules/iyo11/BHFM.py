import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient Channel Attention (very light, learnable)"""
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)  # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(1, 2))  # [B,1,C]
        y = torch.sigmoid(y.transpose(1, 2).unsqueeze(-1))  # [B,C,1,1]
        return y


class CPlusGate(nn.Module):
    """Concat -> 1x1+BN+Act -> gated residual add"""
    def __init__(self, c):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        # start from 0 => near-identity at beginning (stable for backbone->neck)
        self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x_orig, x_proc):
        out = self.fuse(torch.cat([x_proc, x_orig], dim=1))
        return x_orig + self.alpha * out


class OrthoDW_PW(nn.Module):
    """(1,k)+(k,1) depthwise -> PW mix -> BN+Act"""
    def __init__(self, c, k=5, dilation=1):
        super().__init__()
        pad = (k // 2) * dilation
        self.dw = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, pad), dilation=dilation, groups=c, bias=False),
            nn.Conv2d(c, c, kernel_size=(k, 1), padding=(pad, 0), dilation=dilation, groups=c, bias=False),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class BHFMv2(nn.Module):
    """
    Backbone->Neck friendly:
    - ECA channel attention (learnable but tiny)
    - OrthoDW + PW mixing
    - LayerScale gating on multiplicative / additive branches
    - Reduce over-smoothing: dilation=2 instead of 3 by default
    """
    def __init__(self, c1, c2, eca_k=3, k_small=5, k_large=7, dil_large=2):
        super().__init__()
        self.project = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()
        c = c2

        # spatial branches
        self.ortho_small = OrthoDW_PW(c, k=k_small, dilation=1)
        self.ortho_large = OrthoDW_PW(c, k=k_large, dilation=dil_large)

        # fusions
        self.cplus1 = CPlusGate(c)
        self.cplus2 = CPlusGate(c)

        # channel attention
        self.ca = ECA(k_size=eca_k)

        # post
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        self.norm = nn.BatchNorm2d(c)

        # LayerScale gates (start 0 => safe)
        self.gamma_mul = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma_ca  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma_out = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        x = self.project(x)

        # CA weight
        ca_w = self.ca(x)  # [B,C,1,1]

        # spatial small -> cplus
        feat_s = self.ortho_small(x)
        y1 = self.cplus1(x, feat_s)

        # spatial large -> cplus
        feat_l = self.ortho_large(y1)
        y2 = self.cplus2(y1, feat_l)

        # multiplicative node (gated to avoid killing small objects)
        mul = x + self.gamma_mul * (y2 * x)

        # conv + residual
        z = self.conv1x1(mul)
        z = z + x

        # CA modulation (convert to gentle modulation around 1.0)
        # (ca_w in 0..1) -> (-1..1) -> scaled -> (1 + ...)
        ca_delta = (ca_w - 0.5) * 2.0
        z = z * (1.0 + self.gamma_ca * ca_delta)

        z = self.norm(z)

        # final residual (gated)
        return x + self.gamma_out * z
