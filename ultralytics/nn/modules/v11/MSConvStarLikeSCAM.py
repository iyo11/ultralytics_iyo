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
        # 新增：多尺度输出后的融合 + 非线性
        self.fuse = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)
        self.fuse_act = nn.GELU()
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden // 2, dim, 1)
    def forward(self, x):
        x = self.fc1(x)
        # 多尺度DWConv残差
        x = x + self.dw(x)
        # 新增：1x1融合 + GELU（更贴近论文图里的“融合+非线性”位置）
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
                    dim, dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=d,
                    groups=dim,
                    bias=True
                )
            )
            layers.append(nn.GELU())
        self.dw = nn.Sequential(*layers)
        self.pw = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x):
        return self.pw(self.dw(x))


# ============================================================
# SCAM (NCHW) — Spatial-Channel Attention Module
#   - Input/Output: (B,C,H,W)
#   - Designed to be a drop-in replacement for your previous SA/MA
# ============================================================
class SCAM(nn.Module):
    """
    A practical SCAM implementation aligned with the figure's intent:
      - Spatial branch uses GMP/GAP to produce a spatial gate (B,1,H,W)
      - Channel branch uses QK/Value (1x1 conv) to produce a channel gate (B,C,1,1)
      - Gates are applied multiplicatively, then lightly projected
    """
    def __init__(self, dim, qk_bias=True, use_channel=True, use_spatial=True):
        super().__init__()
        self.dim = dim
        self.use_channel = use_channel
        self.use_spatial = use_spatial

        # Spatial branch
        if self.use_spatial:
            self.spatial_gate = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            self.spatial_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # Channel branch
        if self.use_channel:
            self.qk = nn.Conv2d(dim, dim, kernel_size=1, bias=qk_bias)
            self.val = nn.Conv2d(dim, dim, kernel_size=1, bias=qk_bias)
            self.channel_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # Final fuse
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        out = x

        # ---- Spatial attention: (B,1,H,W) ----
        if self.use_spatial:
            # channel-wise pooling
            a = x.mean(dim=1, keepdim=True)       # GAP -> (B,1,H,W)
            m = x.amax(dim=1, keepdim=True)       # GMP -> (B,1,H,W)

            # route softmax over {max, avg}
            s = torch.cat([m, a], dim=1)          # (B,2,H,W)
            s = F.softmax(s, dim=1)
            s = s[:, 0:1] * m + s[:, 1:2] * a     # (B,1,H,W)

            s = self.spatial_gate(s)              # (B,1,H,W)
            s = torch.sigmoid(s)                  # gate stability
            out = out * s                         # broadcast Hadamard
            out = self.spatial_proj(out)

        # ---- Channel attention: (B,C,1,1) ----
        if self.use_channel:
            qk = self.qk(x)                       # (B,C,H,W)
            v = self.val(x)                       # (B,C,H,W)

            qk_g = qk.mean(dim=(2, 3))            # (B,C)
            v_g = v.mean(dim=(2, 3))              # (B,C)

            w = F.softmax(qk_g, dim=1)            # (B,C)
            c_desc = (w * v_g).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)

            c_gate = torch.sigmoid(c_desc)
            out = out * c_gate
            out = self.channel_proj(out)

        return self.fuse(out)


# ============================================================
# UMABL — Unified block (one config everywhere)
#   - Uses SCAM as the attention module
#   - Keeps your SMA + MSConvStar MLPs
# ============================================================
class MABL_SCAM(nn.Module):
    """
    One unified block to use at P2/P3/P4 with the same hyperparams by default.
    """
    def __init__(
        self,
        dim: int,
        # keep for API compatibility; SCAM doesn't use heads
        num_heads: int = 8,

        # SCAM
        qkv_bias: bool = True,
        use_spatial: bool = True,
        use_channel: bool = True,

        # MLP (unified middle values)
        mlp_ratio: float = 1.5,
        dw_sizes=(1, 3, 5, 7),

        # SMA (unified middle values)
        sma_conv_kernel: int = 3,
        sma_dilations=(1, 2),  # (1,2,3) ×  (1,2,2)

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
    blk = MABL_SCAM(dim=96, drop_path=0.1)
    y = blk(x)
    print("input:", x.shape, "output:", y.shape)
