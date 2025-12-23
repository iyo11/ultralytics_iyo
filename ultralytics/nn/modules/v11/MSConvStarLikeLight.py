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
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden // 2, dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x + self.dw(x)
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
# ✅ 替换版：同等功能的“多尺度局部邻域注意力”
#   - 多 range(不同 ks/dilation)
#   - 多头
#   - unfold 邻域注意力
#   - 可选相对位置 bias
#   - range 内独立 qkv（更直观，也更易扩展）
# ============================================================
class NeighborhoodAttention(nn.Module):
    """
    NCHW in/out.

    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kernel_sizes=(5, 7, 9),
        dilations=(1, 2, 3),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos_bias: bool = False,
        head_alloc: str = "favor_small",  # "favor_small" 或 "uniform"
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must match."
        assert dim % num_heads == 0, "dim must be divisible by num_heads."

        self.dim = dim
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.kernel_sizes = list(kernel_sizes)
        self.dilations = list(dilations)
        self.num_ranges = len(self.kernel_sizes)
        self.use_rel_pos_bias = bool(use_rel_pos_bias)

        if self.num_heads < self.num_ranges:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be >= num_ranges ({self.num_ranges}) "
                f"to ensure each range gets at least 1 head."
            )

        # -------------------------
        # heads_per_range 分配
        # -------------------------
        if head_alloc == "uniform":
            base = self.num_heads // self.num_ranges
            rem = self.num_heads - base * self.num_ranges
            self.heads_per_range = [base] * self.num_ranges
            for i in range(rem):
                self.heads_per_range[i] += 1
        else:
            # 默认：favor_small（保持你原本“更偏向小 kernel”的策略）
            weights = torch.tensor([1.0 / (k * k) for k in self.kernel_sizes], dtype=torch.float32)
            weights = weights / weights.sum()

            remain = self.num_heads - self.num_ranges
            extra = torch.floor(weights * remain).to(torch.int64)

            diff = remain - int(extra.sum().item())
            if diff > 0:
                idx = torch.argsort(weights, descending=True)
                for j in range(diff):
                    extra[idx[j % self.num_ranges]] += 1

            self.heads_per_range = (extra + 1).tolist()

        assert sum(self.heads_per_range) == self.num_heads, "heads_per_range allocation error"
        assert all(h > 0 for h in self.heads_per_range), "each range must have >=1 head"

        # -------------------------
        # 每个 range 独立 qkv
        # -------------------------
        self.qkv_per_range = nn.ModuleList()
        for _ in range(self.num_ranges):
            self.qkv_per_range.append(nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        # rel pos bias: 每个 range 一张 (heads_in_range, ks*ks)
        if self.use_rel_pos_bias:
            self.rel_pos_bias = nn.ParameterList()
            for i, ks in enumerate(self.kernel_sizes):
                h_i = self.heads_per_range[i]
                L = ks * ks
                self.rel_pos_bias.append(nn.Parameter(torch.zeros(h_i, L)))
        else:
            self.rel_pos_bias = None

    def _unfold(self, x, ks, dil):
        # x: (B,h,d,H,W) -> unfold -> (B,h,d,L,HW)
        B, h, d, H, W = x.shape
        x = x.reshape(B, h * d, H, W)
        x = F.unfold(
            x,
            kernel_size=ks,
            dilation=dil,
            padding=(ks // 2) * dil,
            stride=1,
        )  # (B, h*d*L, HW)
        L = ks * ks
        x = x.reshape(B, h, d, L, H * W)
        return x

    def _attend(self, q, k, v, ks, dil, rel_pos_bias=None):
        # q/k/v: (B,h,d,H,W)
        B, h, d, H, W = q.shape
        k_nb = self._unfold(k, ks, dil)  # (B,h,d,L,HW)
        v_nb = self._unfold(v, ks, dil)

        q_flat = q.reshape(B, h, d, H * W) * self.scale  # (B,h,d,HW)
        logits = (q_flat.unsqueeze(3) * k_nb).sum(dim=2)  # (B,h,L,HW)

        if rel_pos_bias is not None:
            logits = logits + rel_pos_bias.unsqueeze(0).unsqueeze(-1)  # (1,h,L,1)

        attn = logits.softmax(dim=2)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(2) * v_nb).sum(dim=3)  # (B,h,d,HW)
        return out.reshape(B, h, d, H, W)

    def forward(self, x):
        B, C, H, W = x.shape

        outs = []
        hs = 0
        for i in range(self.num_ranges):
            h_i = self.heads_per_range[i]
            he = hs + h_i

            q, k, v = self.qkv_per_range[i](x).chunk(3, dim=1)  # (B,C,H,W) each
            q = q.reshape(B, self.num_heads, self.head_dim, H, W)[:, hs:he]
            k = k.reshape(B, self.num_heads, self.head_dim, H, W)[:, hs:he]
            v = v.reshape(B, self.num_heads, self.head_dim, H, W)[:, hs:he]

            bias_i = None
            if self.use_rel_pos_bias:
                bias_i = self.rel_pos_bias[i]  # (h_i, L)

            out_i = self._attend(
                q, k, v,
                ks=self.kernel_sizes[i],
                dil=self.dilations[i],
                rel_pos_bias=bias_i,
            )
            outs.append(out_i)
            hs = he

        out = torch.cat(outs, dim=1)          # (B, num_heads, head_dim, H, W)
        out = out.reshape(B, C, H, W)         # (B,C,H,W)
        out = self.proj_drop(self.proj(out))  # (B,C,H,W)
        return out


# ============================================================
# 统一模块：UMABL
# ============================================================
class UMABL(nn.Module):
    """
    Unified Multi-range Attention Block (UMABL)
    - 只保留一套统一默认超参：适配 P2/P3/P4 都能用
    - 你想全网统一就别传参，三处直接用同一个默认
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,

        # --- MA (统一默认) ---
        kernel_sizes=(3, 5),
        ma_dilations=(1, 2),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rel_pos_bias=False,
        head_alloc="favor_small",

        # --- MLP (统一默认) ---
        mlp_ratio=1.5,
        dw_sizes=(1, 3, 5),

        # --- SMA conv (统一默认) ---
        sma_conv_kernel=3,
        sma_dilations=(1, 2),

        # --- regularization ---
        drop_path=0.0,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = LayerNorm2d(dim)
        self.ma = NeighborhoodAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilations=ma_dilations,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rel_pos_bias=use_rel_pos_bias,
            head_alloc=head_alloc,
        )

        self.norm2 = LayerNorm2d(dim)
        self.mlp1 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

        self.norm3 = LayerNorm2d(dim)
        self.sma = CascadedSparseContextConv(
            dim=dim,
            kernel_size=sma_conv_kernel,
            dilations=sma_dilations,
        )

        self.norm4 = LayerNorm2d(dim)
        self.mlp2 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

    def forward(self, x):
        x = x + self.drop_path(self.ma(self.norm1(x)))
        x = x + self.drop_path(self.mlp1(self.norm2(x)))
        x = x + self.drop_path(self.sma(self.norm3(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        return x
