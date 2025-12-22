import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LMAB", "LLMAB"]

# -------------------------
# DropPath (简易版)
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
    """
    True LayerNorm over channel dimension for NCHW tensors.
    It permutes NCHW -> NHWC, applies layer_norm over C, then permutes back.
    """
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
# Local Multi-Range Attention (Pure PyTorch) + Relative Positional Bias B ✅
# -------------------------
class LocalMultiRangeAttention(nn.Module):
    """
    NCHW in/out.
    在 logits 上显式加入 learnable relative positional bias B（按 range、按 head、按相对位置槽位）。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(5, 7, 9),
        dilations=(1, 2, 3),   # ✅ 检测友好：等效更大视野（虚拟大kernel）
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rel_pos_bias=True,  # ✅ 新增：是否启用相对位置偏置 B
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must match."
        assert dim % num_heads == 0, "dim must be divisible by num_heads."

        self.dim = dim
        self.num_heads = num_heads
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

        # heads 分配（保持你的原逻辑：小kernel更偏向）
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

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        # ✅ Learnable relative positional bias B
        # 每个 range 一张表： (heads_in_range, ks*ks)
        # 其中 ks*ks 对应 unfold 的邻域槽位（相对 offset 集合），在 softmax 前加到 logits。
        if self.use_rel_pos_bias:
            self.rel_pos_bias = nn.ParameterList()
            for i, ks in enumerate(self.kernel_sizes):
                h_i = self.heads_per_range[i]
                L = ks * ks
                self.rel_pos_bias.append(nn.Parameter(torch.zeros(h_i, L)))
        else:
            self.rel_pos_bias = None

    def _attend_range(self, q, k, v, ks, dil, rel_pos_bias=None):
        B, h, d, H, W = q.shape
        L = ks * ks

        def unfold(x):
            x = x.reshape(B, h * d, H, W)
            x = F.unfold(
                x,
                kernel_size=ks,
                dilation=dil,
                padding=(ks // 2) * dil,
                stride=1,
            )
            x = x.reshape(B, h, d, L, H * W)
            return x

        k_nb = unfold(k)
        v_nb = unfold(v)

        q_flat = q.reshape(B, h, d, H * W) * self.scale
        logits = (q_flat.unsqueeze(3) * k_nb).sum(dim=2)  # (B,h,L,HW)

        # ✅ 加 B： (h,L) -> broadcast 到 (B,h,L,HW)
        if rel_pos_bias is not None:
            logits = logits + rel_pos_bias.unsqueeze(0).unsqueeze(-1)

        attn = logits.softmax(dim=2)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(2) * v_nb).sum(dim=3)  # (B,h,d,HW)
        return out.reshape(B, h, d, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, self.head_dim, H, W)
        k = k.reshape(B, self.num_heads, self.head_dim, H, W)
        v = v.reshape(B, self.num_heads, self.head_dim, H, W)

        outs = []
        hs = 0
        for i in range(self.num_ranges):
            he = hs + self.heads_per_range[i]

            bias_i = None
            if self.use_rel_pos_bias:
                bias_i = self.rel_pos_bias[i]  # (heads_in_range, ks*ks)

            out_i = self._attend_range(
                q[:, hs:he], k[:, hs:he], v[:, hs:he],
                ks=self.kernel_sizes[i],
                dil=self.dilations[i],
                rel_pos_bias=bias_i,
            )
            outs.append(out_i)
            hs = he

        out = torch.cat(outs, dim=1)
        out = out.reshape(B, C, H, W)
        out = self.proj_drop(self.proj(out))
        return out


# -------------------------
# Cascaded Sparse Context Conv (DWConv cascade + dilation) ✅
#   - 用串联（cascade）吃多尺度上下文
#   - 避免并联多空洞带来的 aliasing
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


# -------------------------
# LMAB：LN2d + (MA + MLP + SMA(cascade) + MLP)
# -------------------------
class LMAB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(5, 7, 9),
        ma_dilations=(1, 2, 3),        # ✅ 检测友好：等效大视野
        sma_dilations=(1, 2, 3),       # ✅ 多 dilation 用串联吃回来
        mlp_ratio=2.0,
        dw_sizes=(1, 3, 5, 7),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        sma_conv_kernel=5,
        layer_scale_init=None,          # LMAB 默认不启用 LayerScale（保持你原意）
        use_full_layerscale=True,       # 如果启用 LayerScale，是否四支路都门控
        use_rel_pos_bias=True,          # ✅ 新增：是否启用 MA 中的相对位置偏置 B
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = LayerNorm2d(dim)
        self.ma = LocalMultiRangeAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilations=ma_dilations,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rel_pos_bias=use_rel_pos_bias,  # ✅ 传入
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

        # 可选 LayerScale（LMAB 默认 None 表示不使用）
        self.use_layerscale = layer_scale_init is not None
        self.use_full_layerscale = use_full_layerscale

        if self.use_layerscale:
            init = float(layer_scale_init)
            if use_full_layerscale:
                self.gamma_ma   = nn.Parameter(init * torch.ones(1, dim, 1, 1))
                self.gamma_mlp1 = nn.Parameter(init * torch.ones(1, dim, 1, 1))
            else:
                self.register_parameter("gamma_ma", None)
                self.register_parameter("gamma_mlp1", None)

            self.gamma_sma  = nn.Parameter(init * torch.ones(1, dim, 1, 1))
            self.gamma_mlp2 = nn.Parameter(init * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        if not self.use_layerscale:
            x = x + self.drop_path(self.ma(self.norm1(x)))
            x = x + self.drop_path(self.mlp1(self.norm2(x)))
            x = x + self.drop_path(self.sma(self.norm3(x)))
            x = x + self.drop_path(self.mlp2(self.norm4(x)))
            return x

        # 使用 LayerScale
        if self.use_full_layerscale:
            x = x + self.drop_path(self.gamma_ma   * self.ma(self.norm1(x)))
            x = x + self.drop_path(self.gamma_mlp1 * self.mlp1(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ma(self.norm1(x)))
            x = x + self.drop_path(self.mlp1(self.norm2(x)))

        x = x + self.drop_path(self.gamma_sma  * self.sma(self.norm3(x)))
        x = x + self.drop_path(self.gamma_mlp2 * self.mlp2(self.norm4(x)))
        return x


# -------------------------
# LLMAB：LN2d + LayerScale（检测更稳）
# -------------------------
class LLMAB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(5, 7, 9),
        ma_dilations=(1, 2, 3),       # ✅ 检测友好：等效大视野
        sma_dilations=(1, 2, 3),      # ✅ 多 dilation 串联
        mlp_ratio=2.0,
        dw_sizes=(1, 3, 5, 7),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        sma_conv_kernel=5,
        layer_scale_init=1e-3,      # attention block 用 1e-3 通常更稳
        use_full_layerscale=False,   # 默认四支路都门控（推荐）
        use_rel_pos_bias=True,      # ✅ 新增：是否启用 MA 中的相对位置偏置 B
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = LayerNorm2d(dim)
        self.ma = LocalMultiRangeAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilations=ma_dilations,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rel_pos_bias=use_rel_pos_bias,  # ✅ 传入
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

        init = float(layer_scale_init)
        self.use_full_layerscale = use_full_layerscale

        if use_full_layerscale:
            self.gamma_ma   = nn.Parameter(init * torch.ones(1, dim, 1, 1))
            self.gamma_mlp1 = nn.Parameter(init * torch.ones(1, dim, 1, 1))
        else:
            self.register_parameter("gamma_ma", None)
            self.register_parameter("gamma_mlp1", None)

        self.gamma_sma  = nn.Parameter(init * torch.ones(1, dim, 1, 1))
        self.gamma_mlp2 = nn.Parameter(init * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        if self.use_full_layerscale:
            x = x + self.drop_path(self.gamma_ma   * self.ma(self.norm1(x)))
            x = x + self.drop_path(self.gamma_mlp1 * self.mlp1(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ma(self.norm1(x)))
            x = x + self.drop_path(self.mlp1(self.norm2(x)))

        x = x + self.drop_path(self.gamma_sma  * self.sma(self.norm3(x)))
        x = x + self.drop_path(self.gamma_mlp2 * self.mlp2(self.norm4(x)))
        return x
