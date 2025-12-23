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

        # 用 float32 采样更稳（AMP/FP16 下更少奇怪抖动）
        rnd = keep_prob + torch.rand(shape, dtype=torch.float32, device=x.device)
        mask = torch.floor(rnd).to(dtype=x.dtype)
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
        # NCHW -> NHWC 做 LN
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
        n = len(self.dw_sizes)

        # ✅ 防 dim 太小导致某分支 ch=0
        if dim < n:
            raise ValueError(f"MSDWConv: dim ({dim}) must be >= number of branches ({n}).")

        self.channels = []
        self.proj = nn.ModuleList()

        for i, k in enumerate(self.dw_sizes):
            ch = dim - (dim // n) * (n - 1) if i == 0 else (dim // n)
            if ch <= 0:
                raise ValueError(f"MSDWConv: got non-positive channel split ch={ch}. dim={dim}, n={n}")
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
# Local Multi-Range Attention (NCHW)
# -------------------------
class LocalMultiRangeAttention(nn.Module):
    """
    NCHW in/out.
    logits 上可加 learnable relative positional bias B（按 range、按 head、按相对位置槽位）。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(5, 7, 9),
        dilations=(1, 2, 3),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rel_pos_bias=True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must match."
        assert dim % num_heads == 0, "dim must be divisible by num_heads."

        # ✅ odd kernel 防止中心对齐问题
        for ks in kernel_sizes:
            if int(ks) % 2 != 1:
                raise ValueError(f"kernel_size must be odd, got {ks}")

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

        # heads 分配：小 kernel 权重大
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

        if rel_pos_bias is not None:
            logits = logits + rel_pos_bias.unsqueeze(0).unsqueeze(-1)

        # ✅ softmax 稳定 trick
        #logits = logits - logits.amax(dim=2, keepdim=True)

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
            bias_i = self.rel_pos_bias[i] if self.use_rel_pos_bias else None

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
# Cascaded Sparse Context Conv (DWConv cascade + dilation)
# -------------------------
class CascadedSparseContextConv(nn.Module):
    def __init__(self, dim, kernel_size=5, dilations=(1, 2, 3)):
        super().__init__()
        if isinstance(dilations, int):
            dilations = (dilations,)
        assert len(dilations) >= 1, "dilations must have at least one element"

        if int(kernel_size) % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

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
# Branch-wise DropPath helper
# -------------------------
def _parse_branch_drop_path(drop_path, dp_max=0.5):
    """
    drop_path: float or (dp_ma, dp_mlp1, dp_sma, dp_mlp2)
    returns: (dp_ma, dp_mlp1, dp_sma, dp_mlp2)
    """
    if isinstance(drop_path, (list, tuple)):
        if len(drop_path) != 4:
            raise ValueError("drop_path as list/tuple must be length 4: (ma, mlp1, sma, mlp2)")
        dp_ma, dp_mlp1, dp_sma, dp_mlp2 = [float(x) for x in drop_path]
    else:
        # float 也允许：认为你给的是“整块强度”，然后内部按 YOLO 默认比例拆分
        dp = float(drop_path)
        # YOLO 默认拆分：ma 最大、sma 次之、mlp 最小
        # (ma, mlp1, sma, mlp2)
        w = torch.tensor([1.2, 0.8, 1.0, 0.8], dtype=torch.float32)
        w = w / w.sum()
        dp_ma, dp_mlp1, dp_sma, dp_mlp2 = (dp * w).tolist()

    if dp_max is not None:
        dp_ma = min(dp_ma, dp_max)
        dp_mlp1 = min(dp_mlp1, dp_max)
        dp_sma = min(dp_sma, dp_max)
        dp_mlp2 = min(dp_mlp2, dp_max)

    for v in (dp_ma, dp_mlp1, dp_sma, dp_mlp2):
        if not (0.0 <= v < 1.0):
            raise ValueError(f"each drop-path must be in [0,1), got {v}")

    return dp_ma, dp_mlp1, dp_sma, dp_mlp2


# -------------------------
# LMAB / LMMAB / LTMAB (branch-wise droppath + YOLO-friendly defaults)
# -------------------------
class LMAB(nn.Module):
    """
    大目标/更深层（推荐 P4/P5, stride 16/32）
    ✅ 默认：适度 DropPath（YOLO 友好）
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(3, 5, 7),
        ma_dilations=(1, 2, 2),
        sma_dilations=(1, 2, 2),
        mlp_ratio=2.0,
        dw_sizes=(1, 3, 5, 7),
        qkv_bias=True,
        drop_path=(0.00, 0.00, 0.00, 0.00),
        attn_drop=0.0,
        proj_drop=0.0,
        sma_conv_kernel=3,
        use_rel_pos_bias=False,
        dp_max=0.5,
    ):
        super().__init__()

        dp_ma, dp_mlp1, dp_sma, dp_mlp2 = _parse_branch_drop_path(drop_path, dp_max=dp_max)
        self.dp_ma = DropPath(dp_ma) if dp_ma > 0 else nn.Identity()
        self.dp_mlp1 = DropPath(dp_mlp1) if dp_mlp1 > 0 else nn.Identity()
        self.dp_sma = DropPath(dp_sma) if dp_sma > 0 else nn.Identity()
        self.dp_mlp2 = DropPath(dp_mlp2) if dp_mlp2 > 0 else nn.Identity()

        self.norm1 = LayerNorm2d(dim)
        self.ma = LocalMultiRangeAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilations=ma_dilations,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rel_pos_bias=use_rel_pos_bias,
        )

        self.norm2 = LayerNorm2d(dim)
        self.mlp1 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

        self.norm3 = LayerNorm2d(dim)
        self.sma = CascadedSparseContextConv(dim=dim, kernel_size=sma_conv_kernel, dilations=sma_dilations)

        self.norm4 = LayerNorm2d(dim)
        self.mlp2 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

    def forward(self, x):
        x = x + self.dp_ma(self.ma(self.norm1(x)))
        x = x + self.dp_mlp1(self.mlp1(self.norm2(x)))
        x = x + self.dp_sma(self.sma(self.norm3(x)))
        x = x + self.dp_mlp2(self.mlp2(self.norm4(x)))
        return x


class LMMAB(nn.Module):
    """
    中等尺度（推荐 P3/P4, stride 8/16）
    ✅ 默认：温和 DropPath
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(3, 5),
        ma_dilations=(1, 2),
        sma_dilations=(1, 2),
        mlp_ratio=1.5,
        dw_sizes=(1, 3, 5),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=(0.00, 0.00, 0.00, 0.00),
        sma_conv_kernel=3,
        use_rel_pos_bias=False,
        dp_max=0.5,
    ):
        super().__init__()

        dp_ma, dp_mlp1, dp_sma, dp_mlp2 = _parse_branch_drop_path(drop_path, dp_max=dp_max)
        self.dp_ma = DropPath(dp_ma) if dp_ma > 0 else nn.Identity()
        self.dp_mlp1 = DropPath(dp_mlp1) if dp_mlp1 > 0 else nn.Identity()
        self.dp_sma = DropPath(dp_sma) if dp_sma > 0 else nn.Identity()
        self.dp_mlp2 = DropPath(dp_mlp2) if dp_mlp2 > 0 else nn.Identity()

        self.norm1 = LayerNorm2d(dim)
        self.ma = LocalMultiRangeAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilations=ma_dilations,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rel_pos_bias=use_rel_pos_bias,
        )

        self.norm2 = LayerNorm2d(dim)
        self.mlp1 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

        self.norm3 = LayerNorm2d(dim)
        self.sma = CascadedSparseContextConv(dim=dim, kernel_size=sma_conv_kernel, dilations=sma_dilations)

        self.norm4 = LayerNorm2d(dim)
        self.mlp2 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

    def forward(self, x):
        x = x + self.dp_ma(self.ma(self.norm1(x)))
        x = x + self.dp_mlp1(self.mlp1(self.norm2(x)))
        x = x + self.dp_sma(self.sma(self.norm3(x)))
        x = x + self.dp_mlp2(self.mlp2(self.norm4(x)))
        return x


class LTMAB(nn.Module):
    """
    小目标/浅层（推荐 P2 优先, stride 4/8）
    ✅ 默认：不 Drop（保 recall）
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(3, 5),
        ma_dilations=(1, 1),
        sma_dilations=(1, 1),
        mlp_ratio=1.25,
        dw_sizes=(1, 3),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=(0.00, 0.00, 0.00, 0.00),
        sma_conv_kernel=3,
        use_rel_pos_bias=False,
        dp_max=0.5,
    ):
        super().__init__()

        dp_ma, dp_mlp1, dp_sma, dp_mlp2 = _parse_branch_drop_path(drop_path, dp_max=dp_max)
        self.dp_ma = DropPath(dp_ma) if dp_ma > 0 else nn.Identity()
        self.dp_mlp1 = DropPath(dp_mlp1) if dp_mlp1 > 0 else nn.Identity()
        self.dp_sma = DropPath(dp_sma) if dp_sma > 0 else nn.Identity()
        self.dp_mlp2 = DropPath(dp_mlp2) if dp_mlp2 > 0 else nn.Identity()

        self.norm1 = LayerNorm2d(dim)
        self.ma = LocalMultiRangeAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilations=ma_dilations,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rel_pos_bias=use_rel_pos_bias,
        )

        self.norm2 = LayerNorm2d(dim)
        self.mlp1 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

        self.norm3 = LayerNorm2d(dim)
        self.sma = CascadedSparseContextConv(dim=dim, kernel_size=sma_conv_kernel, dilations=sma_dilations)

        self.norm4 = LayerNorm2d(dim)
        self.mlp2 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

    def forward(self, x):
        x = x + self.dp_ma(self.ma(self.norm1(x)))
        x = x + self.dp_mlp1(self.mlp1(self.norm2(x)))
        x = x + self.dp_sma(self.sma(self.norm3(x)))
        x = x + self.dp_mlp2(self.mlp2(self.norm4(x)))
        return x
