import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DropPath", "MSDWConv", "MSConvStar", "LocalMultiRangeAttention", "MAB"]


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
# Local Multi-Range Attention (Pure PyTorch)
#
# 论文思想不变：
#   - 多个 range(不同 kernel size / dilation)
#   - head 绑定到某个 range
#   - range 内做局部邻域 attention
#
# 工程改动（适配 YOLO + 强化小目标）：
#   1) 允许 num_heads 不能被 num_ranges 整除
#   2) 用“按 kernel 大小加权”的方式分配 heads：小 kernel 分更多 head -> 更关注细节/小目标
#   3) 保证每个 range 至少 1 个 head（否则 range 不工作）
# -------------------------
class LocalMultiRangeAttention(nn.Module):
    """
    NCHW in/out.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(5, 7, 9),     # ✅ 小目标友好：偏小范围
        dilations=(1, 1, 1),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
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

        # -------- heads per range (small-object biased) --------
        # 每个 range 至少 1 个 head
        if self.num_heads < self.num_ranges:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be >= num_ranges ({self.num_ranges}) "
                f"to ensure each range gets at least 1 head."
            )

        # 权重：w_i = 1/(k^2)  小 kernel 权重大 -> 分更多 heads
        weights = torch.tensor([1.0 / (k * k) for k in self.kernel_sizes], dtype=torch.float32)
        weights = weights / weights.sum()

        remain = self.num_heads - self.num_ranges  # 先给每个 range 1 个 head，剩余再分
        extra = torch.floor(weights * remain).to(torch.int64)

        # 处理 floor 误差，按权重从大到小补齐
        diff = remain - int(extra.sum().item())
        if diff > 0:
            idx = torch.argsort(weights, descending=True)
            for j in range(diff):
                extra[idx[j % self.num_ranges]] += 1

        self.heads_per_range = (extra + 1).tolist()
        assert sum(self.heads_per_range) == self.num_heads, "heads_per_range allocation error"
        assert all(h > 0 for h in self.heads_per_range), "each range must have >=1 head"

        # -------------------------------------------------------
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def _attend_range(self, q, k, v, ks, dil):
        """
        q,k,v: (B, h, d, H, W) for this range
        returns: (B, h, d, H, W)
        """
        B, h, d, H, W = q.shape
        L = ks * ks

        def unfold(x):
            # (B,h,d,H,W) -> (B,h*d,H,W) -> unfold -> (B,h*d*L,HW) -> (B,h,d,L,HW)
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

        k_nb = unfold(k)  # (B,h,d,L,HW)
        v_nb = unfold(v)  # (B,h,d,L,HW)

        q_flat = q.reshape(B, h, d, H * W) * self.scale  # (B,h,d,HW)

        logits = (q_flat.unsqueeze(3) * k_nb).sum(dim=2)  # (B,h,L,HW)
        attn = logits.softmax(dim=2)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(2) * v_nb).sum(dim=3)  # (B,h,d,HW)
        return out.reshape(B, h, d, H, W)

    def forward(self, x):
        """
        x: (B,C,H,W)
        """
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        # (B, heads, head_dim, H, W)
        q = q.reshape(B, self.num_heads, self.head_dim, H, W)
        k = k.reshape(B, self.num_heads, self.head_dim, H, W)
        v = v.reshape(B, self.num_heads, self.head_dim, H, W)

        outs = []
        hs = 0
        for i in range(self.num_ranges):
            he = hs + self.heads_per_range[i]
            out_i = self._attend_range(
                q[:, hs:he], k[:, hs:he], v[:, hs:he],
                ks=self.kernel_sizes[i],
                dil=self.dilations[i],
            )
            outs.append(out_i)
            hs = he

        out = torch.cat(outs, dim=1)  # (B, heads, head_dim, H, W)
        out = out.reshape(B, C, H, W)
        out = self.proj_drop(self.proj(out))
        return out


# -------------------------
# Paper-faithful MAB (4-stage)
# MA(dilation=1) + FFN + SMA(dilation>1) + FFN
#
# 小目标默认配置：
#   - kernel_sizes=(5,7,9)  比 (7,9,11) 更偏细节
#   - SMA dilations=2       比 4 更不容易“跨过”小目标
#   - num_heads=8           适配 YOLO 常见通道（如 1024）
# -------------------------
class MAB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        kernel_sizes=(5, 7, 9),
        ma_dilations=(1, 1, 1),
        sma_dilations=(2, 2, 2),
        mlp_ratio=2.0,
        dw_sizes=(1, 3, 5, 7),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # 用 GroupNorm(1,C) 做每像素通道归一化（NCHW）
        self.norm1 = nn.GroupNorm(1, dim)
        self.ma = LocalMultiRangeAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilations=ma_dilations,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp1 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

        self.norm3 = nn.GroupNorm(1, dim)
        self.sma = LocalMultiRangeAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            dilations=sma_dilations,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm4 = nn.GroupNorm(1, dim)
        self.mlp2 = MSConvStar(dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

    def forward(self, x):  # x: (B,C,H,W)
        x = x + self.drop_path(self.ma(self.norm1(x)))
        x = x + self.drop_path(self.mlp1(self.norm2(x)))
        x = x + self.drop_path(self.sma(self.norm3(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        return x
