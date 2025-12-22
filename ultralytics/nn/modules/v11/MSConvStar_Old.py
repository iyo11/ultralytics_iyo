import math
import torch
import torch.nn as nn

__all__ = ['MSConvStar', 'MAB']

# --------- 小工具：2D LayerNorm（对通道做LN）---------
class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for NCHW.
    等价于对每个像素位置的C做LN。
    """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: (B,C,H,W)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


# --------- 你的多尺度DWConv（复用你给的逻辑）---------
class MSDWConv(nn.Module):
    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.dw_sizes = list(dw_sizes)
        self.channels = []
        self.proj = nn.ModuleList()
        n = len(self.dw_sizes)

        for i, k in enumerate(self.dw_sizes):
            if i == 0:
                ch = dim - dim // n * (n - 1)
            else:
                ch = dim // n
            self.channels.append(ch)
            self.proj.append(
                nn.Conv2d(ch, ch, kernel_size=k, padding=k // 2, groups=ch, bias=True)
            )

    def forward(self, x):
        xs = torch.split(x, split_size_or_sections=self.channels, dim=1)
        ys = [conv(feat) for conv, feat in zip(self.proj, xs)]
        return torch.cat(ys, dim=1)


# --------- Multi-Range Attention（MA）---------
class MultiRangeAttention(nn.Module):
    """
    输入: (B,C,H,W)
    1) 1x1 conv -> q,k,v
    2) 对 q,k 注入 multi-range 信息：q += MSDWConv(q), k += MSDWConv(k)
    3) 标准 multi-head attention 在 token(HW) 上做
    """
    def __init__(self, dim, num_heads=8, dw_sizes=(1, 3, 5, 7), qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

        # multi-range injection
        self.q_dw = MSDWConv(dim, dw_sizes=dw_sizes)
        self.k_dw = MSDWConv(dim, dw_sizes=dw_sizes)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: (B,C,H,W)
        return: (B,C,H,W)
        """
        B, C, H, W = x.shape
        qkv = self.qkv(x)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # multi-range features
        q = q + self.q_dw(q)
        k = k + self.k_dw(k)

        # reshape -> (B, heads, HW, head_dim)
        HW = H * W
        q = q.reshape(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # (B, h, HW, d)
        k = k.reshape(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # (B, h, HW, d)
        v = v.reshape(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # (B, h, HW, d)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B,h,HW,HW)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,h,HW,d)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)  # (B,C,H,W)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# --------- 你的 MSConvStar（你已有的话可直接引用，不用重复定义）---------
class MSConvStar(nn.Module):
    def __init__(self, dim, out_dim, mlp_ratio=2., dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = MSDWConv(dim=hidden_dim, dw_sizes=dw_sizes)
        # 这里保持你原逻辑：chunk(2) 后通道减半，所以 fc2 输入是 hidden_dim//2
        self.fc2 = nn.Conv2d(hidden_dim // 2, out_dim, 1)
        self.act = nn.GELU()
        assert (hidden_dim // len(dw_sizes)) % 2 == 0, "hidden_dim/num_head must be even for chunk(2)"

    def forward(self, x):
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        return x


# --------- MAB：LayerNorm -> MA -> res -> LayerNorm -> MSConvStar -> res ---------
class MAB(nn.Module):
    """
    Multi-Range Attention Block (MAB)
    输入/输出: (B,C,H,W)

    结构:
      x = x + MA(LN(x))
      x = x + FFN(LN(x))   (FFN 用 MSConvStar)
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=2.0,
        dw_sizes_attn=(1, 3, 5, 7),
        dw_sizes_mlp=(1, 3, 5, 7),
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,   # 先占位：如果你项目里有DropPath可替换
        eps=1e-6,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, eps=eps)
        self.attn = MultiRangeAttention(
            dim=dim,
            num_heads=num_heads,
            dw_sizes=dw_sizes_attn,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = LayerNorm2d(dim, eps=eps)
        self.mlp = MSConvStar(dim=dim, out_dim=dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes_mlp)

        # 简化版 DropPath（不依赖timm）。你也可以直接删掉这段，用恒等映射。
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic Depth / DropPath (简易版)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # (B,1,1,1) 广播到所有通道空间位置
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor
