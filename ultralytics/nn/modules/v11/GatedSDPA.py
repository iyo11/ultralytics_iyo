import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules import Conv, C2f

__all__ = ['LiteGatedAttention_SDPA', 'C3k2_LiteGatedAttention_SDPA']

class LiteGatedAttention_SDPA(nn.Module):
    def __init__(self, d_model, n_heads=8, headwise=True, elementwise=True, reduction_ratio=2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.reduction_ratio = reduction_ratio

        # Q, K, V 投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # 空间缩减 (Spatial Reduction)
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.sr_ln = nn.LayerNorm(d_model)
        else:
            self.sr = nn.Identity()
            self.sr_ln = nn.Identity()

        # 门控投影
        self.gate_proj = nn.Linear(d_model, d_model if (headwise and elementwise) else n_heads)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 预处理与 LayerNorm
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        # 2. 生成 Query (Q)
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 3. 空间缩减生成 Key(K) 和 Value(V)
        if self.reduction_ratio > 1:
            x_sr = self.sr(x)
            x_sr = rearrange(x_sr, 'b c h w -> b (h w) c')
            x_sr = self.sr_ln(x_sr)
        else:
            x_sr = x_norm

        k = self.w_k(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 4. 使用官方优化版 SDPA
        # 注意：SDPA 内部已自动处理 scale (1/sqrt(dk))，无需手动除以 d_head**0.5
        # 它会自动根据硬件选择 FlashAttention 或 Memory Efficient Kernel
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        # 5. 门控逻辑 (Gating)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = y * gate

        # 6. 还原维度
        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)

        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

# --- 适配 YOLOv11 的 C3k2 结构 ---

class Bottleneck_LiteGatedAttention_SDPA(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, r=2):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = LiteGatedAttention_SDPA(c2, reduction_ratio=r)

    def forward(self, x):
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C3k2_LiteGatedAttention_SDPA(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, r=2):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            Bottleneck_LiteGatedAttention_SDPA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0, r=r)
            for _ in range(n)
        )



#对比



class LiteAttention_SDPA(nn.Module):
    def __init__(self, d_model, n_heads=8, reduction_ratio=2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.reduction_ratio = reduction_ratio

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.sr_ln = nn.LayerNorm(d_model)
        else:
            self.sr = nn.Identity()
            self.sr_ln = nn.Identity()

        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        if self.reduction_ratio > 1:
            x_sr = self.sr(x)
            x_sr = rearrange(x_sr, 'b c h w -> b (h w) c')
            x_sr = self.sr_ln(x_sr)
        else:
            x_sr = x_norm

        k = self.w_k(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 官方 SDPA 优化
        y = F.scaled_dot_product_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

class StandardAttention_SDPA(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        # 标准 QKV 生成，长度均为 H*W
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 使用 SDPA 优化（在 4090 上处理长序列非常高效）
        y = F.scaled_dot_product_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
