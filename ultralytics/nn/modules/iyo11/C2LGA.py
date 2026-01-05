import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules import Conv

__all__ = ['C2LGA']


class LGA_SDPA(nn.Module):
    """Lite Gated Attention with SDPA optimization."""

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

        # 空间压缩：减小 K, V 的序列长度
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.sr_ln = nn.LayerNorm(d_model)
        else:
            self.sr = nn.Identity()
            self.sr_ln = nn.Identity()

        # 门控分支
        self.gate_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        # 生成 Query
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 生成 Key, Value (带空间压缩)
        if self.reduction_ratio > 1:
            x_sr = self.sr(x)
            x_sr = rearrange(x_sr, 'b c h w -> b (h w) c')
            x_sr = self.sr_ln(x_sr)
        else:
            x_sr = x_norm

        k = self.w_k(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # --- 使用 SDPA 优化内核 ---
        # 4090 上此操作会自动触发 FlashAttention-2
        y = F.scaled_dot_product_attention(q, k, v)

        # 门控调制
        gate = torch.sigmoid(self.gate_proj(x_norm))
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = y * gate

        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)


class PSABlock_LGA(nn.Module):
    """融合了 LGA 的 Bottleneck，参考 C2PSA 的 Block 设计。"""

    def __init__(self, c, n_heads=8, r=2):
        super().__init__()
        # 这里的 c 已经是 split 后的通道数了 (通常是总通道数的一半)
        self.attn = LGA_SDPA(c, n_heads=n_heads, reduction_ratio=r)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C2LGA(nn.Module):
    """
    C2LGA: 基于 C2PSA 结构改进的 Lite Gated Attention 模块
    对应 YOLO11 中的 C2PSA，用于直接对比实验。
    """

    def __init__(self, c1, c2, n=1, e=0.5, r=2):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 核心：只对其中一半通道执行注意力
        self.m = nn.Sequential(*(PSABlock_LGA(self.c, n_heads=self.c // 64, r=r) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class Light_LGA_SDPA(nn.Module):
    def __init__(self, d_model, n_heads=8, reduction_ratio=2):
        super().__init__()
        # 自动调整 head 数量：如果通道数比 head 小，则 head 取通道数，确保能整除
        if d_model < n_heads:
            n_heads = d_model
        while d_model % n_heads != 0:
            n_heads -= 1

        self.d_model = d_model
        self.n_heads = n_heads

        # 1x1 卷积替代 Linear，直接在 4D Tensor 上操作，减少 rearrange 频率
        self.qkv_proj = nn.Conv2d(d_model, inner_dim * 3, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(inner_dim, d_model, kernel_size=1, bias=False)

        # 极简门控：使用通道注意力形式或更小的投影
        self.gate_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, inner_dim, 1),
            nn.Sigmoid()
        )

        # 空间压缩
        self.sr = nn.AvgPool2d(kernel_size=reduction_ratio) if reduction_ratio > 1 else nn.Identity()
        self.bn = nn.BatchNorm2d(d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.bn(x)

        # 1. 生成 QKV (使用 Conv2d 效率更高)
        qkv = self.qkv_proj(x)  # b, inner*3, h, w
        q, k, v = qkv.chunk(3, dim=1)

        # 空间压缩 K, V
        k, v = self.sr(k), self.sr(w)

        # 准备进入 SDPA (rearrange 移到这里)
        q = rearrange(q, 'b (g d) h w -> b g (h w) d', g=self.n_heads)
        k = rearrange(k, 'b (g d) h w -> b g (h w) d', g=self.n_heads)
        v = rearrange(v, 'b (g d) h w -> b g (h w) d', g=self.n_heads)

        # 2. SDPA 优化
        y = F.scaled_dot_product_attention(q, k, v)

        # 3. 门控调制 (此时 gate 是 b, inner, 1, 1)
        y = rearrange(y, 'b g (h w) d -> b (g d) h w', h=h, w=w)
        gate = self.gate_proj(x)
        y = y * gate

        # 4. 投影回原维度
        return self.o_proj(y)