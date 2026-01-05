import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules import Conv, DWConv

__all__ = ['C2LGA_V2']

class ImprovedLGA_SDPA(nn.Module):
    """
    改进版 Lite Gated Attention:
    1. 增加 DWConv 位置编码
    2. 优化多头逻辑
    3. 稳定空间压缩分支
    """

    def __init__(self, d_model, n_heads=8, reduction_ratio=2):
        super().__init__()
        # 确保 n_heads 至少为 1 且能被整除
        self.n_heads = n_heads if d_model % n_heads == 0 else 1
        self.d_head = d_model // self.n_heads
        self.d_model = d_model
        self.reduction_ratio = reduction_ratio

        # Q, K, V 投影
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        # 隐式位置编码 (Local Context Extractor)
        self.pe = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        # 改进的空间压缩：Conv + BN
        if reduction_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio, groups=d_model),
                nn.BatchNorm2d(d_model)
            )
        else:
            self.sr = nn.Identity()

        # 轻量化门控：基于输入特征的通道调整
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model, 1),
            nn.Sigmoid()
        )

        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 注入位置信息
        x = x + self.pe(x)

        # 2. 准备 Query
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 3. 准备 Key & Value (带有空间压缩)
        x_sr = self.sr(x)
        x_sr_flat = rearrange(x_sr, 'b c h w -> b (h w) c')
        # 这里对压缩后的 K, V 同样应用 LayerNorm (可选)
        k = self.w_k(x_sr_flat).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x_sr_flat).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 4. SDPA 加速内核
        y = F.scaled_dot_product_attention(q, k, v)

        # 5. 门控调制 (使用空间/通道门控替代全量线性门控)
        # 将门控作用于 4D 特征图上，更加节省计算量
        y = y.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        y = y * self.gate_conv(x)  # 动态调整通道权重

        # 6. 输出投影
        y = rearrange(y, 'b c h w -> b (h w) c')
        out = self.w_o(y)
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)


class C2LGA_V2(nn.Module):
    """
    改进的 C2LGA: 增加参数鲁棒性
    """

    def __init__(self, c1, c2, n=1, e=0.5, r=2):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 动态计算 head，确保最小为 1 且能整除
        # 通常推荐 head_dim 为 32 或 64
        n_heads = max(1, self.c // 32)

        self.m = nn.Sequential(*(
            PSABlock_LGA_V2(self.c, n_heads=n_heads, r=r) for _ in range(n)
        ))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class PSABlock_LGA_V2(nn.Module):
    def __init__(self, c, n_heads=8, r=2):
        super().__init__()
        self.attn = ImprovedLGA_SDPA(c, n_heads=n_heads, reduction_ratio=r)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x