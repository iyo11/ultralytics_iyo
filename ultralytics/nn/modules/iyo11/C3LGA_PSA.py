import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules import Conv

__all__ = ['C3LGA_PSA']

class LGA_SDPA(nn.Module):
    """Lite Gated Attention with SDPA optimization (原版)。"""
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

        self.gate_proj = nn.Linear(d_model, d_model)
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

        y = F.scaled_dot_product_attention(q, k, v)

        gate = torch.sigmoid(self.gate_proj(x_norm))
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = y * gate

        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)


class PSABlock(nn.Module):
    """原 PSA Block，可以作为三路之一"""
    def __init__(self, c, n_heads=8, r=2):
        super().__init__()
        self.attn = LGA_SDPA(c, n_heads=n_heads, reduction_ratio=r)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C3LGA_PSA(nn.Module):
    """
    三路融合模块：
    - Identity (50%)
    - PSA (30%)
    - LGA (20%)
    带门控融合，适合中小目标检测
    """
    def __init__(self, c: int, r: int = 2):
        super().__init__()
        c_mid = c

        self.c_id  = int(0.5 * c_mid)
        self.c_psa = int(0.3 * c_mid)
        self.c_lga = c_mid - self.c_id - self.c_psa  # 剩余通道

        # 通道变换
        self.cv1 = Conv(c, c_mid, 1)

        # 三路模块
        self.psa = PSABlock(self.c_psa)
        self.lga = LGA_SDPA(self.c_lga, reduction_ratio=r)

        # PSA门控（由 Identity 控制）
        self.gate_psa = nn.Sequential(
            Conv(self.c_id, self.c_psa, 1),
            nn.Sigmoid()
        )
        # LGA门控（由全局上下文控制）
        self.gate_lga = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c_mid, self.c_lga, 1),
            nn.Sigmoid()
        )

        # 输出融合后的 FFN
        self.ffn = nn.Sequential(
            Conv(c_mid, c_mid * 2, 1),
            Conv(c_mid * 2, c_mid, 1, act=False)
        )

    def forward(self, x):
        x = self.cv1(x)
        x_id, x_psa, x_lga = torch.split(x, [self.c_id, self.c_psa, self.c_lga], dim=1)

        # 三路
        psa_out = self.psa(x_psa) * self.gate_psa(x_id)
        lga_out = self.lga(x_lga) * self.gate_lga(x)

        # 加权融合
        y = x_id + psa_out + lga_out
        y = y + self.ffn(y)
        return y
