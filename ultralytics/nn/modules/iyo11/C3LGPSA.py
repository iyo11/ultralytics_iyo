import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules import Conv

__all__ = ['C3LGPSA']


class GatedSpatialAttention(nn.Module):
    """带门控的空间注意力模块 (GSA)"""

    def __init__(self, c, kernel_size=7):
        super().__init__()
        # 空间路径：关注“在哪里看”
        self.spa_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # 门控路径：关注“哪些特征重要” (类似 LGA 的 gate_proj)
        self.gate_conv = nn.Conv2d(c, c, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 空间权重生成
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spa_weight = self.sigmoid(self.spa_conv(torch.cat([avg_out, max_out], dim=1)))

        # 2. 门控信号生成
        gate = self.sigmoid(self.gate_conv(x))

        # 3. 融合：输入 * 空间权重 * 门控信号
        return x * spa_weight * gate


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

        # RTX 4090 优化
        y = F.scaled_dot_product_attention(q, k, v)

        gate = torch.sigmoid(self.gate_proj(x_norm))
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = y * gate

        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)


class C3LGPSA(nn.Module):
    """
    C3LGPSA: 带有双重门控的三路并行模块
    1. Identity: 保留原始信息
    2. LGA: 门控注意力 (处理全局/通道关系)
    3. GSA: 门控空间注意力 (处理局部显著性)
    """

    def __init__(self, c1, c2, n=1, e=0.5, r=2):
        super().__init__()
        assert c1 == c2
        self.c_part = int(c1 * e)
        self.cv1 = Conv(c1, 3 * self.c_part, 1, 1)
        self.cv2 = Conv(3 * self.c_part, c1, 1)

        # LGA 分支
        self.lga_branch = nn.Sequential(
            *(LGA_SDPA(self.c_part, n_heads=self.c_part // 64 if self.c_part >= 64 else 1, reduction_ratio=r) for _ in
              range(n))
        )

        # 修改后的门控空间分支 (GSA)
        self.gsa_branch = GatedSpatialAttention(self.c_part, kernel_size=7)

    def forward(self, x):
        x_a, x_b, x_c = self.cv1(x).chunk(3, 1)

        # 三路并行并行计算
        out_a = x_a
        out_b = self.lga_branch(x_b)
        out_c = self.gsa_branch(x_c)

        return self.cv2(torch.cat((out_a, out_b, out_c), 1))