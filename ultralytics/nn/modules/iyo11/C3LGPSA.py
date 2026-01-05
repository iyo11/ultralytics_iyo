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
        self.spa_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.gate_conv = nn.Conv2d(c, c, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spa_weight = self.sigmoid(self.spa_conv(torch.cat([avg_out, max_out], dim=1)))
        gate = self.sigmoid(self.gate_conv(x))
        return x * spa_weight * gate


class LGA_SDPA(nn.Module):
    """Lite Gated Attention with SDPA optimization."""

    def __init__(self, d_model, n_heads=8, reduction_ratio=2):
        super().__init__()
        # 确保头数能被整除，若平分后通道太小，默认至少 1 个头
        self.n_heads = max(1, n_heads)
        assert d_model % self.n_heads == 0, f"d_model {d_model} must be divisible by n_heads {self.n_heads}"

        self.d_model = d_model
        self.d_head = d_model // self.n_heads
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


class C3LGPSA(nn.Module):
    """
    C3LGPSA: 全通道并行版本
    融合策略: Concat + Conv 1x1 Fusion (替代原有的 split + softmax)
    """

    def __init__(self, c1, c2, n=1, e=0.5, r=2):
        super().__init__()
        assert c1 == c2
        self.c = c1

        # 1. 预处理卷积
        self.cv1 = Conv(c1, c1, 1, 1)

        # 2. 三路分支 (全部使用完整通道 c1)
        # 第一路: Identity (无参数，直接用 x_in)

        # 第二路: LGA (全局注意力)
        # 此时 c1 通常是 16, 32, 64 等，n_heads 必能整除
        n_heads = max(1, c1 // 32)
        self.lga_branch = nn.Sequential(
            *(LGA_SDPA(c1, n_heads=n_heads, reduction_ratio=r) for _ in range(n))
        )

        # 第三路: GSA (局部空间注意力)
        self.gsa_branch = GatedSpatialAttention(c1, kernel_size=7)

        # 3. 融合层: 将三路 (c1 + c1 + c1) 融合回 c1
        self.cv_fusion = Conv(c1 * 3, c1, 1, 1)

        # 4. 最后的输出缩放/调整
        self.cv2 = Conv(c1, c1, 1)

    def forward(self, x):
        x_in = self.cv1(x)

        # 并行计算三路特征
        out_identity = x_in
        out_lga = self.lga_branch(x_in)
        out_spa = self.gsa_branch(x_in)

        # Concat 拼接: [B, 3*C, H, W]
        combined = torch.cat([out_identity, out_lga, out_spa], dim=1)

        # 通道融合
        out = self.cv_fusion(combined)

        # 全局残差连接
        return x + self.cv2(out)