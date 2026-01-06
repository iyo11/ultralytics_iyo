import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from timm.layers import DropPath
except ImportError:
    DropPath = nn.Identity

__all__ = ['Qwen_GatedAttention_RWD']


class Qwen_GatedAttention_RWD(nn.Module):
    def __init__(self, d_model, n_heads=8, reduction_ratio=1, dropout=0.1, layerscale_init=1e-5, drop_path=0.):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.reduction_ratio = reduction_ratio

        # Q, K, V 映射
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 空间感知门控 (DWConv)
        self.gate_proj = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 正则化三件套
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # LayerScale: 极其关键，防止初始权重过大破坏预训练特征流
        self.gamma = nn.Parameter(layerscale_init * torch.ones((d_model)), requires_grad=True)

        # 【要求 1】：空间压缩使用 DWConv
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio, groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)
        else:
            self.sr = nn.Identity()
            self.sr_ln = nn.Identity()

        self.ln = nn.LayerNorm(d_model)
        self.attn_drop_p = dropout if self.training else 0.0

        self._init_weights()

    def _init_weights(self):
        # 门控初始化为负，初期倾向于关闭，让残差主导，保证训练平滑
        if self.gate_proj.bias is not None:
            nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(self, x):
        # 【要求 2】：残差连接起始点
        identity = x

        b, c, h, w = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        if self.reduction_ratio > 1:
            x_sr = self.sr(x)
            x_sr = rearrange(x_sr, 'b c h w -> b (h w) c')
            x_sr = self.sr_ln(x_sr)
            k = self.w_k(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        else:
            k = self.w_k(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 4090 FlashAttention
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_p)

        # 门控计算
        gate_input = rearrange(x_norm, 'b (h w) c -> b c h w', h=h, w=w)
        gate = self.gate_proj(gate_input)
        gate = rearrange(gate, 'b c h w -> b (h w) c')
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = y * torch.sigmoid(gate)

        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)

        # LayerScale + Dropout
        out = self.proj_drop(self.w_o(y * self.gamma))
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        # 残差融合
        return identity + self.drop_path(out)