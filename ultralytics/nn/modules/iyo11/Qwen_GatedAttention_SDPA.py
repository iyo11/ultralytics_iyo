import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from timm.layers import DropPath
except ImportError:
    DropPath = nn.Identity

__all__ = ['Qwen_GatedAttention_SDPA']


class Qwen_GatedAttention_SDPA(nn.Module):
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

        # 正则化
        self.proj_drop = nn.Dropout(dropout)
        # DropPath: 在训练过程中随机将整个分支丢弃，进一步增强泛化能力
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # LayerScale: 极其关键，初始值 1e-5 使模块从近乎“恒等映射”开始学习
        self.gamma = nn.Parameter(layerscale_init * torch.ones((d_model)), requires_grad=True)

        # 空间压缩逻辑
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
        if self.gate_proj.bias is not None:
            nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(self, x):
        # 【新增】：保存输入作为残差
        identity = x

        b, c, h, w = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        # --- 注意力分支 ---
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

        # FlashAttention (4090 优化点)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_p)

        # --- 门控调制分支 ---
        gate_input = rearrange(x_norm, 'b (h w) c -> b c h w', h=h, w=w)
        gate = self.gate_proj(gate_input)
        gate = rearrange(gate, 'b c h w -> b (h w) c')
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = y * torch.sigmoid(gate)

        # --- 合并与投影 ---
        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)

        # LayerScale + Dropout
        out = self.proj_drop(self.w_o(y * self.gamma))

        # 还原回 4D 结构
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        # 【核心改进】：返回时加上残差连接
        return identity + self.drop_path(out)