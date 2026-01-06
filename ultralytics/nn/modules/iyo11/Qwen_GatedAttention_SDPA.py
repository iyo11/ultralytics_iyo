import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['Qwen_GatedAttention_SDPA']


class Qwen_GatedAttention_SDPA(nn.Module):
    """
    Qwen 门控注意力的视觉进化版：
    1. 移除残差：不再执行 shortcut + out，适配外部 Bottleneck。
    2. 空间压缩：SR 改为标准卷积，增强跨通道特征学习。
    3. 门控机制：保留 3x3 空间感知门控与负偏置初始化。
    4. 性能优化：适配 4090 的 SDPA 计算。
    """

    def __init__(self, d_model, n_heads=8, reduction_ratio=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.reduction_ratio = reduction_ratio

        # Q, K, V 映射
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 空间感知门控 (Spatial-Aware Gate)
        # 仍然保留 DWConv 门控以实现高效的 3x3 局部空间建模
        self.gate_proj = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=True)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 【修改】：空间压缩 - 改为标准卷积 (去除 groups=d_model)
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.sr_ln = nn.LayerNorm(d_model)
        else:
            self.sr = nn.Identity()
            self.sr_ln = nn.Identity()

        self.ln = nn.LayerNorm(d_model)

        # 执行权重初始化
        self._init_weights()

    def _init_weights(self):
        # 延续 Qwen 建议：门控偏置负初始化，抑制背景噪声
        if self.gate_proj.bias is not None:
            nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(self, x):
        b, c, h, w = x.shape

        # 准备进入注意力层
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        # 1. 生成 Query
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 2. 生成 Key, Value (标准卷积空间压缩)
        if self.reduction_ratio > 1:
            x_sr = self.sr(x)
            x_sr = rearrange(x_sr, 'b c h w -> b (h w) c')
            x_sr = self.sr_ln(x_sr)
            k = self.w_k(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        else:
            k = self.w_k(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 3. SDPA 高效计算
        y = F.scaled_dot_product_attention(q, k, v)

        # 4. 空间感知门控调制
        gate_input = rearrange(x_norm, 'b (h w) c -> b c h w', h=h, w=w)
        gate = self.gate_proj(gate_input)

        # 变回注意力形状：[b, n_heads, seq_len, d_head]
        gate = rearrange(gate, 'b c h w -> b (h w) c')
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 逐元素相乘
        y = y * torch.sigmoid(gate)

        # 5. 合并输出
        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)

        # 【修改】：直接返回输出，移除了 shortcut
        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)