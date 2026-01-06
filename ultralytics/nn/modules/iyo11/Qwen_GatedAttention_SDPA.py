import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['Qwen_GatedAttention_SDPA']


class Qwen_GatedAttention_SDPA(nn.Module):
    """
    Qwen 门控注意力的视觉进化版：
    1. 门控升级：从 Linear 改为 3x3 DWConv，引入局部空间上下文。
    2. 保持 SDPA：适配 4090 的高效注意力计算。
    3. 负偏置初始化：抑制背景噪声，提高训练稳定性。
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

        # 【核心改进】：空间感知门控 (Spatial-Aware Gate)
        # 使用 groups=d_model (DWConv) 可以在不显著增加参数量的情况下，获得 3x3 的感受野
        self.gate_proj = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=True)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 空间压缩逻辑
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio, groups=d_model)
        else:
            self.sr = nn.Identity()

        self.ln = nn.LayerNorm(d_model)

        # 执行权重初始化
        self._init_weights()

    def _init_weights(self):
        # 延续 Qwen 建议：门控偏置负初始化，默认让门处于“微闭”状态，过滤背景
        if self.gate_proj.bias is not None:
            nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x

        # 准备进入注意力层
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        # 1. 生成 Q, K, V
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        if self.reduction_ratio > 1:
            x_spatial = rearrange(self.sr(x), 'b c h w -> b (h w) c')
            k = self.w_k(x_spatial).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_spatial).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        else:
            k = self.w_k(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 2. SDPA 高效计算
        y = F.scaled_dot_product_attention(q, k, v)

        # 3. 【空间感知门控调制】
        # 将 x_norm 还原为 4D 形状进行卷积，从而让门控能“看周围”
        # 这里用 x_norm 而不是 x，是为了保证门控信号是经过归一化的，训练更稳
        gate_input = rearrange(x_norm, 'b (h w) c -> b c h w', h=h, w=w)
        gate = self.gate_proj(gate_input)

        # 变回注意力形状：[b, n_heads, seq_len, d_head]
        gate = rearrange(gate, 'b c h w -> b (h w) c')
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        gate = torch.sigmoid(gate)

        # 逐元素相乘
        y = y * gate

        # 4. 合并输出
        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)

        # 还原形状
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        return shortcut + out