import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['Qwen_GatedAttention_SDPA']


class Qwen_GatedAttention_SDPA(nn.Module):
    """
    基于 NeurIPS 2025 最佳论文 (Alibaba Qwen Team) 实现的门控注意力。
    核心点：在 SDPA 输出后直接施加 Sigmoid 门控。
    """

    def __init__(self, d_model, n_heads=8, reduction_ratio=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.reduction_ratio = reduction_ratio

        # Q, K, V 投影
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 【核心改进】：论文建议的元素级/头级门控
        # 该线性层用于生成门控分值，直接作用于 SDPA 的输出
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

        # 输出投影
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 空间压缩 (针对 YOLO 小图优化)
        if reduction_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio, groups=d_model),
                nn.BatchNorm2d(d_model)  # YOLO 体系更适配 BN
            )
        else:
            self.sr = nn.Identity()

        self.ln = nn.LayerNorm(d_model)

        # 可训练的残差系数：确保模块插入 SPPF 后不破坏原有预训练权重
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x  # 保存残差

        # 展平特征图: [B, C, H, W] -> [B, N, C]
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        # 1. 生成 Q
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 2. 生成 K, V (带空间压缩以适配低算力或 SPPF 后小图)
        if self.reduction_ratio > 1:
            x_spatial = rearrange(self.sr(x), 'b c h w -> b (h w) c')
        else:
            x_spatial = x_norm

        k = self.w_k(x_spatial).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x_spatial).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 3. 标准 SDPA (4090 上触发 FlashAttention)
        # attn_out 形状: [b, n_heads, seq_len, d_head]
        attn_out = F.scaled_dot_product_attention(q, k, v)

        # 4. 【最佳论文核心实现】：Query-Dependent Gating
        # 论文指出：在 Softmax(QK^T)V 后乘以 Sigmoid(Gate)
        # 我们先将注意力输出还原到 d_model 维度
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, -1, self.d_model)

        # 生成门控信号并应用 Sigmoid
        # 门控是基于输入特征 x 动态生成的（Query-dependent）
        gate = torch.sigmoid(self.gate_proj(x_norm))

        # 门控调制：这里实现了论文中的 Element-wise gating
        gated_attn = attn_out * gate

        # 5. 输出投影与残差
        out = self.w_o(gated_attn)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        # 这里的 gamma 初始化为0，保证了插入 SPPF 后的初始稳定性
        return shortcut + self.gamma * out