import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['Qwen_GatedAttention_SDPA']


class Qwen_GatedAttention_SDPA(nn.Module):
    """
    专注改进动态门控逻辑的版本。
    1. 保持名字不变。
    2. 采用多头对齐（Head-wise）的动态门控。
    3. 简单的残差连接，确保梯度直接回传。
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

        # 【动态门控核心】：生成与每个 Head 维度一致的门控分值
        # Qwen 论文建议：门控应当具有足够的表达力来过滤注意力噪声
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 空间压缩逻辑
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio, groups=d_model)
        else:
            self.sr = nn.Identity()

        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        # 记录原始输入用于残差
        shortcut = x

        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        # 1. 生成 Q, K, V (K,V 可选空间压缩)
        q = self.w_q(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        if self.reduction_ratio > 1:
            x_spatial = rearrange(self.sr(x), 'b c h w -> b (h w) c')
            # 这里的 K, V 投影
            k = self.w_k(x_spatial).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_spatial).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        else:
            k = self.w_k(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.w_v(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 2. SDPA 注意力计算 (4090 优化路径)
        # y 的形状是 [b, n_heads, seq_len, d_head]
        y = F.scaled_dot_product_attention(q, k, v)

        # 3. 【核心步骤：动态门控调制】
        # 这里是动态性的来源：gate 是根据当前输入 x_norm 实时生成的
        # 将 gate 拆分为多头形状，确保每一个 Head 都能被精准控制
        gate = self.gate_proj(x_norm).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        gate = torch.sigmoid(gate)

        # 逐元素相乘：这实现了 NeurIPS 2025 论文中的 Head-wise 过滤
        y = y * gate

        # 4. 合并多头并映射输出
        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)

        # 还原回图像形状
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        # 5. 标准残差相加 (x + out)
        return shortcut + out