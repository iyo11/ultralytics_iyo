import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules import Conv, C2f, C3

__all__ = ['LiteGatedAttention', 'C3k2_LiteGatedAttention']

# --- 新增 LiteGatedAttention ---

class LiteGatedAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, headwise=True, elementwise=True, reduction_ratio=2):
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

        # --- 修改部分开始 ---
        if reduction_ratio > 1:
            # 使用普通的卷积层进行降采样，后面接一个维度转换即可，不需要在这里放 LayerNorm
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=reduction_ratio, stride=reduction_ratio)
            # 增加一个专门给降采样后的特征使用的 BN 或 LN
            self.sr_ln = nn.LayerNorm(d_model)
        else:
            self.sr = nn.Identity()
            self.sr_ln = nn.Identity()
        # --- 修改部分结束 ---

        self.gate_proj = nn.Linear(d_model, d_model if (headwise and elementwise) else n_heads)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape

        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.ln(x_flat)

        q = self.w_q(x_norm)
        q = q.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        # --- 修改部分开始 ---
        if self.reduction_ratio > 1:
            # 先卷积，再转换维度，最后才过 LayerNorm
            x_sr = self.sr(x) # [B, C, H/r, W/r]
            x_sr = rearrange(x_sr, 'b c h w -> b (h w) c') # 转换成 [B, L_new, C]
            x_sr = self.sr_ln(x_sr) # 现在维度匹配了，[*, 256] 对 [*, 256]
        else:
            x_sr = x_norm
        # --- 修改部分结束 ---

        k = self.w_k(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x_sr).view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn_probs, v)

        gate = torch.sigmoid(self.gate_proj(x_norm))
        gate = gate.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        y = y * gate

        y = y.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.w_o(y)

        return rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

# --- 修改对应的 C3k2 结构 ---

class Bottleneck_LiteGatedAttention(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, r=2):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        # 使用新定义的 LiteGatedAttention
        self.Attention = LiteGatedAttention(c2, reduction_ratio=r)

    def forward(self, x):
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C3k2_LiteGatedAttention(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, r=2):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            # 这里可以根据需要微调，P3 层建议 r=4，P4/P5 建议 r=2
            Bottleneck_LiteGatedAttention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0, r=r)
            for _ in range(n)
        )