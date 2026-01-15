import torch
import torch.nn as nn
import torch.nn.functional as F


class HFP(nn.Module):
    """
    轻量化高频感知模块 (Lightweight HFP)
    优化点：
    1. 融合层改为 1x1 卷积 (减少 9倍 参数)。
    2. 通道注意力 MLP 缩减比率增加 (c//4)。
    """

    def __init__(self, c1, alpha=0.25):
        super().__init__()
        self.alpha = alpha

        # 优化1: 减少 MLP 中间层通道数 (c//2 -> c//4)
        self.hidden_dim = max(8, c1 // 4)
        self.cp_conv1 = nn.Conv1d(c1, self.hidden_dim, 1)
        self.cp_conv2 = nn.Conv1d(self.hidden_dim, c1, 1)

        # 空间路径
        self.sp_conv = nn.Conv2d(c1, 1, 1)

        # 优化2: 将 3x3 卷积改为 1x1 卷积
        # 3x3 conv param = 9 * C^2
        # 1x1 conv param = 1 * C^2 -> 节省 89% 参数
        self.fusion_conv = nn.Conv2d(c1, c1, 1)

    def dct_filter(self, x):
        """利用 FFT 模拟高通滤波器"""
        B, C, H, W = x.shape
        fft_x = torch.fft.rfft2(x, norm='backward')
        mask = torch.ones_like(fft_x, device=x.device)
        h_cut, w_cut = int(H * self.alpha), int(W * (self.alpha / 2))
        mask[:, :, :h_cut, :w_cut] = 0
        x_high = torch.fft.irfft2(fft_x * mask, s=(H, W), norm='backward')
        return x_high

    def forward(self, x):
        f_i = self.dct_filter(x)

        # Channel Path (无 permute 问题)
        gap = F.adaptive_avg_pool2d(f_i, 1).squeeze(-1)
        gmp = F.adaptive_max_pool2d(f_i, 1).squeeze(-1)

        y_gap = self.cp_conv2(F.relu(self.cp_conv1(gap)))
        y_gmp = self.cp_conv2(F.relu(self.cp_conv1(gmp)))
        w_cp = torch.sigmoid(y_gap + y_gmp).unsqueeze(-1)

        # Spatial Path
        w_sp = torch.sigmoid(self.sp_conv(f_i))

        # 融合
        out = x * w_cp * w_sp + x
        return self.fusion_conv(out)


class SDP(nn.Module):
    """
    轻量化空间依赖感知模块 (Lightweight SDP)
    优化点：
    1. 引入 reduction ratio，降低 Q, K 的通道维度。
    """

    def __init__(self, c_lateral, c_top, ratio=2):
        super().__init__()
        self.proj_p = nn.Conv2d(c_top, c_lateral, 1) if c_top != c_lateral else nn.Identity()

        # 优化3: 降低注意力计算的维度 (Key/Query Dimension Reduction)
        # 例如 128 -> 64，参数量减少一半
        self.head_dim = max(16, c_lateral // ratio)

        self.conv_q = nn.Conv2d(c_lateral, self.head_dim, 1)
        self.conv_k = nn.Conv2d(c_lateral, self.head_dim, 1)

        # Value 保持原维度以保留特征信息，或者也可以缩减 (此处选择保留以维持精度)
        self.conv_v = nn.Conv2d(c_lateral, c_lateral, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, c_feat, p_feat):
        p_feat = self.proj_p(p_feat)
        B, C, H, W = c_feat.shape

        # Q, K 投影到低维 (B, head_dim, N)
        q = self.conv_q(c_feat).view(B, self.head_dim, -1).permute(0, 2, 1)
        k = self.conv_k(p_feat).view(B, self.head_dim, -1)

        # V 保持原维 (B, C, N)
        v = self.conv_v(p_feat).view(B, C, -1).permute(0, 2, 1)

        # Attention (B, N, N)
        attn = torch.bmm(q, k)
        attn = F.softmax(attn * (self.head_dim ** -0.5), dim=-1)

        # Output (B, C, H, W)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)

        return c_feat + self.gamma * out


class HSFPN_Fusion(nn.Module):
    """
    轻量化融合模块总成
    """

    def __init__(self, c1, c2):
        super().__init__()
        # 使用轻量化版本
        self.hfp = HFP(c1)
        self.sdp = SDP(c1, c2, ratio=2)

    def forward(self, x):
        p_feat = x[0]
        c_feat = x[1]
        c_enhanced = self.hfp(c_feat)
        out = self.sdp(c_enhanced, p_feat)
        return out