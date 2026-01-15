import torch
import torch.nn as nn
import torch.nn.functional as F


class HFP_Light(nn.Module):
    """
    轻量化高频感知模块 (Lightweight HFP)
    保持不变，已优化。
    """

    def __init__(self, c1, alpha=0.25):
        super().__init__()
        self.alpha = alpha
        self.hidden_dim = max(8, c1 // 4)
        self.cp_conv1 = nn.Conv1d(c1, self.hidden_dim, 1)
        self.cp_conv2 = nn.Conv1d(self.hidden_dim, c1, 1)
        self.sp_conv = nn.Conv2d(c1, 1, 1)
        self.fusion_conv = nn.Conv2d(c1, c1, 1)

    def dct_filter(self, x):
        B, C, H, W = x.shape
        # 使用 float32 进行 FFT 以避免半精度溢出
        x_fp32 = x.float()
        fft_x = torch.fft.rfft2(x_fp32, norm='backward')

        mask = torch.ones_like(fft_x, device=x.device)
        h_cut, w_cut = int(H * self.alpha), int(W * (self.alpha / 2))
        mask[:, :, :h_cut, :w_cut] = 0

        x_high = torch.fft.irfft2(fft_x * mask, s=(H, W), norm='backward')
        return x_high.type_as(x)  # 恢复原始精度

    def forward(self, x):
        f_i = self.dct_filter(x)

        # Channel Path
        gap = F.adaptive_avg_pool2d(f_i, 1).squeeze(-1)
        gmp = F.adaptive_max_pool2d(f_i, 1).squeeze(-1)

        y_gap = self.cp_conv2(F.relu(self.cp_conv1(gap)))
        y_gmp = self.cp_conv2(F.relu(self.cp_conv1(gmp)))
        w_cp = torch.sigmoid(y_gap + y_gmp).unsqueeze(-1)

        # Spatial Path
        w_sp = torch.sigmoid(self.sp_conv(f_i))

        out = x * w_cp * w_sp + x
        return self.fusion_conv(out)


class SDP_Block(nn.Module):
    """
    分块空间依赖感知模块 (Block-wise SDP)
    严格遵循论文 Section 3.2，将特征图分块计算注意力，解决 OOM 问题。
    """

    def __init__(self, c_lateral, c_top, window_size=8, ratio=2):
        super().__init__()
        self.window_size = window_size
        self.proj_p = nn.Conv2d(c_top, c_lateral, 1) if c_top != c_lateral else nn.Identity()

        # 降维以进一步节省显存
        self.head_dim = max(16, c_lateral // ratio)

        self.conv_q = nn.Conv2d(c_lateral, self.head_dim, 1)
        self.conv_k = nn.Conv2d(c_lateral, self.head_dim, 1)
        self.conv_v = nn.Conv2d(c_lateral, c_lateral, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def window_partition(self, x, window_size):
        """
        将特征图 (B, C, H, W) 划分为窗口 (B*num_windows, window_size*window_size, C)
        """
        B, C, H, W = x.shape
        # Pad 如果 H, W 不能被 window_size 整除
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        H_pad, W_pad = x.shape[2], x.shape[3]

        # Reshape 为 (B, C, H//ws, ws, W//ws, ws)
        x = x.view(B, C, H_pad // window_size, window_size, W_pad // window_size, window_size)

        # Permute 为 (B, H//ws, W//ws, ws, ws, C) -> 合并 batch 和 windows
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
        return windows, H_pad, W_pad

    def window_reverse(self, windows, window_size, H_pad, W_pad, B, C):
        """
        将窗口还原为特征图
        """
        H_wins, W_wins = H_pad // window_size, W_pad // window_size
        x = windows.view(B, H_wins, W_wins, window_size, window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H_pad, W_pad)
        return x

    def forward(self, c_feat, p_feat):
        # 1. 投影与对齐
        p_feat = self.proj_p(p_feat)
        B, C, H, W = c_feat.shape

        # 2. 生成 Q, K, V
        q = self.conv_q(c_feat)  # (B, head_dim, H, W)
        k = self.conv_k(p_feat)  # (B, head_dim, H, W)
        v = self.conv_v(p_feat)  # (B, C, H, W)

        # 3. 分块 (Window Partition)
        # 将全图 attention 转换为局部窗口 attention，显存消耗从 O((HW)^2) 降为 O(HW * window^2)
        q_windows, H_pad, W_pad = self.window_partition(q, self.window_size)  # (N_wins, ws*ws, head_dim)
        k_windows, _, _ = self.window_partition(k, self.window_size)
        v_windows, _, _ = self.window_partition(v, self.window_size)  # (N_wins, ws*ws, C)

        # 4. 局部 Attention 计算
        # 使用 PyTorch 内置的 scaled_dot_product_attention (自动优化显存)
        # q_windows shape: (Batch_Wins, Tokens, Dim) -> 需要 permute 为 (B, Heads, Tokens, Dim) 但这里是单头简化版

        # 注意：conv_q 输出是 (B, head_dim, H, W)，partition 后最后一维是 head_dim
        # scaled_dot_product_attention 期望 (Batch, ..., L, E)
        # 我们这里把 (B*num_windows) 当作 Batch 维度处理

        attn_out = F.scaled_dot_product_attention(q_windows, k_windows, v_windows)  # (N_wins, ws*ws, C)

        # 5. 还原 (Reverse)
        out = self.window_reverse(attn_out, self.window_size, H_pad, W_pad, B, C)

        # 去除 Padding
        if H_pad > H or W_pad > W:
            out = out[:, :, :H, :W]

        return c_feat + self.gamma * out


class HSFPN_Fusion(nn.Module):
    """
    HS-FPN 融合模块 (Block-wise 版本)
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.hfp = HFP_Light(c1)
        # 使用 Block 版 SDP，默认窗口大小 8 (对应论文中 P2相对于P5的比例)
        self.sdp = SDP_Block(c1, c2, window_size=8, ratio=2)

    def forward(self, x):
        p_feat = x[0]
        c_feat = x[1]
        c_enhanced = self.hfp(c_feat)
        out = self.sdp(c_enhanced, p_feat)
        return out