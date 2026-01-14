import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 轻量 DropPath（可选）
# -----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B,1,1,1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * random_tensor


# -----------------------------
# 自写简易 VSSBlock：四向 1D 扫描 + 门控
# (B,C,H,W) -> (B,C,H,W)
# -----------------------------
class SimpleVSSBlock(nn.Module):
    """
    简易 VSSBlock（不依赖 selective_scan）：
    - norm(GN) -> 1x1 in_proj -> split(value, gate)
    - dwconv2d 做局部
    - 四向1D（H正/反 + W正/反）用 depthwise Conv1d 做序列混合
    - 融合后 out_proj
    - residual + droppath

    参数名尽量贴近你原来 VSSBlock 的用法，方便替换。
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = None,
        ssm_ratio: float = 2.0,
        ssm_d_state: int = 16,     # 占位：这里不用，但保留接口
        ssm_conv: int = 3,         # 1D conv kernel
        ssm_drop_rate: float = 0.0,
        mlp_ratio: float = 0.0,    # 占位：这里不做 MLP
        mlp_drop_rate: float = 0.0 # 占位
    ):
        super().__init__()
        c = in_channels if hidden_dim is None else hidden_dim
        assert c == in_channels, "简易版为了稳定融合，建议 hidden_dim == in_channels"

        self.norm = nn.GroupNorm(num_groups=min(32, c), num_channels=c)

        # in_proj: 生成 value + gate
        self.in_proj = nn.Conv2d(c, 2 * c, kernel_size=1, bias=True)

        # 局部 DWConv2d（类似 VSSBlock 里的局部分支）
        self.local_dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)

        k = int(ssm_conv)
        if k % 2 == 0:
            k += 1  # 保证 odd，padding更对称
        pad = k // 2

        # 四向 1D：沿 H 的 Conv1d (depthwise)
        self.h_conv = nn.Conv1d(c, c, kernel_size=k, padding=pad, groups=c, bias=False)
        # 沿 W 的 Conv1d (depthwise)
        self.w_conv = nn.Conv1d(c, c, kernel_size=k, padding=pad, groups=c, bias=False)

        self.act = nn.SiLU()
        self.drop = nn.Dropout(ssm_drop_rate) if ssm_drop_rate > 0 else nn.Identity()

        self.out_proj = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.drop_path = DropPath(ssm_drop_rate)

        # 一个小技巧：让块初始更“像恒等映射”
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _scan_h(self, x):
        # x: [B,C,H,W] -> treat H as sequence length
        b, c, h, w = x.shape
        seq = x.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)  # [B*W, C, H]
        y_f = self.h_conv(seq)
        y_b = torch.flip(self.h_conv(torch.flip(seq, dims=[2])), dims=[2])
        y = 0.5 * (y_f + y_b)
        y = y.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()     # [B,C,H,W]
        return y

    def _scan_w(self, x):
        # x: [B,C,H,W] -> treat W as sequence length
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)  # [B*H, C, W]
        y_f = self.w_conv(seq)
        y_b = torch.flip(self.w_conv(torch.flip(seq, dims=[2])), dims=[2])
        y = 0.5 * (y_f + y_b)
        y = y.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()     # [B,C,H,W]
        return y

    def forward(self, x):
        # Pre-norm
        x0 = x
        x = self.norm(x)

        # in_proj -> value + gate
        vg = self.in_proj(x)
        v, g = torch.chunk(vg, 2, dim=1)
        g = torch.sigmoid(g)

        # local + four-direction 1D scan
        v = self.act(v)
        v_local = self.local_dw(v)
        v_h = self._scan_h(v)
        v_w = self._scan_w(v)

        v_mix = (v_local + v_h + v_w) / 3.0
        v_mix = self.act(v_mix)
        v_mix = self.drop(v_mix)

        out = self.out_proj(v_mix * g)

        # residual
        out = x0 + self.drop_path(out)
        return out


# -----------------------------
# 你的 HybridSS2DUpsample：把 VSSBlock 换成 SimpleVSSBlock
# -----------------------------
class HybridSS2DUpsample(nn.Module):
    """
    结构：
      A: 2D bilinear ↑ -> refine
      B: 2D bilinear ↑ -> SimpleVSSBlock(自写简易四向1D扫描)
      gate: sigmoid(Conv1x1(concat(A,B)))，并且 gate 最后一层 0-init
      out: A + gate * B
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        scale: int = 2,
        # ---- B: 简易VSSBlock 超参 ----
        ssm_ratio: float = 2.0,
        ssm_d_state: int = 16,
        ssm_conv: int = 3,
        ssm_drop_rate: float = 0.0,
        # ---- gate ----
        gate_hidden_ratio: float = 0.25,
        # ---- A refine ----
        refine_dw_kernel: int = 3,
        use_pw: bool = True,
        # ---- interpolate ----
        mode: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        assert scale >= 1
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners if mode in ("bilinear", "bicubic") else None

        # 通道对齐
        self.pre = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        ) if c1 != c2 else nn.Identity()

        # A: refine
        refine = [
            nn.Conv2d(c2, c2, kernel_size=refine_dw_kernel, padding=refine_dw_kernel // 2,
                      groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        ]
        if use_pw:
            refine += [
                nn.Conv2d(c2, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(),
            ]
        self.refineA = nn.Sequential(*refine)

        # B: 使用自写简易VSSBlock
        self.ssmB = SimpleVSSBlock(
            in_channels=c2,
            hidden_dim=c2,
            ssm_ratio=ssm_ratio,
            ssm_d_state=ssm_d_state,
            ssm_conv=ssm_conv,
            ssm_drop_rate=ssm_drop_rate,
            mlp_ratio=0.0,
            mlp_drop_rate=0.0,
        )

        # gate：0-init 保证初期输出≈A
        gate_hidden = max(8, int(c2 * gate_hidden_ratio))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * c2, gate_hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)

        # 输出再稳定一下
        self.out = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def _upsample(self, x):
        if self.scale == 1:
            return x
        if self.mode in ("nearest", "area"):
            return F.interpolate(x, scale_factor=self.scale, mode=self.mode)
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners)

    def forward(self, x):
        x = self.pre(x)

        # A path
        A = self._upsample(x)
        A = self.refineA(A)

        # B path
        B = self._upsample(x)
        B = self.ssmB(B)

        # gate fusion
        g = self.gate(torch.cat([A, B], dim=1))
        y = A + g * B
        return self.out(y)
