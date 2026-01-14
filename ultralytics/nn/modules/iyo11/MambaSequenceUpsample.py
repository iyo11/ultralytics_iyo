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
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = None,
        ssm_ratio: float = 2.0,
        ssm_d_state: int = 16,
        ssm_conv: int = 3,
        ssm_drop_rate: float = 0.0,
        mlp_ratio: float = 0.0,
        mlp_drop_rate: float = 0.0
    ):
        super().__init__()
        c = in_channels if hidden_dim is None else hidden_dim
        assert c == in_channels, "简易版建议 hidden_dim == in_channels"

        self.norm = nn.GroupNorm(num_groups=min(32, c), num_channels=c)

        self.in_proj = nn.Conv2d(c, 2 * c, kernel_size=1, bias=True)

        self.local_dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)

        k = int(ssm_conv)
        if k % 2 == 0:
            k += 1
        pad = k // 2

        self.h_conv = nn.Conv1d(c, c, kernel_size=k, padding=pad, groups=c, bias=False)
        self.w_conv = nn.Conv1d(c, c, kernel_size=k, padding=pad, groups=c, bias=False)

        # 关键：不要 inplace
        self.act = nn.SiLU(inplace=False)
        self.drop = nn.Dropout(ssm_drop_rate) if ssm_drop_rate > 0 else nn.Identity()

        self.out_proj = nn.Conv2d(c, c, kernel_size=1, bias=True)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _scan_h(self, x):
        b, c, h, w = x


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
