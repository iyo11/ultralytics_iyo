import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.iyo11.mamba_yolo import VSSBlock


# 你已经有 VSSBlock（就是你贴的那份实现）了
# 把下面这一行改成你工程里 VSSBlock 的真实导入路径


class HybridSS2DUpsample(nn.Module):
    """
    YOLO/Ultralytics 风格模块：
    - __init__(c1, c2, scale=2, ...)
    - forward(x): [B,c1,H,W] -> [B,c2,H*scale,W*scale]

    结构：
      A: 2D bilinear ↑ -> refine (几何稳定主干)
      B: 2D bilinear ↑ -> VSSBlock(真·2D SSM)
      gate: sigmoid(Conv1x1( concat(A,B) ))，并且 gate 最后一层 0-init，训练更稳
      out: A + gate * B
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        scale: int = 2,
        # ---- B: VSSBlock/SS2D 超参（按需改） ----
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

        # 先把通道对齐到 c2（YOLO neck 常见做法）
        self.pre = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        ) if c1 != c2 else nn.Identity()

        # A: 上采样后 refine（几何稳定）
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

        # B: 上采样后做真·2D SSM
        # hidden_dim=c2，保证输入输出通道一致，方便与A融合
        self.ssmB = VSSBlock(
            in_channels=c2,
            hidden_dim=c2,
            ssm_ratio=ssm_ratio,
            ssm_d_state=ssm_d_state,
            ssm_conv=ssm_conv,
            ssm_drop_rate=ssm_drop_rate,
            mlp_ratio=0.0,       # 只保留SSM分支（更轻更稳）
            mlp_drop_rate=0.0,
        )

        # gate：concat(A,B) -> gate(B)  (0-init，初期≈0 => 输出≈A)
        gate_hidden = max(8, int(c2 * gate_hidden_ratio))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * c2, gate_hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)

        # 可选：再做一次轻量稳定（你觉得多余可删）
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
