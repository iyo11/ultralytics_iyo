#################################################################################################
##E0
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#
#         self.semantic_path = nn.Sequential(
#             nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
#             nn.Conv2d(c1, c2, 1, bias=False)
#         )
#
#         self.detail_gate = nn.Sequential(
#             nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
#             nn.BatchNorm2d(c1),
#             nn.SiLU(),
#             nn.Conv2d(c1, c2, 1, bias=False),
#             nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
#             nn.Sigmoid()
#         )
#
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#     def forward(self, x):
#         base_feat = self.semantic_path(x)
#         detail_mask = self.detail_gate(x)
#         out = base_feat * (1 + detail_mask)
#         return self.refine(out)




#################################################################################################
##E1
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class StableDSU(nn.Module):
#     def __init__(self, c1, c2, scale=2):
#         super().__init__()
#         self.scale = scale
#         # 预先降维，减少后续计算量
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         # 改进的轻量化门控：在低分辨率下完成卷积
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#     def forward(self, x):
#         # 1. 先压缩通道
#         x_low = self.compress(x)
#         # 2. 在低分辨率下生成 mask (省显存/计算量)
#         mask_low = self.gate_conv(x_low)
#         # 3. 应用 mask
#         out_low = x_low * (1 + mask_low)
#         # 4. 统一进行一次上采样
#         out = F.interpolate(out_low, scale_factor=self.scale, mode='bilinear', align_corners=False)
#         # 5. 最后平滑
#         return self.refine(out)



#################################################################################################
##E2
#################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class StableDSU(nn.Module):
#     """
#     Version B (更实用的保守版):
#     - gamma_gate = 0.05 近 0-init：既稳又容易学出收益
#     - gate 用 sigmoid 并中心化：梯度更友好
#     - refine 继续 0-init：避免 RSOD 被细化卷积伤到
#     """
#     def __init__(self, c1, c2, scale=2, gn=False):
#         super().__init__()
#         self.scale = scale
#
#         self.compress = nn.Conv2d(c1, c2, 1, bias=False)
#
#         Norm = (lambda c: nn.GroupNorm(32, c)) if gn else (lambda c: nn.BatchNorm2d(c))
#
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             Norm(c2),
#             nn.SiLU(),
#             nn.Conv2d(c2, c2, 1, bias=True),
#             nn.Sigmoid()
#         )
#
#         self.refine = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
#
#         # 门控强度：给一个很小的初值，避免“学不起来”
#         self.gamma_gate = nn.Parameter(torch.ones(1, c2, 1, 1) * 0.05)
#         # 细化强度：继续 0-init，减少跨数据集副作用
#         self.gamma_ref = nn.Parameter(torch.zeros(1, c2, 1, 1))
#
#     def forward(self, x):
#         x_low = self.compress(x)
#
#         gate = self.gate_conv(x_low)          # (0, 1)
#         gate = gate - 0.5                     # 中心化到 (-0.5, 0.5)
#
#         out_low = x_low * (1.0 + self.gamma_gate * gate)
#
#         out = F.interpolate(out_low, scale_factor=self.scale,
#                             mode='bilinear', align_corners=False)
#
#         out = out + self.gamma_ref * self.refine(out)
#         return out


#################################################################################################
##E3
#################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(c, max_groups=32):
    # 让 Group 数能整除通道数；常见 c2=64/128/256 都没问题
    g = min(max_groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)


class StableDSU(nn.Module):
    """
    StableDSU vBest (small-object & cross-dataset friendly)

    Core ideas:
    - GN instead of BN for small-batch + domain shift stability
    - multi-scale depthwise gating (local + dilated context)
    - strong regularized gate: 1-channel spatial mask (anti-overfit)
    - multiplicative gate in (0,2): y = x * (2*sigmoid(mask_logits))
      -> can both suppress & enhance (unlike 1+sigmoid)
    - refine residual strength controlled by sigmoid(alpha) (stable yet learnable)
    - upsample option: pixelshuffle-lite (default) or bilinear

    Args:
        c1: input channels
        c2: output channels
        scale: upsample factor (2 usually)
        upsample: "ps" (pixelshuffle-lite) or "bilinear"
        gate_channels: 1 for most stable; can set 8/16 for stronger gate if needed
        gn_groups: max GN groups (auto-adjusted to divide channels)
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        scale: int = 2,
        upsample: str = "ps",
        gate_channels: int = 1,
        gn_groups: int = 32,
    ):
        super().__init__()
        assert upsample in ("ps", "bilinear")
        assert scale in (2, 4), "scale usually 2 (or 4)."

        self.scale = scale
        self.upsample_mode = upsample

        # 1) compress + norm + act
        self.compress = nn.Conv2d(c1, c2, 1, bias=False)
        self.norm1 = _make_gn(c2, gn_groups)
        self.act = nn.SiLU()

        # 2) multi-scale depthwise gating trunk
        self.gate_local = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.gate_context = nn.Conv2d(c2, c2, 3, padding=2, dilation=2, groups=c2, bias=False)

        # gate fusion -> produce low-dim spatial mask (default 1 channel)
        # using a small bottleneck to stabilize + avoid overfit
        hidden = max(8, c2 // 8)
        self.gate_fuse = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=False),
            _make_gn(hidden, gn_groups),
            nn.SiLU(),
            nn.Conv2d(hidden, gate_channels, 1, bias=True)  # logits
        )

        # 3) upsample head
        if self.upsample_mode == "ps":
            # pixelshuffle-lite: (c2 -> c2*scale^2) then PixelShuffle
            self.up_proj = nn.Conv2d(c2, c2 * (scale ** 2), 1, bias=False)
            self.up_norm = _make_gn(c2 * (scale ** 2), gn_groups)
            self.up_act = nn.SiLU()
            self.ps = nn.PixelShuffle(scale)
        else:
            self.up_proj = None

        # 4) refine block (DW + PW) with controlled residual strength
        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.Conv2d(c2, c2, 1, bias=False),
            _make_gn(c2, gn_groups),
            nn.SiLU(),
        )
        # alpha init: sigmoid(-2) ~ 0.12, "weak but not zero" at start
        self.alpha = nn.Parameter(torch.tensor(-2.0))

        # ---- init: keep it stable ----
        # make gate logits start near 0 => sigmoid ~ 0.5 => gate ~ 1.0 (neutral)
        nn.init.zeros_(self.gate_fuse[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compress
        x_low = self.act(self.norm1(self.compress(x)))

        # multi-scale gating features
        g = self.gate_local(x_low) + self.gate_context(x_low)
        logits = self.gate_fuse(g)  # (B, gate_channels, H, W)

        # multiplicative gate in (0,2): suppress/enhance
        gate = 2.0 * torch.sigmoid(logits)

        # if gate is low-dim (e.g., 1 channel), broadcast to c2
        if gate.shape[1] != x_low.shape[1]:
            gate = gate.expand(-1, x_low.shape[1], -1, -1)

        out_low = x_low * gate

        # upsample
        if self.upsample_mode == "ps":
            u = self.up_act(self.up_norm(self.up_proj(out_low)))
            out = self.ps(u)
        else:
            out = F.interpolate(out_low, scale_factor=self.scale, mode="bilinear", align_corners=False)

        # controlled residual refine
        w = torch.sigmoid(self.alpha)  # (0,1)
        out = out + w * self.refine(out)
        return out
