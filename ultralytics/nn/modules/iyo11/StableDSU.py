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


import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1) Reproducibility helper
# -------------------------
def set_deterministic(seed: int = 0, deterministic: bool = True) -> None:
    """
    Make results reproducible (as much as PyTorch allows).

    Call this ONCE at program start (before model init / dataloader workers).
    For dataloader workers, also set worker_init_fn separately if needed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN / cuBLAS determinism settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For CUDA matmul determinism (needed on some setups)
        # Choose one of these (":4096:8" is more compatible but uses more workspace).
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Force deterministic algorithms; will error if an op is nondeterministic
        torch.use_deterministic_algorithms(True)

        # Optional: avoid TF32 differences (A100/30xx etc.)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)


# -------------------------
# 2) Lightweight ECA module
# -------------------------
class ECA(nn.Module):
    """Efficient Channel Attention (very light)."""
    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)    # (B, 1, C)
        y = self.conv(y)                     # (B, 1, C)
        y = self.sig(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


# -------------------------
# 3) Enhanced StableDSU
# -------------------------
class StableDSU(nn.Module):
    """
    Enhanced StableDSU for small/mid objects:
    - Semantic upsample: PixelShuffle (sharper than bilinear) [recommended for P3->P2]
    - Detail gate: high-frequency residual (x - blur(x)) + ECA + tanh (centered)
    - gamma (zero-init) for stable start (near identity)
    - GroupNorm for small-batch stability
    - Refine: DWConv + PWConv (spatial + channel mix)

    Args:
        c1: input channels
        c2: output channels
        scale: upsample factor (usually 2)
        gn_groups: group count for GroupNorm (auto-adjusted to divisor of c1)
        eca_k: kernel size for ECA (3 or 5)
        upsample_mode: "pixelshuffle" (default) or "bilinear"
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        scale: int = 2,
        gn_groups: int = 32,
        eca_k: int = 3,
        upsample_mode: str = "pixelshuffle",
    ):
        super().__init__()
        assert scale in (2, 4), "scale usually 2 (or 4). For other scales, extend PixelShuffle logic."
        assert upsample_mode in ("pixelshuffle", "bilinear"), "upsample_mode must be 'pixelshuffle' or 'bilinear'."
        self.scale = scale
        self.upsample_mode = upsample_mode

        # -------------------------
        # Semantic path (enhanced upsample)
        # -------------------------
        if upsample_mode == "pixelshuffle":
            # Conv -> PixelShuffle -> light smooth
            self.semantic_path = nn.Sequential(
                nn.Conv2d(c1, c2 * (scale * scale), kernel_size=1, bias=False),
                nn.PixelShuffle(scale),
                nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False),
                nn.SiLU(),
            )
        else:
            # Bilinear (more stable/cheap) + 1x1 + light smooth
            self.semantic_path = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False),
                nn.SiLU(),
            )

        # -------------------------
        # GroupNorm (auto valid groups)
        # -------------------------
        g = min(gn_groups, c1)
        while c1 % g != 0 and g > 1:
            g -= 1
        self.gate_norm = nn.GroupNorm(g, c1)

        # -------------------------
        # High-frequency gate
        # -------------------------
        self.blur = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)

        self.gate_dw = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)
        self.gate_act = nn.SiLU()
        self.gate_pw = nn.Conv2d(c1, c2, kernel_size=1, bias=True)
        self.eca = ECA(c2, k=eca_k)

        if upsample_mode == "pixelshuffle":
            # For gate upsample, bilinear is fine (gate is a mask-like modulation)
            self.gate_up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            self.gate_up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

        self.gate_out = nn.Tanh()

        # gamma starts at 0 => initially behaves like base_feat (stable)
        self.gamma = nn.Parameter(torch.zeros(1))

        # -------------------------
        # Refine (keep; if you want more speed, replace with nn.Conv2d(c2,c2,1) or nn.Identity)
        # -------------------------
        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_feat = self.semantic_path(x)  # (B, c2, H*scale, W*scale)

        # High-frequency residual helps focus on small/mid edges/textures
        high = x - self.blur(x)

        gate = self.gate_dw(high)
        gate = self.gate_norm(gate)
        gate = self.gate_act(gate)
        gate = self.gate_pw(gate)      # (B, c2, H, W)
        gate = self.eca(gate)
        gate = self.gate_up(gate)      # (B, c2, H*scale, W*scale)
        gate = self.gate_out(gate)     # [-1, 1]

        out = base_feat * (1.0 + self.gamma * gate)
        return self.refine(out)


# -------------------------
# 4) Minimal determinism test
# -------------------------
if __name__ == "__main__":
    set_deterministic(123, deterministic=True)

    m = StableDSU(64, 64, scale=2, upsample_mode="pixelshuffle").cuda().eval()
    x = torch.randn(1, 64, 80, 80, device="cuda")

    with torch.inference_mode():
        y1 = m(x)
        y2 = m(x)

    print("max abs diff:", (y1 - y2).abs().max().item())
