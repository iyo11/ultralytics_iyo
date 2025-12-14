import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "MFAM",
    "ECA",
    "PConv",
    "PCMFAM"
)

import torch
import torch.nn as nn
import torch.nn.functional as F


def dwconv(ch, k, stride=1, padding=0, bias=False):
    return nn.Conv2d(ch, ch, k, stride=stride, padding=padding, groups=ch, bias=bias)


def get_norm(norm, ch, gn_groups=8):
    norm = (norm or "bn").lower()
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "gn":
        # groups 不要超过通道数，且要整除
        g = min(gn_groups, ch)
        while ch % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, ch)
    if norm in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown norm: {norm}")


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act=True, norm="bn", gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = get_norm(norm, out_ch, gn_groups)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ECA(nn.Module):
    """ECA: 超轻量通道注意力"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg(x)                     # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)   # (B,1,C)
        y = torch.sigmoid(self.conv(y))     # (B,1,C)
        y = y.transpose(1, 2).unsqueeze(-1) # (B,C,1,1)
        return x * y


class PConv(nn.Module):
    """
    Partial Convolution：只对部分通道做 3x3 Conv，其余通道直通
    改动：默认 ratio 提高（你也可传 1.0 全卷积）
    """
    def __init__(self, channels, ratio=1.0, k=3, act=True, norm="bn", gn_groups=8):
        super().__init__()
        c_conv = max(1, int(channels * ratio))
        c_id = channels - c_conv
        self.c_conv = c_conv
        self.c_id = c_id

        self.conv = ConvNormAct(c_conv, c_conv, k=k, s=1, p=k//2, act=act, norm=norm, gn_groups=gn_groups)

    def forward(self, x):
        if self.c_id <= 0:
            return self.conv(x)
        x1, x2 = torch.split(x, [self.c_conv, self.c_id], dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


class MFAM(nn.Module):
    """
    MFAM 改进点：
    1) 分支仍用 depthwise 做空间建模（省算力）
    2) 每个分支后加一个 1x1（pointwise）做通道混合/对齐，再相加（避免“各玩各的+分布不齐”）
    3) residual 的 id 分支使用 adaptive pool 对齐尺寸（保持你原来的稳定性）
    """
    def __init__(self, channels, stride=1, out_channels=None, act=True, norm="bn", gn_groups=8):
        super().__init__()
        out_channels = out_channels if out_channels is not None else channels
        self.stride = stride

        def dw_block(k, padding, stride_):
            return nn.Sequential(
                dwconv(channels, k, stride=stride_, padding=padding),
                get_norm(norm, channels, gn_groups),
                nn.SiLU(inplace=True) if act else nn.Identity(),
                # ✅ 关键：分支内部做一次通道混合与分布对齐
                ConvNormAct(channels, channels, k=1, s=1, p=0, act=act, norm=norm, gn_groups=gn_groups),
            )

        self.b0 = dw_block(3, 1, stride)
        self.b1 = dw_block(5, 2, stride)

        self.b2 = nn.Sequential(
            dwconv(channels, (1, 7), stride=stride, padding=(0, 3)),
            get_norm(norm, channels, gn_groups),
            nn.SiLU(inplace=True) if act else nn.Identity(),
            dwconv(channels, (7, 1), stride=1, padding=(3, 0)),
            get_norm(norm, channels, gn_groups),
            nn.SiLU(inplace=True) if act else nn.Identity(),
            ConvNormAct(channels, channels, k=1, s=1, p=0, act=act, norm=norm, gn_groups=gn_groups),
        )

        self.b3 = nn.Sequential(
            dwconv(channels, (1, 9), stride=stride, padding=(0, 4)),
            get_norm(norm, channels, gn_groups),
            nn.SiLU(inplace=True) if act else nn.Identity(),
            dwconv(channels, (9, 1), stride=1, padding=(4, 0)),
            get_norm(norm, channels, gn_groups),
            nn.SiLU(inplace=True) if act else nn.Identity(),
            ConvNormAct(channels, channels, k=1, s=1, p=0, act=act, norm=norm, gn_groups=gn_groups),
        )

        self.fuse = ConvNormAct(channels, out_channels, k=1, s=1, p=0, act=act, norm=norm, gn_groups=gn_groups)

    def _id_align(self, x, y_ref):
        if x.shape[-2:] == y_ref.shape[-2:]:
            return x
        return F.adaptive_avg_pool2d(x, output_size=y_ref.shape[-2:])

    def forward(self, x):
        y0 = self.b0(x)
        y = y0 + self.b1(x) + self.b2(x) + self.b3(x) + self._id_align(x, y0)
        return self.fuse(y)


class PCMFAM(nn.Module):
    """
    PCMFAM 改进点（对应你 1~5 点）：
    1) 默认 split_ratio 更大（更多通道走卷积分支）
    2) 默认 pconv_ratio 更大（甚至 1.0 全部卷）
    3) 注意力分支：先 1x1 再 ECA（先产生一点新组合再 re-weight）
    4) MFAM 升级为 MFAMv2（分支相加前做通道混合/对齐）
    5) 残差：x + gamma*y（gamma 初值 0，更稳、更容易学到“何时用模块”）
    """
    def __init__(self, in_ch, out_ch=None,
                 split_ratio=0.75,          # ✅ 默认从 0.5 -> 0.75
                 pconv_ratio=1.0,           # ✅ 默认从 0.5 -> 1.0
                 attn='eca',
                 use_residual=True,
                 act=True,
                 norm="bn",                 # 小 batch 可改 "gn"
                 gn_groups=8):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch

        c_conv = max(1, int(in_ch * split_ratio))
        c_att = in_ch - c_conv
        self.c_conv = c_conv
        self.c_att = c_att
        self.use_residual = use_residual and (out_ch == in_ch)

        # 卷积分支：PConv + MFAMv2
        self.pconv = PConv(c_conv, ratio=pconv_ratio, k=3, act=act, norm=norm, gn_groups=gn_groups)
        self.mfam = MFAM(c_conv, stride=1, out_channels=c_conv, act=act, norm=norm, gn_groups=gn_groups)

        # 注意力分支：✅ 1x1 -> ECA
        if c_att > 0 and attn.lower() == "eca":
            self.attn_pre = ConvNormAct(c_att, c_att, k=1, s=1, p=0, act=act, norm=norm, gn_groups=gn_groups)
            self.attn = ECA(c_att, k_size=3)
        else:
            self.attn_pre = nn.Identity()
            self.attn = nn.Identity()

        # 融合
        self.fuse = ConvNormAct(in_ch, out_ch, k=1, s=1, p=0, act=act, norm=norm, gn_groups=gn_groups)

        # ✅ 残差缩放系数（初值 0：一开始像 identity，训练更稳）
        self.gamma = nn.Parameter(torch.zeros(1)) if self.use_residual else None

    def forward(self, x):
        if self.c_att == 0:
            y = self.mfam(self.pconv(x))
            y = self.fuse(y)
            if self.use_residual:
                return x + self.gamma * y
            return y

        x_conv, x_att = torch.split(x, [self.c_conv, self.c_att], dim=1)

        y_conv = self.mfam(self.pconv(x_conv))
        y_att = self.attn(self.attn_pre(x_att))

        y = torch.cat([y_conv, y_att], dim=1)
        y = self.fuse(y)

        if self.use_residual:
            return x + self.gamma * y
        return y


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ====== 1) 测 PCMFAM（最常用）======
    B, C, H, W = 2, 64, 80, 80
    x = torch.randn(B, C, H, W, device=device)

    print("\n=== Test: PCMFAM (residual on, out_ch=in_ch) ===")
    m1 = PCMFAM(in_ch=C, out_ch=C, split_ratio=0.5, pconv_ratio=0.5, attn="eca", use_residual=True).to(device)
    y1 = m1(x)
    print("Input :", tuple(x.shape))
    print("Output:", tuple(y1.shape))
    assert y1.shape == x.shape, "PCMFAM residual版：输出shape应与输入一致"

    loss = y1.mean()
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in m1.parameters())
    print("Backward OK:", grad_ok)

    # ====== 2) 测 PCMFAM（改变 out_ch，自动关闭 residual）======
    print("\n=== Test: PCMFAM (out_ch != in_ch, residual off automatically) ===")
    out_ch = 96
    m2 = PCMFAM(in_ch=C, out_ch=out_ch, split_ratio=0.5, pconv_ratio=0.5, attn="eca", use_residual=True).to(device)
    y2 = m2(x)
    print("Output:", tuple(y2.shape))
    assert y2.shape == (B, out_ch, H, W), "PCMFAM改通道数：输出通道不匹配"

    # ====== 3) 测 PCMFAM（split_ratio=1.0，纯卷积分支；c_att=0 分支覆盖）======
    print("\n=== Test: PCMFAM (split_ratio=1.0, conv-only path) ===")
    m3 = PCMFAM(in_ch=C, out_ch=C, split_ratio=1.0, pconv_ratio=0.5, attn="eca", use_residual=True).to(device)
    y3 = m3(x)
    print("Output:", tuple(y3.shape))
    assert y3.shape == x.shape, "纯卷积分支：输出shape应与输入一致"

    # ====== 4) 单测 MFAM stride=2（空间降采样）======
    print("\n=== Test: MFAM stride=2 (downsample) ===")
    mfam = MFAM(channels=C, stride=2, out_channels=C).to(device)
    y4 = mfam(x)
    print("Output:", tuple(y4.shape))
    assert y4.shape == (B, C, H // 2, W // 2), "MFAM stride=2：输出分辨率不对"

    print("\nAll tests passed ✅")

if __name__ == "__main__":
    main()

