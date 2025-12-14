import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "MFAM",
    "ECA",
    "PConv",
    "PCMFAM"
)

def dwconv(ch, k, stride=1, padding=0, bias=False):
    return nn.Conv2d(ch, ch, k, stride=stride, padding=padding, groups=ch, bias=bias)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MFAM(nn.Module):
    """
    改动点：
    - 仍然保留“捷径分支”概念（id）
    - 但不再用 AvgPool2d(kernel_size=stride, stride=stride)
    - 而是把输入 x 自适应池化到与主分支输出同尺寸（永远合法，不会 0x0）
    """
    def __init__(self, channels, stride=1, out_channels=None, act=True):
        super().__init__()
        out_channels = out_channels if out_channels is not None else channels
        self.stride = stride

        self.b0 = nn.Sequential(
            dwconv(channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True) if act else nn.Identity()
        )

        self.b1 = nn.Sequential(
            dwconv(channels, 5, stride=stride, padding=2),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True) if act else nn.Identity()
        )

        self.b2 = nn.Sequential(
            dwconv(channels, (1, 7), stride=stride, padding=(0, 3)),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True) if act else nn.Identity(),
            dwconv(channels, (7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True) if act else nn.Identity()
        )

        self.b3 = nn.Sequential(
            dwconv(channels, (1, 9), stride=stride, padding=(0, 4)),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True) if act else nn.Identity(),
            dwconv(channels, (9, 1), stride=1, padding=(4, 0)),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True) if act else nn.Identity()
        )

        self.fuse = ConvBNAct(channels, out_channels, k=1, s=1, p=0, act=act)

    def _id_align(self, x, y_ref):
        """
        把 x 变成和 y_ref 一样的 (H, W)，用于残差相加。
        这比“kernel=stride 的 AvgPool”更稳定：输入再小也不会崩。
        """
        if x.shape[-2:] == y_ref.shape[-2:]:
            return x
        return F.adaptive_avg_pool2d(x, output_size=y_ref.shape[-2:])

    def forward(self, x):
        y0 = self.b0(x)
        y = y0 + self.b1(x) + self.b2(x) + self.b3(x) + self._id_align(x, y0)
        return self.fuse(y)

class ECA(nn.Module):
    """超轻量通道注意力：ECA"""
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
    """
    def __init__(self, channels, ratio=0.5, k=3, act=True):
        super().__init__()
        c_conv = max(1, int(channels * ratio))
        c_id = channels - c_conv
        self.c_conv = c_conv
        self.c_id = c_id

        self.conv = ConvBNAct(c_conv, c_conv, k=k, s=1, p=k//2, act=act)  # 普通卷积即可

    def forward(self, x):
        x1, x2 = torch.split(x, [self.c_conv, self.c_id], dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


class PCMFAM(nn.Module):
    """
    一半走卷积增强：PConv -> MFAM
    一半走注意力增强：ECA(或换SE/CoordAtt)
    最后 concat -> 1x1 fuse，可选残差
    """
    def __init__(self, in_ch, out_ch=None, split_ratio=0.5,
                 pconv_ratio=0.5, attn='eca', use_residual=True, act=True):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        c_conv = max(1, int(in_ch * split_ratio))
        c_att = in_ch - c_conv
        self.c_conv = c_conv
        self.c_att = c_att
        self.use_residual = use_residual and (out_ch == in_ch)

        # 卷积分支：PConv + MFAM
        self.pconv = PConv(c_conv, ratio=pconv_ratio, k=3, act=act)
        self.mfam = MFAM(c_conv, stride=1, out_channels=c_conv, act=act)

        # 注意力分支：默认 ECA（你也可以换 SE/CBAM/CoordAtt）
        if attn.lower() == 'eca':
            self.attn = ECA(c_att, k_size=3) if c_att > 0 else nn.Identity()
        else:
            # 兜底：不想要注意力就传 attn='none'
            self.attn = nn.Identity()

        # 融合
        self.fuse = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, act=act)

    def forward(self, x):
        if self.c_att == 0:
            y = self.mfam(self.pconv(x))
            y = self.fuse(y)
            return x + y if self.use_residual else y

        x_conv, x_att = torch.split(x, [self.c_conv, self.c_att], dim=1)

        y_conv = self.mfam(self.pconv(x_conv))
        y_att = self.attn(x_att)

        y = torch.cat([y_conv, y_att], dim=1)
        y = self.fuse(y)
        return x + y if self.use_residual else y



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

