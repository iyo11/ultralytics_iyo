import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PFMM']


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.LeakyReLU) or isinstance(m,
                                                                                                           nn.AdaptiveAvgPool2d) or isinstance(
            m, nn.AdaptiveMaxPool2d) or isinstance(m, nn.ReLU6) or isinstance(m, nn.MaxPool2d) or isinstance(m,
                                                                                                             nn.Softmax) or isinstance(
            m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        else:
            m.initialize()


class CoordAtt(nn.Module):

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [B, C, 1, W]
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = identity * a_h * a_w
        return out

    def initialize(self):
        weight_init(self)


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=1):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        return f_fg, f_bg


class PFMM(nn.Module):
    def __init__(self, channels, channels2, alpha_init=0.1, reduction=32):
        super().__init__()
        self.coord_att = CoordAtt(channels, reduction)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.fg_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bg_weight = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1)
        self.DecoupleLayer = DecoupleLayer(channels2)

    def forward(self, data):
        Ft, fg = data
        P_fg, P_bg = self.DecoupleLayer(fg)
        if P_fg.size()[2:] != Ft.size()[2:]:
            P_fg = F.interpolate(P_fg, size=Ft.size()[2:], mode='bilinear', align_corners=False)
        if P_bg.size()[2:] != Ft.size()[2:]:
            P_bg = F.interpolate(P_bg, size=Ft.size()[2:], mode='bilinear', align_corners=False)

        epsilon = 1e-6
        prob_fg = (self.fg_weight * P_fg) / (self.fg_weight * P_fg + self.bg_weight * P_bg + epsilon)  # [B, 1, H, W]
        gate = self.gate_conv(prob_fg)  # [B, C, H, W]
        F_mod = Ft + self.alpha * (Ft * gate)
        F_fuse = self.fusion_conv(torch.cat([Ft, F_mod], dim=1))
        F_ca = self.coord_att(F_fuse)
        return F_ca

    def initialize(self):
        weight_init(self)