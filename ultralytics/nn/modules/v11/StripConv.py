import torch
import torch.nn as nn
from timm.models.layers import DropPath
from ultralytics.nn.modules import C3, C2f

__all__ = ['StripConvC3k2', 'DSC3k2_StripBlock']


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StripConv(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2), stride=1, padding=(k1 // 2, k2 // 2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1), stride=1, padding=(k2 // 2, k1 // 2), groups=dim)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)

        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model, k1, k2):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripConv(d_model, k1, k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class StripBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k1=1, k2=19, drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


# YOLOv13
class DSC3k_StripBlock(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                StripBlock(c_)
                for _ in range(n)
            )
        )


class DSC3k2_StripBlock(C2f):

    def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k_StripBlock(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k1=k1, k2=k2, d2=d2) for _ in
                range(n))
        else:
            self.m = nn.ModuleList(StripBlock(self.c) for _ in range(n))


class DSBottleneck_StripConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1, den=None):
        super().__init__()
        c_ = int(c2 * e)
        # self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        # self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.cv1 = StripConv(c1, 1, 19)  # 在这里可以替换一种DSConv，也可以都替换。自己可以做一下消融实验
        self.cv2 = StripConv(c_, 1, 19)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k_StripConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1, den=None):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                DSBottleneck_StripConv(
                    c_, c_,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2,
                    den=den
                )
                for _ in range(n)
            )
        )


# class DSC3k2(C2f):
class StripConvC3k2(C2f):
    def __init__(self, c1, c2, n=1, den=None, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k_StripConv(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k1=k1, k2=k2, d2=d2, den=den) for _
                in range(n))
        else:
            self.m = nn.ModuleList(
                DSBottleneck_StripConv(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2, den=den) for _ in
                range(n))