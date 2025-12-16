import torch
from timm.models.layers import DropPath
from torch import nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3, C2f


__all__ = {
    'Partial_conv3',
    'Faster_Block',
    'C2f_Faster',
    'C3_Faster',
    'C2f_Faster_GELUv2',
    'C3_Faster_GELUv2',
}

######################################## C2f-Faster begin ########################################



class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

#
# class Faster_Block(nn.Module):
#     def __init__(self,
#                  inc,
#                  dim,
#                  n_div=4,
#                  mlp_ratio=2,
#                  drop_path=0.1,
#                  layer_scale_init_value=0.0,
#                  pconv_fw_type='split_cat'
#                  ):
#         super().__init__()
#         self.dim = dim
#         self.mlp_ratio = mlp_ratio
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.n_div = n_div
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         mlp_layer = [
#             Conv(dim, mlp_hidden_dim, 1),
#             nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
#         ]
#
#         self.mlp = nn.Sequential(*mlp_layer)
#
#         self.spatial_mixing = Partial_conv3(
#             dim,
#             n_div,
#             pconv_fw_type
#         )
#
#         self.adjust_channel = None
#         if inc != dim:
#             self.adjust_channel = Conv(inc, dim, 1)
#
#         if layer_scale_init_value > 0:
#             self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             self.forward = self.forward_layer_scale
#         else:
#             self.forward = self.forward
#
#     def forward(self, x):
#         if self.adjust_channel is not None:
#             x = self.adjust_channel(x)
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(self.mlp(x))
#         return x
#
#     def forward_layer_scale(self, x):
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(
#             self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
#         return x

class Faster_Block(nn.Module):
    def __init__(
            self,
            inc,
            dim,
            n_div=4,
            mlp_ratio=2,
            drop_path=0.1,
            layer_scale_init_value=0.0,
            pconv_fw_type='split_cat',
            act='silu'  # ★ 新增 act 参数
    ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        # 激活函数选择
        self.act = self.get_act(act)

        mlp_hidden_dim = int(dim * mlp_ratio)

        # MLP 结构加入激活函数
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1, act=act),  # Conv 自带 act
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1, act=act)

        # layer scale 情况
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)),
                requires_grad=True
            )
            self.forward = self.forward_layer_scale

    def get_act(self, act):
        """统一激活函数接口"""
        act = act.lower()
        if act == 'relu':
            return nn.ReLU(inplace=True)
        if act == 'gelu':
            return nn.GELU()
        if act == 'geluv2':
            return nn.GELU()  # ★ 若你有自定义 GELUv2，可替换
        if act == 'lrelu':
            return nn.LeakyReLU(0.1, inplace=True)
        return nn.SiLU(inplace=True)

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x

        x = self.spatial_mixing(x)
        x = self.drop_path(self.mlp(x))

        return shortcut + x

    def forward_layer_scale(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x

        x = self.spatial_mixing(x)
        x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        x = self.drop_path(x)

        return shortcut + x


class C3_Faster(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Faster_Block(c_, c_) for _ in range(n))
        )




class C2f_Faster(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            Faster_Block(self.c, self.c)
            for _ in range(n)
        )


class C3_Faster_GELUv2(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Faster_Block(c_, c_,act='geluv2') for _ in range(n))
        )




class C2f_Faster_GELUv2(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            Faster_Block(self.c, self.c,act='geluv2')
            for _ in range(n)
        )



######################################## C2f-Faster end ########################################