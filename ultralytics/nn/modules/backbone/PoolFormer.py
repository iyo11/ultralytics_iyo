import torch
import torch.nn as nn

__all__ = ['Pooling', 'Mlp', 'ModifiedLayerNorm', 'PoolFormerBlock']

class Pooling(nn.Module):
    # PoolFormer token mixer: AvgPool(stride=1) - x
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2,
                                 count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class Mlp(nn.Module):
    # Channel MLP: 1x1 conv -> act -> 1x1 conv
    def __init__(self, dim, mlp_ratio=4.0, act=nn.GELU, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = act()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ModifiedLayerNorm(nn.GroupNorm):
    # MLN = GroupNorm(1, C) in channel-first [B,C,H,W]
    def __init__(self, dim, eps=1e-5):
        super().__init__(1, dim, eps=eps)

class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = ModifiedLayerNorm(dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = ModifiedLayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
