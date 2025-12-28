import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PoolFormerMSConvBlock']


# -------------------------
# 1. 改进的归一化层 (适配 NCHW)
# -------------------------
class ModifiedLayerNorm(nn.GroupNorm):
    def __init__(self, dim, eps=1e-5):
        super().__init__(1, dim, eps=eps)


# -------------------------
# 2. 改进的自适应残差池化 (针对小目标保留原始像素)
# -------------------------
class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
        # 可学习参数：控制局部对比度提取的强度，初始化为 0 表示起始状态为 Identity
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 结果 = 原始特征 + 缩放后的高通滤波特征
        return x + self.beta * (self.pool(x) - x)


# -------------------------
# 3. 随机深度 (Stochastic Depth)
# -------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask


# -------------------------
# 4. 改进的多尺度卷积 (引入 Dilated Conv 保持锐度)
# -------------------------
class MSDWConv(nn.Module):
    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()

        # 处理 YAML 传参兼容性
        if isinstance(dw_sizes, (int, float)):
            dw_sizes = (int(dw_sizes),)
        elif isinstance(dw_sizes, str):
            dw_sizes = tuple(int(s.strip()) for s in dw_sizes.split(",") if s.strip())

        self.dw_sizes = list(dw_sizes) if dw_sizes else [1, 3, 5, 7]
        self.channels = []
        self.proj = nn.ModuleList()
        n = len(self.dw_sizes)

        for i, k in enumerate(self.dw_sizes):
            ch = dim - (dim // n) * (n - 1) if i == 0 else (dim // n)
            self.channels.append(ch)

            if k <= 3:
                # 1x1 和 3x3 使用普通深度可分离卷积
                self.proj.append(nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=True))
            else:
                # 5x5, 7x7 使用 3x3 空洞卷积替代，减少参数平滑，增强对孤立像素（小目标）的响应
                dilation = (k - 1) // 2
                self.proj.append(nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, groups=ch, bias=True))

    def forward(self, x):
        xs = torch.split(x, self.channels, dim=1)
        ys = [conv(t) for conv, t in zip(self.proj, xs)]
        return torch.cat(ys, dim=1)


# -------------------------
# 5. 增强型 MSConvStar (引入空间注意力)
# -------------------------
class MSConvStar(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        hidden = int(dim * float(mlp_ratio))
        if hidden % 2 != 0:
            hidden += 1

        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dw = MSDWConv(hidden, dw_sizes=dw_sizes)
        self.act = nn.GELU()

        # 空间门控：学习哪些像素点是重要的（小目标所在区域）
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden // 2, 1, 1),
            nn.Sigmoid()
        )

        self.fc2 = nn.Conv2d(hidden // 2, dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x + self.dw(x)
        x1, x2 = x.chunk(2, dim=1)

        # 门控交互
        x_gated = self.act(x1) * x2
        # 应用空间注意力增强
        x_gated = x_gated * self.spatial_gate(x_gated)

        return self.fc2(x_gated)


# -------------------------
# 6. 核心 Block
# -------------------------
class PoolFormerMSConvBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=2.0, drop_path=0.0, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.norm1 = ModifiedLayerNorm(dim)
        self.token_mixer = Pooling(pool_size)

        self.norm2 = ModifiedLayerNorm(dim)
        self.mlp = MSConvStar(dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)

        self.drop_path = DropPath(drop_path) if float(drop_path) > 0.0 else nn.Identity()

    def forward(self, x):
        # 第一阶段：局部空间聚合
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        # 第二阶段：多尺度通道与空间动态交互
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x