import torch
from timm.models.layers import trunc_normal_
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f, C3

# 声明导出的类，确保 model 解析器能找到
__all__ = ['C3k2_LGFB', 'LGFB']


class PatchSA(nn.Module):
    def __init__(self, dim, heads, patch_size, stride):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.patch_size = patch_size
        self.stride = stride

        # [优化修改]
        # 原代码输入 dim*3, groups=dim*3 (需要先复制数据)
        # 现代码输入 dim, 输出 dim*3, groups=dim (直接生成 QKV，无需复制内存)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, groups=dim, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

        # 位置编码初始化
        self.pos_encode = nn.Parameter(torch.zeros((2 * patch_size - 1) ** 2, heads))
        trunc_normal_(self.pos_encode, std=0.02)
        coord = torch.arange(patch_size)
        coords = torch.stack(torch.meshgrid([coord, coord], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += patch_size - 1
        relative_coords[:, :, 1] += patch_size - 1
        relative_coords[:, :, 0] *= 2 * patch_size - 1
        pos_index = relative_coords.sum(-1)
        self.register_buffer('pos_index', pos_index)

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保宽高一致（根据原逻辑）
        assert H == W, f"PatchSA currently requires H==W, got {H}x{W}"

        pad_num = self.patch_size - self.stride
        patch_num = ((H + pad_num - self.patch_size) // self.stride + 1) ** 2

        # Padding
        expan_x = F.pad(x, (0, pad_num, 0, pad_num), mode='replicate')

        # [优化修改]
        # 删除了 repeat_x = [expan_x] * 3 和 torch.cat，直接计算
        qkv = self.to_qkv(expan_x)

        # Unfold (Im2Col) 提取 patches
        # 注意：此处显存占用依然较大，如果依然 OOM，需减少 patch_size 或 batch_size
        qkv_patches = F.unfold(qkv, kernel_size=self.patch_size, stride=self.stride)
        qkv_patches = qkv_patches.view(B, 3, self.heads, -1, self.patch_size ** 2, patch_num).permute(1, 0, 2, 5, 4, 3)
        q, k, v = qkv_patches[0], qkv_patches[1], qkv_patches[2]

        # Attention 计算
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 加入相对位置编码
        pos_encode = self.pos_encode[self.pos_index.view(-1)].view(self.patch_size ** 2, self.patch_size ** 2, -1)
        pos_encode = pos_encode.permute(2, 0, 1).contiguous().unsqueeze(1).repeat(1, patch_num, 1, 1)
        attn = attn + pos_encode.unsqueeze(0)

        attn = self.softmax(attn)
        _res = (attn @ v)

        # 重组回特征图
        _res = _res.view(B, self.heads, patch_num, self.patch_size, self.patch_size, -1)[:, :, :, :self.stride,
               :self.stride]
        _res = _res.transpose(2, 5).contiguous().view(B, -1, patch_num)
        res = F.fold(_res, output_size=(H, W), kernel_size=self.stride, stride=self.stride)

        return self.to_out(res)


class EfficientGlobalSA(nn.Module):
    def __init__(self, dim, heads, reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.rd = reduction_ratio

        self.to_q = nn.Conv2d(dim, dim, 1, bias=True, groups=dim)
        self.to_k = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=True, groups=dim)
        self.to_v = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=True, groups=dim)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保能被 reduction_ratio 整除
        assert (H == W and (
                    W % self.rd == 0)), f"EfficientGlobalSA input size {H}x{W} mismatch with reduction {self.rd}"

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        _q = q.reshape(B, self.heads, -1, H * W).transpose(-2, -1)
        _k = k.reshape(B, self.heads, -1, (H // self.rd) ** 2)
        _v = v.reshape(B, self.heads, -1, (H // self.rd) ** 2).transpose(-2, -1)
        attn = (_q @ _k) * self.scale

        attn = self.softmax(attn)
        res = (attn @ _v)
        res = res.transpose(-2, -1).reshape(B, -1, H, W)
        return self.to_out(res)


class SALayer(nn.Module):
    def __init__(self, channel, patch_size=8, stride=4, heads=4, dim_ratio=4, reduction_ratio=None):
        super(SALayer, self).__init__()
        # 只有当 reduction_ratio 存在时才使用 GlobalSA，否则使用 PatchSA
        if reduction_ratio:
            self.sa = EfficientGlobalSA(channel, heads, reduction_ratio)
        else:
            self.sa = PatchSA(channel, heads, patch_size, stride)

        hidden_dim = int(channel * dim_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channel, 1)
        )
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = self.sa(self.bn1(x)) + x
        x = self.mlp(self.bn2(x)) + x
        return x


class LGFB(nn.Module):
    def __init__(self, channel):
        super(LGFB, self).__init__()
        # 根据经验，Local Attention 显存消耗最大，如仍 OOM 可尝试调大 stride (如 stride=4 -> stride=8)
        self.Local = SALayer(channel=channel, reduction_ratio=None, patch_size=8, stride=4)
        self.GLobal = SALayer(channel=channel, reduction_ratio=4)

    def forward(self, x):
        # 先 Global 后 Local (或者反过来，原论文顺序为先 Local 后 Global，这里保持原代码逻辑)
        return self.GLobal(self.Local(x))


# --------------------------------------------------------------------------
# YOLOv11 Adapter Classes
# --------------------------------------------------------------------------

class C3k_LGFB(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # 使用 LGFB 替换原来的 Bottleneck
        self.m = nn.Sequential(*(LGFB(c_) for _ in range(n)))


class C3k2_LGFB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_LGFB(self.c, self.c, 2, shortcut, g) if c3k else LGFB(self.c) for _ in range(n)
        )