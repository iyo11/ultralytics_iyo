import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAxisAttention(nn.Module):
    """
    适配 Ultralytics YOLOv11 的交叉轴注意力模块。
    通过 c1, c2 接口确保与 parse_model 兼容。
    """

    def __init__(self, c1, c2, num_heads=8):
        super().__init__()
        # YOLO 默认逻辑：如果 c1 != c2，通常需要一个 1x1 Conv 调整通道
        # 但注意力模块通常用于 c1 == c2 的场景
        self.conv_adjust = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.num_heads = num_heads
        self.dim = c2  # 使用输出通道数作为内部处理维度

        # 对应图 4 的 Norm 层
        self.norm = nn.LayerNorm(self.dim)

        # 投影层
        self.project_in = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.project_out = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        # 多尺度条形卷积 (使用 c2/self.dim 作为 groups 确保深度可分离)
        self.conv_x7 = nn.Conv2d(self.dim, self.dim, (1, 7), padding=(0, 3), groups=self.dim)
        self.conv_x11 = nn.Conv2d(self.dim, self.dim, (1, 11), padding=(0, 5), groups=self.dim)
        self.conv_x21 = nn.Conv2d(self.dim, self.dim, (1, 21), padding=(0, 10), groups=self.dim)

        self.conv_y7 = nn.Conv2d(self.dim, self.dim, (7, 1), padding=(3, 0), groups=self.dim)
        self.conv_y11 = nn.Conv2d(self.dim, self.dim, (11, 1), padding=(5, 0), groups=self.dim)
        self.conv_y21 = nn.Conv2d(self.dim, self.dim, (21, 1), padding=(10, 0), groups=self.dim)

    def forward(self, x):
        # 1. 调整通道并保存残差
        x = self.conv_adjust(x)
        shortcut = x

        b, c, h, w = x.shape

        # 2. 归一化处理
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)

        # 3. 多尺度特征提取 (Fx, Fy)
        fx = self.project_in(self.conv_x7(x_norm) + self.conv_x11(x_norm) + self.conv_x21(x_norm))
        fy = self.project_in(self.conv_y7(x_norm) + self.conv_y11(x_norm) + self.conv_y21(x_norm))

        # 4. Cross-Axis Attention 逻辑
        # X-path
        q1 = rearrange(fy, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k1 = rearrange(fx, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(fx, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1, k1 = F.normalize(q1, dim=-1), F.normalize(k1, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1)).softmax(dim=-1)
        out_x = rearrange((attn1 @ v1) + q1, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Y-path
        q2 = rearrange(fx, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        k2 = rearrange(fy, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(fy, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2, k2 = F.normalize(q2, dim=-1), F.normalize(k2, dim=-1)
        attn2 = (q2 @ k2.transpose(-2, -1)).softmax(dim=-1)
        out_y = rearrange((attn2 @ v2) + q2, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 5. 残差输出
        return self.project_out(out_x) + self.project_out(out_y) + shortcut