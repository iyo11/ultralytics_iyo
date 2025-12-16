import torch
import torch.nn as nn


class InceptionDWConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        square_kernel_size=3,
        band_kernel_size=11,
        branch_ratio=0.125,
    ):
        super().__init__()

        # 根据 branch_ratio 计算分支通道数
        gc = int(in_channels * branch_ratio)

        # 方形深度可分离卷积
        self.dwconv_hw = nn.Conv2d(
            gc,
            gc,
            kernel_size=square_kernel_size,
            padding=square_kernel_size // 2,
            groups=gc,
        )

        # 水平方向条形深度可分离卷积
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )

        # 垂直方向条形深度可分离卷积
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )

        # 通道拆分索引
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        """
        输入:
            x: [B, C, H, W]

        拆分后:
            x_id: 身份映射
            x_hw: 方形卷积
            x_w : 水平条形卷积
            x_h : 垂直条形卷积
        """
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)

        return torch.cat(
            (
                x_id,
                self.dwconv_hw(x_hw),
                self.dwconv_w(x_w),
                self.dwconv_h(x_h),
            ),
            dim=1,
        )


if __name__ == "__main__":
    model = InceptionDWConv2d(in_channels=64)
    input_tensor = torch.randn(1, 64, 224, 224)
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
