import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    右侧分支：通道注意力机制 (保持不变)
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConcatAddFusion(nn.Module):
    """
    对应黄色流程图 (image_7789c6.png)
    也就是绿色图中 'C+' 节点的具体实现

    流程:
    1. Concat(原始特征A, 处理后特征B) -> 2C
    2. Conv 1x1 -> C
    3. Add(结果, 原始特征A)
    """

    def __init__(self, c):
        super().__init__()
        # 输入是 2C (A+B), 输出是 C
        self.conv1x1 = nn.Conv2d(c * 2, c, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)  # 通常Conv后会接激活，图中未明确画出，如不需要可注释掉

    def forward(self, x_orig, x_processed):
        # x_orig: 原始特征 A (图中下方的输入)
        # x_processed: 处理后特征 B (图中上方的输入)

        # 1. Concat
        cat_feat = torch.cat([x_processed, x_orig], dim=1)

        # 2. Conv 1x1
        out = self.conv1x1(cat_feat)
        # out = self.act(out) # 可选: 激活函数

        # 3. Add (Residual connection with Original Feature A)
        return out + x_orig


class BHFM(nn.Module):
    """
    Bio-inspired Hierarchical Feature Modulation (BHFM)
    已更新: 'C+' 节点使用 ConcatAddFusion 模块
    """

    def __init__(self, c1, c2, kernel_size=5, dilation=3):
        super().__init__()
        # 假设 c1 == c2，如果不同建议先在外部用 1x1 统一
        c = c1

        # === 1. 特征提取分支 ===

        # 第一层：5x5 正交卷积
        self.ortho_conv5 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, 5), padding=(0, 2), groups=c),
            nn.Conv2d(c, c, kernel_size=(5, 1), padding=(2, 0), groups=c)
        )
        # 第一个 C+ 融合模块
        self.fusion1 = ConcatAddFusion(c)

        # 第二层：19x19 正交卷积
        self.ortho_conv19 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, 7), padding=(0, 9), dilation=3, groups=c),
            nn.Conv2d(c, c, kernel_size=(7, 1), padding=(9, 0), dilation=3, groups=c)
        )
        # 第二个 C+ 融合模块
        self.fusion2 = ConcatAddFusion(c)

        # === 2. 混合与注意力分支 ===
        self.mid_conv1x1 = nn.Conv2d(c, c, 1)
        self.channel_attn = ChannelAttention(c)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x):
        # x: Input

        # --- 阶梯式处理流程 ---

        # Step 1: 5x5 分支
        feat5 = self.ortho_conv5(x)  # 处理后特征 B
        # C+ 融合: 输入是 x (原始A) 和 feat5 (处理B)
        node_1 = self.fusion1(x_orig=x, x_processed=feat5)

        # Step 2: 19x19 分支
        feat19 = self.ortho_conv19(node_1)  # 处理后特征 B
        # C+ 融合: 输入是 node_1 (作为这一级的原始A) 和 feat19 (处理B)
        node_2 = self.fusion2(x_orig=node_1, x_processed=feat19)

        # --- 全局融合 (参考绿色图结构) ---

        # 1. 乘法交互: Input * 最后一级融合特征
        merged_mul = x * node_2

        # 2. Conv 1x1 变换
        conv_out = self.mid_conv1x1(merged_mul)

        # 3. 加法融合: Input + Conv结果
        merged_add = x + conv_out

        # --- 注意力与输出 ---

        # 计算注意力权重
        attn_out = self.channel_attn(x)

        # 最终乘法: 混合特征 * 注意力
        final_interaction = merged_add * attn_out

        # Norm + 残差
        out = self.norm(final_interaction) + x

        return out