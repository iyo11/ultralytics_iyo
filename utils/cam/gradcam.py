import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. 加载你训练好的模型 (包含 LGASDPA 或 StandardSDPA)
model = YOLO(r"C:\Users\IYO\Desktop\fsdownload\train\v11n_LGASDPA_NWPU_300\weights\best.pt")

# 用于存储提取的注意力权重
attention_weights = []


# 2. 定义 Hook 函数
def hook_fn(module, input, output):
    # 注意：SDPA 的输出通常是 (batch, heads, seq_len, seq_len) 或 (batch, heads, h, w)
    # 这里我们捕获它并存入列表
    attention_weights.append(output.detach())


# 3. 注册 Hook
# 你需要根据模型结构找到具体的注意力层名称
# 假设你的 SDPA 模块在某个 C3k2 或 C2PSA 内部
target_layer = None
for name, module in model.model.named_modules():
    if "StandardAttention_SDPA" in str(type(module)):
        target_layer = module
        print(f"成功定位目标层: {name}")
        target_layer.register_forward_hook(hook_fn)
        break

# 4. 读取图片并进行推理
img_path = r"E:\datas\NWPU_VHR\valid\images\000563_jpg.rf.58f27b06e0a2b512f3cfa416b270527e.jpg"
results = model.predict(img_path, conf=0.25)

# 5. 处理并可视化热力图
if attention_weights:
    # 假设权重形状为 [1, num_heads, h*w, h*w]，取所有 head 的平均值
    attn = attention_weights[0][0].mean(dim=0).cpu().numpy()

    # 将长向量 Reshape 回二维特征图形状 (根据你的特征图大小调整)
    # 例如特征图是 20x20
    side = int(np.sqrt(attn.shape[-1]))
    heatmap = attn.mean(axis=0).reshape(side, side)

    # 归一化处理
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.uint8(255 * heatmap)

    # 使用 OpenCV 生成伪彩色图
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 读取原图并叠加
    raw_img = cv2.imread(img_path)
    heatmap_img = cv2.resize(heatmap_img, (raw_img.shape[1], raw_img.shape[0]))
    combined = cv2.addWeighted(raw_img, 0.6, heatmap_img, 0.4, 0)

    # 展示结果
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_COLOR_BGR2RGB))
    plt.title("SDPA Attention Heatmap")
    plt.show()
else:
    print("未捕获到注意力权重，请检查 target_layer 是否正确。")