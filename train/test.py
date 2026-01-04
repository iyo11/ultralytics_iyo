import warnings

from build.lib.ultralytics.nn import DetectionModel

warnings.simplefilter("ignore")          # 比 filterwarnings("ignore") 更“总开关”
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ultralytics import YOLO

# 检测运行环境 Win or Linux
import platform

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # 1. 加载模型逻辑
    cfg = '../models/v11/yolov11n_MSAWA.yaml'  # 确保路径正确
    model = DetectionModel(cfg)

    # 2. 模拟前向传播并打印尺寸
    x = torch.zeros(1, 3, 640, 640)
    print(f"Input: {x.shape}")

    for i, m in enumerate(model.model):
        try:
            if m.f != -1:  # 处理多输入层
                input_data = [x if j == -1 else y[j] for j in m.f] if isinstance(m.f, list) else y[m.f]
            else:
                input_data = x

            x = m(input_data)
            if i == 0: y = []  # 初始化记录列表
            y.append(x)
            print(f"Layer {i} ({m.type.split('.')[-1]}): Output shape = {x.shape}")
        except Exception as e:
            print(f"--- Layer {i} ({m.type.split('.')[-1]}) 发生错误！ ---")
            if isinstance(input_data, list):
                print(f"Input shapes: {[id.shape for id in input_data]}")
            else:
                print(f"Input shape: {input_data.shape}")
            print(f"Error: {e}")
            break

