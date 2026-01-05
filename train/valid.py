import warnings
warnings.simplefilter("ignore")          # 比 filterwarnings("ignore") 更“总开关”
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ultralytics import YOLO
if __name__ == '__main__':
    # 1. 加载模型
    # 注意：一定要加载训练好的权重文件 (例如 best.pt)，不要只加载配置文件 (.yaml)
    # 如果你刚才训练完，路径通常在 runs/detect/trainXX/weights/best.pt
    model = YOLO(r"C:\Users\IYO\Desktop\fsdownload\train\v11n_LGASDPA_NWPU_300\weights\best.pt")

    # 2. 运行验证
    # data参数通常不需要指定，因为 .pt 文件里已经记住了数据集路径
    # 但为了保险，你可以显式指定 data='你的数据集.yaml'
    metrics = model.val(
        data='F:/ultralytics-main/datasets_local/NWPU_VHR.yaml',  # 替换为你实际使用的 yaml 路径
        split='val',  # 验证集 (默认)
        batch=16,  # 根据显存调整，你的4060 8GB可以尝试 16 或 32
        device=0,  # 使用 GPU
        project='runs/val',  # 结果保存的主目录
        name='exp_reval',  # 结果保存的子目录名
        visualize=True,  # 可视化验证结果
    )

    # 3. 打印结果 (可选)
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")