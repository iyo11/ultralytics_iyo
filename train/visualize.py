from ultralytics import YOLO

# 加载模型
model = YOLO("../pts/v8m_best.pt")

#获取model名
name = model.model_name


# 可视化特征图
# 修改后
model.predict(
    source='E:/datasets/vehicle/images/val2017/20080816144901-01005535_11_10.jpg',
    visualize=True,  # <--- 关键参数：开启特征图可视化
    project="../pts/visualize", # 结果保存的根目录
    name=name          # 结果保存的子文件夹名
)
