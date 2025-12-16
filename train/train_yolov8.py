import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO


# 检测运行环境 Win or Linux
import platform
if platform.system() == 'Windows':
    datasets_path = '../datasets_local'
    batch_size = 8
    workers = 4
    cacheTF =  False
else:
    datasets_path = '../datasets'
    batch_size = 24
    workers = 10
    cacheTF =  True


if __name__ == '__main__':
    #model = YOLO('yolov8n.yaml')
    model = YOLO('../models/v8/yolov8n.yaml')
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov8s.yaml就是使用的v8s,
    # 类似某个改进的yaml文件名称为yolov8-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov8l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    model.train(data= datasets_path + '/VisDrone.yaml',
                cache=cacheTF,
                imgsz=640,
                epochs=200,
                single_cls=False,  # 是否是单类别检测
                batch=batch_size,
                close_mosaic=10,
                workers=workers,
                device='0',
                optimizer='SGD', # using SGD
                #resume='C:\\Users\\IYO\\Desktop\\Strip\\ultralytics-main\\runs\\train\\exp4', # 续训设置last.pt的地址
                resume=False,
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='../runs/train',
                name='exp',
                save_period=10,
                )
