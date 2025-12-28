import warnings
warnings.simplefilter("ignore")          # 比 filterwarnings("ignore") 更“总开关”
warnings.filterwarnings("ignore", category=DeprecationWarning)



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

model = YOLO(r'F:\ultralytics-main\runs\train\v11n_PoolFormerMSConvV2_P2_RSOD_300_old\weights\best.pt')

if __name__ == '__main__':
    model.val(
        data= datasets_path + '/RSOD.yaml',  # 数据集配置文件
        split='val',  # 明确跑测试集
        imgsz=640,
        device='0',
        batch=batch_size,
        workers=workers,
        cache=cacheTF,
    )
