# pip install ultralytics grad-cam opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import (
    EigenCAM, EigenGradCAM, GradCAM, GradCAMElementWise, GradCAMPlusPlus,
    HiResCAM, LayerCAM, RandomCAM, ScoreCAM, XGradCAM
)

np.random.seed(3407)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False,
             scaleFill=False, scaleup=True, stride=32):
    """简单 letterbox：保持比例缩放 + padding 到 new_shape"""
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w,h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im


def resolve_layer(det_model: torch.nn.Module, layer_expr: str):
    """
    允许你传类似 'model.model[-4]'、'model.model[10]' 这种表达式
    这里只给 eval 一个很小作用域：只有 model 可用
    """
    scope = {"model": det_model}
    try:
        layer = eval(layer_expr, {"__builtins__": {}}, scope)
    except Exception as e:
        raise ValueError(
            f"解析 layer 失败：{layer_expr}\n报错：{e}\n"
            f"提示：先打印 blocks 索引，再写 layer 表达式。"
        )
    return layer


class YOLOBoxScoreTarget:
    """
    ✅ 修复版 target：
    - 必须从 pytorch-grad-cam 内部 forward 得到的 model_output 里取分数
    - det_index_in_raw：原始 N 个候选里的索引（不是排序后的 i）
    - mode: 'class' | 'box' | 'all'
    """
    def __init__(self, det_index_in_raw: int, nc: int, mode: str = "class"):
        self.det_i = int(det_index_in_raw)
        self.nc = int(nc)
        self.mode = mode

    def __call__(self, model_output):
        # model_output 可能是 list/tuple
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]

        # 统一成 [1, N, C]
        if model_output.dim() != 3 or model_output.size(0) != 1:
            raise RuntimeError(f"model_output 形状不符合预期：{tuple(model_output.shape)}")

        # 如果是 [1, C, N]，转成 [1, N, C]
        if model_output.shape[1] < model_output.shape[2]:
            model_output = model_output.permute(0, 2, 1).contiguous()

        x = model_output[0, self.det_i]  # [C]
        boxes = x[:4]
        logits = x[4:4 + self.nc]

        if self.mode == "class":
            return logits.max()
        elif self.mode == "box":
            return boxes.sum()
        else:
            return logits.max() + boxes.sum()


class YOLOv8GradCAM:
    def __init__(
        self,
        weight="yolov8n.pt",
        device="cuda:0",
        method="GradCAM",
        layer="model.model[-4]",
        backward_type="all",     # 'class' | 'box' | 'all'
        conf_threshold=0.6,
        ratio=0.05,
        imgsz=640,
        print_blocks=True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        yolo = YOLO(weight)

        # 打印 block 级结构（用于选 layer）
        if print_blocks:
            m0 = yolo.model
            blocks = getattr(m0, "model", None)
            print("模型结构（block级，选 layer 用）：")
            if blocks is None:
                print(m0)
            else:
                for i, b in enumerate(blocks):
                    p = sum(x.numel() for x in b.parameters())
                    print(f"{i:>3}: {b.__class__.__name__:<18} params={p}")

        self.model = yolo.model.to(self.device).eval()
        self.names = yolo.names if hasattr(yolo, "names") else self.model.names

        self.target_layer = resolve_layer(self.model, layer)
        self.backward_type = backward_type
        self.conf_threshold = float(conf_threshold)
        self.ratio = float(ratio)
        self.imgsz = int(imgsz)

        # Detect head 的 nc（最稳）
        self.nc = None
        try:
            self.nc = getattr(self.model.model[-1], "nc", None)
        except Exception:
            self.nc = None

        method_cls = eval(method)
        self.cam = method_cls(model=self.model, target_layers=[self.target_layer])

        print("\n你选择的 target layer 是：", layer)
        print("层类型：", self.target_layer.__class__.__name__)
        print("层结构：\n", self.target_layer)
        p = sum(x.numel() for x in self.target_layer.parameters())
        print("该层参数量：", p)

    def _name(self, cid: int) -> str:
        if isinstance(self.names, dict):
            return self.names.get(cid, str(cid))
        return self.names[cid]

    def _preprocess(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"读图失败：{img_path}")

        img_bgr = letterbox(img_bgr, new_shape=(self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return img_rgb, tensor

    def _postprocess_sort(self, preds):
        """
        兼容两种常见输出：
        - [B, N, 4+nc]
        - [B, 4+nc, N]  （例如 [1,84,8400]）
        统一成 [B, N, 4+nc] 再处理
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.dim() != 3 or preds.size(0) != 1:
            raise RuntimeError(f"preds 形状不符合预期：{tuple(preds.shape)}")

        # 如果是 [B, C, N]（C 小、N 大），转成 [B, N, C]
        if preds.shape[1] < preds.shape[2]:
            preds = preds.permute(0, 2, 1).contiguous()  # [1, N, C]

        preds = preds[0]  # [N, C]
        C = preds.shape[1]

        # nc 优先来自 Detect head
        nc = self.nc
        if nc is None:
            nc = C - 4
        if C < 4 + nc:
            raise RuntimeError(f"通道数不够：C={C}, nc={nc}，无法切分 boxes/cls。")

        boxes = preds[:, :4]            # [N,4]
        logits = preds[:, 4:4 + nc]     # [N,nc]

        conf = logits.max(dim=1).values
        sorted_conf, idx = torch.sort(conf, descending=True)

        logits_sorted = logits[idx]
        boxes_sorted = boxes[idx]
        boxes_xyxy = xywh2xyxy(boxes_sorted).detach().cpu().numpy()
        return logits_sorted, boxes_sorted, boxes_xyxy, sorted_conf, idx, nc

    def debug_layer_output_shape(self, img_path):
        img_rgb, tensor = self._preprocess(img_path)

        feats = {}

        def hook_fn(m, inp, out):
            if isinstance(out, (list, tuple)):
                feats["out"] = [tuple(x.shape) for x in out if hasattr(x, "shape")]
            else:
                feats["out"] = tuple(out.shape)

        h = self.target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(tensor)

        h.remove()

        print("\n=== Layer Debug ===")
        print("layer expr:", self.target_layer)
        print("output shape:", feats.get("out", "没抓到输出（可能该层没走到或输出不是Tensor）"))

    def __call__(self, img_path, save_path="./cam_results"):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        img_rgb, tensor = self._preprocess(img_path)

        # ✅ grad-cam 需要梯度
        torch.set_grad_enabled(True)
        tensor.requires_grad_(True)

        # 第一次 forward：只用于排序挑选 topk
        preds = self.model(tensor)
        logits_sorted, boxes_sorted, boxes_xyxy, sorted_conf, idx_sorted, nc = self._postprocess_sort(preds)

        topk = max(1, int(logits_sorted.size(0) * self.ratio))

        saved = 0
        for i in range(topk):
            if float(sorted_conf[i]) < self.conf_threshold:
                break

            # ✅ 关键：把 “原始候选索引” 传给 target，让它在 CAM 内部 forward 的 model_output 里取分数
            raw_i = int(idx_sorted[i].item())
            targets = [YOLOBoxScoreTarget(raw_i, nc, self.backward_type)]

            grayscale_cam = self.cam(input_tensor=tensor, targets=targets)
            cam_map = grayscale_cam[0]  # [H,W]

            cam_img = show_cam_on_image(img_rgb.copy(), cam_map, use_rgb=True)

            cls_id = int(logits_sorted[i].argmax().item())
            label = f"{self._name(cls_id)}_{float(sorted_conf[i]):.2f}"

            # 如需画框，取消注释
            # x1,y1,x2,y2 = boxes_xyxy[i].astype(int).tolist()
            # cv2.rectangle(cam_img, (x1,y1), (x2,y2), (255, 182, 193), 2)
            # cv2.putText(cam_img, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,182,193), 2)

            Image.fromarray(cam_img).save(os.path.join(save_path, f"{i:03d}_{label}.png"))
            saved += 1

        return saved


def get_params():
    return dict(
        weight="../../pts/best.pt",
        device="cuda:0",
        method="XGradCAM",          # 如果仍然偏弱，可试 XGradCAM / EigenCAM
        layer="model.model[23]",   # 你之前 debug 的 PCMFAM 是 19（日志里显示 19: PCMFAM）
        backward_type="all",       # class / box / all
        conf_threshold=0.6,
        ratio=0.05,
        imgsz=640,
        print_blocks=True
    )


if __name__ == "__main__":
    params = get_params()
    cam = YOLOv8GradCAM(**params)

    img_path = "E:/datas/NWPU_VHR/images/2.jpg"

    # 可选：先看输出特征图 shape
    cam.debug_layer_output_shape(img_path)

    save_dir = "./cam_results"
    n = cam(img_path, save_dir)
    print(f"Saved {n} CAM images to: {os.path.abspath(save_dir)}")
