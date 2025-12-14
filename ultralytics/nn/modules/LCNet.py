import torch
import torch.nn as nn

__all__ = (
    "LCNetTimm",
)

import torch.nn as nn
from sympy import false
from timm import create_model

class LCNetTimm(nn.Module):
    """
    A lightweight backbone for YOLOv11 using timm MobileNetV3-small
    width=0.75 roughly matches LCNet075
    """
    def __init__(self, width=0.75, pretrained=True):
        super().__init__()
        model = create_model(
            "mobilenetv3_small_075",
            pretrained=pretrained,
            features_only=false,      # <-- key: get C2/C3/C4/C5
            out_indices=(1, 2, 3, 4)
        )
        self.model = model

    def forward(self, x):
        # timm returns [C2, C3, C4, C5]
        return self.model(x)
