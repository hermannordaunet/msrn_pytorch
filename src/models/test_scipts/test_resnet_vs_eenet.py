import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

import numpy as np
import random

# Local import
from src.models.ee_cnn_residual import EE_CNN_Residual
from src.models.resnet_original import ResNet


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(1027)
    ee_net = EE_CNN_Residual(
        input_shape=(5, 280, 280),
        num_ee=0,
        init_planes=64,
        planes=[64, 128, 256, 512],
        num_classes=3,
        repetitions=[2, 2, 2, 2],
    )

    set_seed(1027)
    resnet_original = ResNet(
        input_shape=(5, 280, 280),
        init_planes=64,
        planes=[64, 128, 256, 512],
        num_classes=3,
        repetitions=[2, 2, 2, 2],
    )

    image = torch.rand((1, 5, 280, 280))

    print(ee_net(image))
    print(resnet_original(image))


if __name__ == "__main__":
    main()
