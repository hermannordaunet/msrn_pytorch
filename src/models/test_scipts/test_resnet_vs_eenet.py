import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

# Local import
from src.models.ee_cnn_residual import EE_CNN_Residual


class DQN_ResNet(nn.Module):
    def __init__(
        self, nn_inputs, h, w, outputs, memory, weight_init="kaiminghe"
    ) -> None:
        super(DQN_ResNet, self).__init__()

        self.nn_inputs = nn_inputs
        self.h = h
        self.w = w
        self.outputs = outputs
        self.memory = memory
        self.weight_init = weight_init

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )

        self.num_classes = outputs

        pretrained_resnet = True
        pretrained_net = models.resnet18(pretrained=pretrained_resnet)

        print(f"[INFO] Pretrained ResNet18 loaded: {pretrained_resnet}")

        print(pretrained_net)
        original_layer = pretrained_net.conv1
        current_weights = original_layer.weight.clone()

        pretrained_net.conv1 = nn.Conv2d(
            self.nn_inputs,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            dilation=original_layer.dilation,
            bias=original_layer.bias,
        )

        new_first_layer = pretrained_net.conv1.weight.clone()

        # TODO: Change the first layer of resnet when Unity is involved
        new_first_layer[:, :3] = current_weights

        if self.weight_init == "kaiminghe":
            new_first_layer[:, 3] = nn.init.kaiming_normal_(new_first_layer[:, 3])
            new_first_layer[:, 4] = nn.init.kaiming_normal_(new_first_layer[:, 4])
        else:
            new_first_layer[:, 3] = nn.init.xavier_uniform_(new_first_layer[:, 3])
            new_first_layer[:, 4] = nn.init.xavier_uniform_(new_first_layer[:, 4])

        pretrained_net.conv1.weight = nn.Parameter(new_first_layer)

        num_ftrs = pretrained_net.fc.in_features
        pretrained_net.fc = nn.Linear(num_ftrs, self.outputs)

        self.net = pretrained_net

    def forward(self, inputs):
        x = self.net(inputs)
        conf = torch.max(self.softmax(x)).item()
        return x, conf


def main():
    ee_net = EE_CNN_Residual(
        input_shape=(5, 280, 280),
        num_ee=0,
        init_planes=64,
        planes=[64, 128, 256, 512],
        num_classes=3,
        repetitions=[2, 2, 2, 2],
    )

    resnet_original = DQN_ResNet(
        5,
        280,
        280,
        3,
        None,
    )

    image = torch.rand((1, 5, 280, 280))

    print(ee_net(image))
    print(resnet_original(image))


if __name__ == "__main__":
    main()
