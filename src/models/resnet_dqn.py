import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from src.models.utils.flops_counter import get_model_complexity_info

class ResNet_DQN(nn.Module):
    def __init__(
        self, input_shape=(3, 280, 280), num_classes=10, num_ee=0,  weight_init="kaiminghe", **kwargs
    ) -> None:
        super(ResNet_DQN, self).__init__()

        self.input_shape = tuple(input_shape)
        self.channel = self.input_shape[0]
        self.num_classes = num_classes
        self.weight_init = weight_init
        self.num_ee = num_ee
        self.exits = nn.ModuleList()

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )

        self.complexity = list()

        pretrained_resnet = True
        pretrained_net = models.resnet18(pretrained=pretrained_resnet)

        print(f"[INFO] Pretrained ResNet18 loaded: {pretrained_resnet}")

        original_layer = pretrained_net.conv1
        current_weights = original_layer.weight.clone()

        pretrained_net.conv1 = nn.Conv2d(
            self.channel,
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
            new_first_layer[:, 3] = nn.init.kaiming_uniform_(new_first_layer[:, 3])
            new_first_layer[:, 4] = nn.init.kaiming_uniform_(new_first_layer[:, 4])
        else:
            new_first_layer[:, 3] = nn.init.xavier_uniform_(new_first_layer[:, 3])
            new_first_layer[:, 4] = nn.init.xavier_uniform_(new_first_layer[:, 4])

        pretrained_net.conv1.weight = nn.Parameter(new_first_layer)

        num_ftrs = pretrained_net.fc.in_features
        pretrained_net.fc = nn.Linear(num_ftrs, self.num_classes)

        self.net = pretrained_net

        total_flops, total_params = self.get_complexity(self.net)
        self.complexity.append((total_flops, total_params))

    def get_complexity(self, model, print_per_layer=False):
        """get model complexity in terms of FLOPs and the number of parameters"""
        flops, params = get_model_complexity_info(
            model,
            self.input_shape,
            print_per_layer_stat=print_per_layer,
            as_strings=False,
        )

        return flops, params

    def forward(self, inputs):

        pred = self.net(inputs)
        conf = torch.max(self.softmax(pred)).item()

        if not self.training:
            return pred, conf, 1, 1.0
        else:
            return [pred], [conf], [1.0]