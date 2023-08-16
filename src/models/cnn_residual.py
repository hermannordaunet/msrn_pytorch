import torch
import torch.nn as nn
import torch.nn.functional as F

# Local import
# from src.models.utils.exitblock import ExitBlock
# from src.models.utils.classifier import classifier_linear_softmax
# from src.models.utils.confidence import confidence_linear_sigmoid
from src.models.utils.basicblock import BasicBlock


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CNN_Residual(nn.Module):
    def __init__(
        self,
        input_shape=(3, 280, 280),
        num_classes=10,
        block=BasicBlock,
        repetitions=list(),
        planes=list(),
    ):
        super(CNN_Residual, self).__init__()
        self.input_shape = input_shape
        self.channel = self.input_shape[0]
        self.num_classes = num_classes
        self.block = block
        self.planes = planes

        # self.num_ee = num_ee
        # self.exit_type = exit_type
        # self.exit_threshold = exit_threshold

        self.layers = nn.ModuleList()
        self.stage = None

        self.stage_id = 0
        self.inplanes = self.planes[0]

        # Inital layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.channel,
                    self.inplanes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        )

        planes = self.inplanes
        stride = 1

        for idx, repetition in enumerate(repetitions):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            self.layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion

            for _ in range(1, repetition):
                self.layers.append(block(self.inplanes, planes))

            planes = self.planes[idx + 1]
            stride = 2

        planes = self.planes[-1]

        self.stage = nn.Sequential(*self.layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected = nn.Linear(planes * block.expansion, num_classes)

    def forward(self, x):
        x = self.stage(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        return x
