import torch
import torch.nn as nn
import torch.nn.functional as F

# Local import
from src.models.utils.exitblock import ExitBlock
from src.models.utils.classifier import simple_classifier
from src.models.utils.confidence import simple_confidence
from src.models.utils.basicblock import BasicBlock
from cnn_residual import CNN_residual


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EE_CNN_residual(nn.Module):
    def __init__(
        self,
        input_shape=(3, 280, 280),
        num_classes=10,
        block=BasicBlock,
        num_ee=1,
        exit_type="bnpool",
        exit_threshold=0.9,
        repetitions=[],
        planes=[],
        dropout_prob=0.5,
    ):
        super(EE_CNN_residual, self).__init__()

        counterpart_model = CNN_residual(
            input_shape=input_shape,
            num_classes=num_classes,
            block=block,
            repetitions=repetitions,
            planes=planes,
        )
        self.input_shape = input_shape
        self.channel = self.input_shape(0)
        self.num_classes = num_classes
        self.block = block

        self.num_ee = num_ee
        self.exit_type = exit_type
        self.exit_threshold = exit_threshold

        self.layers = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.stages = nn.ModuleList()

        self.cost = []
        self.complexity = []

        self.stage_id = 0
        self.inplanes = self.planes[0]

        # Inital layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.channel, 64, kernel_size=7, stride=2, padding=3, bias=False
                ),
                nn.BatchNorm2d(64),
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
            if self.is_suitable_for_exit():
                self.add_exit_block(exit_type, total_flops)

            for _ in range(1, repetition):
                self.layers.append(block(self.inplanes, planes))
                if self.is_suitable_for_exit():
                    self.add_exit_block(exit_type, total_flops)

            planes = self.planes[idx + 1]
            stride = 2

        planes = self.planes[-1]
        self.layers.append(nn.AdaptiveAvgPool2d(1))

        in_size = planes * block.expansion
        self.classifier = simple_classifier(self.num_classes, in_size)
        self.confidence = simple_confidence(in_size)

        self.stages.append(nn.Sequential(*self.layers))
        self.complexity.append((total_flops, total_params))
        self.parameter_initializer(zero_init_residual)

    def set_thresholds(self, distribution, total_flops):
        """set thresholds

        Arguments are
        * distribution:  distribution method of the early-exit blocks.
        * total_flops:   the total FLOPs of the counterpart model.

        This set FLOPs thresholds for each early-exit blocks according to the distribution method.
        """
        gold_rate = 1.61803398875
        flop_margin = 1.0 / (self.num_ee + 1)
        self.threshold = []
        for i in range(self.num_ee):
            if distribution == "pareto":
                self.threshold.append(total_flops * (1 - (0.8 ** (i + 1))))
            elif distribution == "fine":
                self.threshold.append(total_flops * (1 - (0.95 ** (i + 1))))
            elif distribution == "linear":
                self.threshold.append(total_flops * flop_margin * (i + 1))
            else:
                self.threshold.append(total_flops * (gold_rate ** (i - self.num_ee)))
