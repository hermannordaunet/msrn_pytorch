import torch
import torch.nn as nn
import torch.nn.functional as F

# Local import
from src.models.utils.exitblock import ExitBlock
from src.models.utils.classifier import simple_classifier
from src.models.utils.confidence import simple_confidence
from src.models.utils.basicblock import BasicBlock
from cnn_residual import CNN_Residual

from utils.flops_counter import get_model_complexity_info


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EE_CNN_Residual(nn.Module):
    def __init__(
        self,
        input_shape=(3, 280, 280),
        num_classes=10,
        block=BasicBlock,
        num_ee=1,
        exit_type="bnpool",
        exit_threshold=0.9,
        repetitions=list(),
        planes=list(),
        distribution=None,
    ):
        super(EE_CNN_Residual, self).__init__()

        # Get the model just without the ee blocks
        counterpart_model = CNN_Residual(
            input_shape=input_shape,
            num_classes=num_classes,
            block=block,
            repetitions=repetitions,
            planes=planes,
        )

        self.planes = planes
        self.input_shape = input_shape
        self.channel = self.input_shape[0]
        self.num_classes = num_classes
        self.block = block

        # Create the early exit variables
        self.num_ee = num_ee
        self.exit_type = exit_type
        self.exit_threshold = exit_threshold

        # Containers for the network
        self.layers = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Cost at each exit and the complexity
        self.cost = list()
        self.complexity = list()

        self.stage_id = 0
        self.inplanes = self.planes[0]

        # Complexity of the entire model and threshold for the early exit
        total_flops, total_params = self.get_complexity(counterpart_model)
        self.set_thresholds(distribution, total_flops)

        # Inital layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
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
            if self.is_suitable_for_exit():
                self.add_exit_block(exit_type, total_flops)
                print(f"Added exit at repetition: {idx+1}, after first block")

            for _ in range(1, repetition):
                self.layers.append(block(self.inplanes, planes))
                if self.is_suitable_for_exit():
                    self.add_exit_block(exit_type, total_flops)
                    print(f"Added exit at repetition: {idx+1}, after second block")

            planes = self.planes[idx + 1]
            stride = 2

        planes = self.planes[-1]
        self.layers.append(nn.AdaptiveAvgPool2d(1))

        # Dropout layer for generalization and overfitting
        # TODO: Find out if this is nice to have in the CNN
        # self.dropout = nn.Dropout(dropout_prob)

        in_size = planes * block.expansion
        self.classifier = simple_classifier(self.num_classes, in_size)
        self.confidence = simple_confidence(in_size)

        self.stages.append(nn.Sequential(*self.layers))
        
        self.complexity.append((total_flops, total_params))
        self.parameter_initializer()

    def set_thresholds(self, distribution, total_flops):
        """set thresholds

        Arguments are
        * distribution:  distribution method of the early-exit blocks.
        * total_flops:   the total FLOPs of the counterpart model.

        This set FLOPs thresholds for each early-exit blocks according to the distribution method.
        """
        gold_rate = 1.61803398875
        flop_margin = 1.0 / (self.num_ee + 1)
        self.threshold = list()
        for i in range(self.num_ee):
            if distribution == "pareto":
                self.threshold.append(total_flops * (1 - (0.8 ** (i + 1))))
            elif distribution == "fine":
                self.threshold.append(total_flops * (1 - (0.95 ** (i + 1))))
            elif distribution == "linear":
                self.threshold.append(total_flops * flop_margin * (i + 1))
            else:
                self.threshold.append(total_flops * (gold_rate ** (i - self.num_ee)))

    def get_complexity(self, model):
        """get model complexity in terms of FLOPs and the number of parameters"""
        flops, params = get_model_complexity_info(
            model, self.input_shape, print_per_layer_stat=False, as_strings=False
        )
        return flops, params

    def parameter_initializer(self, zero_init_residual=False):
        """
        Zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros,
        and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def is_suitable_for_exit(self):
        """is the position suitable to locate an early-exit block"""
        intermediate_model = nn.Sequential(*(list(self.stages) + list(self.layers)))
        flops, _ = self.get_complexity(intermediate_model)
        return self.stage_id < self.num_ee and flops >= self.threshold[self.stage_id]

    def add_exit_block(self, exit_type, total_flops):
        """add early-exit blocks to the model

        Argument is
        * total_flops:   the total FLOPs of the counterpart model.

        This add exit blocks to suitable intermediate position in the model,
        and calculates the FLOPs and parameters until that exit block.
        These complexity values are saved in the self.cost and self.complexity.
        """

        self.stages.append(nn.Sequential(*self.layers))
        self.exits.append(
            ExitBlock(self.inplanes, self.num_classes, self.input_shape, exit_type)
        )
        intermediate_model = nn.Sequential(*(list(self.stages) + list(self.exits)[-1:]))
        flops, params = self.get_complexity(intermediate_model)
        self.cost.append(flops / total_flops)
        self.complexity.append((flops, params))
        self.layers = nn.ModuleList()
        self.stage_id += 1

    def forward(self, x):
        preds, confs = list(), list()

        for idx, exitblock in enumerate(self.exits):
            x = self.stages[idx](x)
            pred, conf = exitblock(x)

            if not self.training and conf.item() > self.exit_threshold:
                return pred, idx, self.cost[idx], conf.item()
            
            preds.append(pred)
            confs.append(conf)

        x = self.stages[-1](x)
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        conf = self.confidence(x)

        if not self.training:
            return pred, len(self.exits), 1.0, conf.item()
        
        preds.append(pred)
        confs.append(conf)

        return preds, confs, self.cost