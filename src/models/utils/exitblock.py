import torch
import torch.nn as nn

from src.models.utils.classifier import classifier_linear_softmax, classifier_linear
from src.models.utils.confidence import confidence_linear_sigmoid


class ExitBlock(nn.Module):
    """Exit Block defition.

    This allows the model to terminate early when it is confident for classification.
    """

    def __init__(self, inplanes, num_classes, input_shape, exit_type):
        super(ExitBlock, self).__init__()
        _, width, height = input_shape
        self.expansion = width * height if exit_type == "plain" else 1

        self.layers = nn.ModuleList()
        if exit_type == "bnpool":
            self.layers.append(nn.BatchNorm2d(inplanes))
        if exit_type != "plain":
            self.layers.append(nn.AdaptiveAvgPool2d((1,1)))

        in_size = inplanes * self.expansion
        self.confidence = confidence_linear_sigmoid(in_size)
        self.classifier = classifier_linear(in_size, num_classes)
        # self.classifier = classifier_linear_softmax(in_size, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = torch.flatten(x, 1)
        conf = self.confidence(x)
        pred = self.classifier(x)

        return pred, conf
