import torch.nn as nn

class ExitBlock(nn.Module):
    """Exit Block defition.

    This allows the model to terminate early when it is confident for classification.
    """

    def __init__(self, inplanes, num_classes, input_shape, exit_type):
        super(ExitBlock, self).__init__()
        _, width, height = input_shape
        self.expansion = width * height if exit_type == "plain" else 1

        self.layers = []
        if exit_type == "bnpool":
            self.layers.append(nn.BatchNorm2d(inplanes))
        if exit_type != "plain":
            self.layers.append(nn.AdaptiveAvgPool2d(1))

        self.confidence = nn.Sequential(
            nn.Linear(inplanes * self.expansion, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(inplanes * self.expansion, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).to(x.device)
        x = x.view(x.size(0), -1)
        conf = self.confidence(x)
        pred = self.classifier(x)
        return pred, conf
