import torch
import torch.nn as nn
import torch.nn.functional as F


# Local import
from src.models.utils.exitblock import ExitBlock
from src.models.utils.classifier import simple_classifier
from src.models.utils.confidence import simple_confidence


class small_DQN_EE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        img_height=280,
        img_width=280,
        num_classes=10,
        dropout_prob=0.5,
    ):
        super(small_DQN_EE, self).__init__()
        # ---- CONVOLUTIONAL NEURAL NETWORK ----
        HIDDEN_LAYER_1_OUT = 16
        HIDDEN_LAYER_2_OUT = 32
        HIDDEN_LAYER_3_OUT = 64
        KERNEL_SIZE = 3  # original = 5
        STRIDE = 1  # original = 2

        self._in_channels = in_channels
        self._img_height = img_height
        self._img_width = img_width
        self._num_classes = num_classes

        self.exits = nn.ModuleList()

        # TODO: Add this back
        # self.memory = memory

        # TODO: Add all the layers into nn.ModuleList() so that it can be ran through in for loops.

        # Layer 1 with batch norm
        self.conv1 = nn.Conv2d(
            self._in_channels,
            HIDDEN_LAYER_1_OUT,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
        )
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1_OUT)

        # Layer 2 with batch norm
        self.conv2 = nn.Conv2d(
            HIDDEN_LAYER_1_OUT,
            HIDDEN_LAYER_2_OUT,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
        )
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2_OUT)

        # Layer 3 with batch norm
        self.conv3 = nn.Conv2d(
            HIDDEN_LAYER_2_OUT,
            HIDDEN_LAYER_3_OUT,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
        )
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3_OUT)

        # Dropout layer for generalization and overfitting
        self.dropout = nn.Dropout(dropout_prob)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size. This function computes it
        def conv2d_size_out(size, kernel_size=KERNEL_SIZE, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Calculate the number of in features for the bootleneck linear layer (fc1).
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self._img_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self._img_height)))
        linear_input_size = convw * convh * HIDDEN_LAYER_3_OUT

        # bottleneck Linear layer
        self.fc1 = nn.Linear(linear_input_size, 500)

        # Last linear for class probability distribution
        # DELETE: Is this the same as the first part of the classifier?
        # self.fc2 = nn.Linear(500, self._num_classes)
        # self.logSoftmax = nn.LogSoftmax(dim=1)

        # TODO: Clean up all these
        self.input_shape = (self._in_channels, self._img_width, self._img_height)
        self.exit_threshold = 0.3
        self.exit_type = "bnpool"

        # TODO: Make this dynamic in terms of how far into the network the exit is
        self.cost = [0.08, 0.26]

        self.exits.append(
            ExitBlock(
                HIDDEN_LAYER_1_OUT, self._num_classes, self.input_shape, self.exit_type
            )
        )
        self.exits.append(
            ExitBlock(
                HIDDEN_LAYER_2_OUT, self._num_classes, self.input_shape, self.exit_type
            )
        )
        self.last_confidence = simple_confidence(30976)
        self.last_classifier = simple_classifier(num_classes, 30976)
        self.pool = nn.AdaptiveAvgPool2d(1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        preds, confs = list(), list()

        # First layer
        x = self.conv1(x)

        # First EE block
        pred, conf = self.exits[0](x)

        if not self.training and conf.item() > self.exit_threshold:
            return pred, 0, self.cost[0], conf.item()

        preds.append(pred)
        confs.append(conf)

        x = F.leaky_relu(self.bn1(x))

        # Second layer
        x = self.conv2(x)

        # Second EE block
        pred, conf = self.exits[1](x)
        if not self.training and conf.item() > self.exit_threshold:
            return pred, 1, self.cost[1], conf.item()

        preds.append(pred)
        confs.append(conf)

        x = F.leaky_relu(self.bn2(x))

        # Third layer
        x = self.conv3(x)

        e_x = x.view(x.size(0), -1)
        conf = self.last_confidence(e_x)
        pred = self.last_classifier(e_x)

        x = F.leaky_relu(self.bn3(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)

        # TODO: Find out if this is the same as running through conf
        # x = self.fc2(x)

        if not self.training:
            return pred, len(self.exits), 1.0, conf.item()

        preds.append(pred)
        confs.append(conf)

        return preds, confs, self.cost
