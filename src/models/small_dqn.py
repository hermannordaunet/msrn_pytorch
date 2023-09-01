import torch.nn as nn
import torch.nn.functional as F

# Local import
from src.models.utils.classifier import classifier_linear_softmax, classifier_linear
from src.models.utils.confidence import confidence_linear_sigmoid


class small_DQN(nn.Module):
    def __init__(
        self,
        input_shape=[3, 280, 280],
        num_classes=10,
        **kwargs,
    ):
        super(small_DQN, self).__init__()
        # ---- CONVOLUTIONAL NEURAL NETWORK ----
        HIDDEN_LAYER_1_OUT = 64
        HIDDEN_LAYER_2_OUT = 64
        HIDDEN_LAYER_3_OUT = 32
        KERNEL_SIZE = 5  # original = 5
        STRIDE = 2  # original = 2

        self._in_channels = input_shape[0]
        self._img_height = input_shape[1]
        self._img_width = input_shape[2]
        self.num_classes = num_classes

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

        # DELETE: Softmax for making a conf score. Delete when EE is implemented
        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )
        self.exits = nn.ModuleList()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size. This function computes it
        def conv2d_size_out(size, kernel_size=KERNEL_SIZE, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Calculate the number of in features for the bootleneck linear layer (fc1).
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self._img_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self._img_height)))
        linear_input_size = convw * convh * HIDDEN_LAYER_3_OUT

        self.head = nn.Linear(linear_input_size, self.num_classes)
        # bottleneck Linear layer
        self.fc1 = nn.Linear(linear_input_size, 500)

        # Last linear for class probability distribution
        self.fc2 = nn.Linear(500, self.num_classes)

        self.classifier = classifier_linear(linear_input_size, self.num_classes)
        self.confidence = confidence_linear_sigmoid(linear_input_size)

        self.logSoftmax = nn.LogSoftmax(dim=1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if self.training:
            preds, confs, costs = list(), list(), list()

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        pred = self.head(x)
        conf = self.confidence(x)

        if self.training:
            preds.append(pred)
            confs.append(conf)
            costs.append(1.0)

        # DELETE: Remove the conf
        # conf = torch.max(self.softmax(x)).item()
        if self.training:
            return preds, confs, costs
        else:
            return pred, conf, 1, 1.0
