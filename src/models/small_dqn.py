import torch.nn as nn
import torch.nn.functional as F


class small_DQN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        img_height=280,
        img_width=280,
        num_classes=10,
        dropout_prob=0.5,
    ):
        super(small_DQN, self).__init__()
        # ---- CONVOLUTIONAL NEURAL NETWORK ----
        HIDDEN_LAYER_1_OUT = 16
        HIDDEN_LAYER_2_OUT = 32
        HIDDEN_LAYER_3_OUT = 32
        KERNEL_SIZE = 3  # original = 5
        STRIDE = 1  # original = 2

        self._in_channels = in_channels
        self._img_height = img_height
        self._img_width = img_width
        self._num_classes = num_classes

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

        # DELETE: Softmax for making a conf score. Delete when EE is implemented
        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )

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
        self.fc2 = nn.Linear(500, self._num_classes)

        self.logSoftmax = nn.LogSoftmax(dim=1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.logSoftmax(x)

        # DELETE: Remove the conf
        # conf = torch.max(self.softmax(x)).item()
        return x  # , conf
