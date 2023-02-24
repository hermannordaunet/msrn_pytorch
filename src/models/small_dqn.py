import torch
import torch.nn as nn
import torch.nn.functional as F


class small_DQN(nn.Module):
    def __init__(self, in_channels, img_height, img_width, num_classes, memory):
        super(small_DQN, self).__init__()
        # ---- CONVOLUTIONAL NEURAL NETWORK ----
        HIDDEN_LAYER_1_OUT = 16
        HIDDEN_LAYER_2_OUT = 32
        HIDDEN_LAYER_3_OUT = 64
        KERNEL_SIZE = 3  # original = 5
        STRIDE = 1  # original = 2

        self._in_channels = in_channels
        self._img_height = img_height
        self._img_height = img_width
        self._num_classes = num_classes
        self.memory = memory

        self.conv1 = nn.Conv2d(
            self._num_classes, HIDDEN_LAYER_1_OUT, kernel_size=KERNEL_SIZE, stride=STRIDE
        )
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1_OUT)
        self.conv2 = nn.Conv2d(
            HIDDEN_LAYER_1_OUT,
            HIDDEN_LAYER_2_OUT,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
        )
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2_OUT)
        self.conv3 = nn.Conv2d(
            HIDDEN_LAYER_2_OUT,
            HIDDEN_LAYER_3_OUT,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
        )
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3_OUT)

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=KERNEL_SIZE, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, self.num_classes)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))

        conf = torch.max(self.softmax(x)).item()
        return x, conf
