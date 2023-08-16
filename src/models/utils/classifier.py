import torch.nn as nn


def classifier_linear_softmax(num_classes, in_size):
    """get classifier as nn.Sequential"""
    return nn.Sequential(nn.Linear(in_size, num_classes), nn.Softmax(dim=1))


def classifier_linear(num_classes, in_size):
    """get classifier as nn.Sequential"""
    return nn.Sequential(nn.Linear(in_size, num_classes))

