import torch.nn as nn


def classifier_linear_softmax(in_size, num_classes):
    """get classifier as nn.Sequential"""
    return nn.Sequential(nn.Linear(in_size, num_classes), nn.Softmax(dim=1))


def classifier_linear(in_size, num_classes):
    """get classifier as nn.Sequential"""
    return nn.Linear(in_size, num_classes)

