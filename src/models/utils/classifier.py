import torch.nn as nn


def simple_classifier(num_classes, in_size):
    """get classifier as nn.Sequential"""
    return nn.Sequential(nn.Linear(in_size, num_classes), nn.Softmax(dim=1))
