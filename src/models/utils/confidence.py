import torch.nn as nn


def confidence_linear_sigmoid(in_size):
    """get confidence as nn.Sequential"""
    return nn.Sequential(nn.Linear(in_size, 1), nn.Sigmoid())

def confidence_linear_softmax(in_size):
    """get confidence as nn.Sequential"""
    return nn.Sequential(nn.Linear(in_size, 1), nn.Softmax(dim=1))