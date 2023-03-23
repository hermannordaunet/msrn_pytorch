import torch.nn as nn


def simple_confidence(in_size):
    """get confidence as nn.Sequential"""
    return nn.Sequential(nn.Linear(in_size, 1), nn.Sigmoid())
