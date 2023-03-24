import torch
import torch.nn as nn
import torch.nn.functional as F

# Local import
from src.models.utils.exitblock import ExitBlock
from src.models.utils.classifier import simple_classifier
from src.models.utils.confidence import simple_confidence
from src.models.utils.basicblock import BasicBlock

class CNN_residual(nn.Module):
    def __init__(
        self,
        input_shape=(3, 280, 280), 
        num_classes=10,
        block=BasicBlock, 
        dropout_prob=0.5,
    ):
        super(CNN_residual, self).__init__()
        self.layers = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.stages = nn.ModuleList()

        self.cost = []
        self.complexity = []

        self.stage_id = 0

        

