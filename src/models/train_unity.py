import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# import the necessary torch packages
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# import the necessary torchvision packages
from torchvision.datasets import KMNIST, CIFAR10, ImageNet
from torchvision.transforms import ToTensor

# import the necessary sklearn packages
from sklearn.metrics import classification_report

# import the necassary scipy packages
from scipy import stats

# Local import
from small_dqn import small_DQN
from small_dqn_ee import small_DQN_EE
from ee_cnn_residual import EE_CNN_Residual
from utils.loss_functions import loss_v1, loss_v2
from utils.data_utils import min_max_conf_from_dataset
from utils.print_utils import print_min_max_conf, print_cost_of_exits

# unity imports
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[INFO] Device is: {device}")
    
    SEED = 1804


    env = UnityEnvironment(
        # file_name=ENV_PATH,
        seed=SEED,
        # side_channels=[side_channel_1],
        no_graphics=False,
    )
    env.reset()

    behavior_specs = env.behavior_specs
    env_object = list(env._env_specs)[0]

    decision_steps, terminal_steps = env.get_steps(env_object)

    print()

if __name__ == "__main__":
    main()