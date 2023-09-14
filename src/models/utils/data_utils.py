import torch

import numpy as np


def min_max_conf_from_dataset(conf_list: list()) -> tuple():
    # Calculate max and min conf of each exit during training
    # get the number of columns
    # rng = np.random.default_rng(seed)
    # random_idx = rng.integers(low=0, high=len(conf_list))

    tensor_of_conf = torch.Tensor()
    for batch_conf_list in conf_list:
        tensor_of_conf_idx = torch.cat(batch_conf_list, dim=1).cpu()
        tensor_of_conf = torch.cat((tensor_of_conf, tensor_of_conf_idx), dim=0)

    min_vals, _ = torch.min(tensor_of_conf, dim=0)
    max_vals, _ = torch.max(tensor_of_conf, dim=0)
    mean_vals = torch.mean(tensor_of_conf, dim=0)

    min_vals = min_vals.tolist()
    max_vals = max_vals.tolist()
    mean_vals = mean_vals.tolist()

    return (min_vals, max_vals, mean_vals)


def get_grid_based_perception_numpy(agent_obs):
    grid_state_numpy = agent_obs[0]
    grid_based_perception = np.transpose(grid_state_numpy, (2, 0, 1))

    return np.expand_dims(grid_based_perception, axis=0)


def get_grid_based_perception(agent_obs):
    grid_state = agent_obs[0]
    grid_based_perception = torch.tensor(
        grid_state.transpose((2, 0, 1)), dtype=torch.float32
    )

    return grid_based_perception.unsqueeze(0)
