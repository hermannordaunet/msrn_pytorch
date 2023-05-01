import torch


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

    min_vals = min_vals.tolist()
    max_vals = max_vals.tolist()

    return (min_vals, max_vals)