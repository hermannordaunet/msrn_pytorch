import os
import json
import torch

def save_model(q_net_local, folder_path, model_type="last"):
    """
    Save PyTorch models to the given folder path.
    """
    # Create folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save models
    model_path = os.path.join(folder_path, f'{model_type}_model.pt')
    torch.save(q_net_local.state_dict(), model_path)


def save_dict_to_json(dict_obj, file_path):
    """
    Save a dictionary as a JSON file at the given file path.
    """
    with open(file_path, 'w') as f:
        json.dump(dict_obj, f)