import os
import json
import torch

# CRITICAL: When saving best model, it has to be seralized.


def save_model(q_policy_net, folder_path, model_type="last"):
    """
    Save PyTorch models to the given folder path.
    """
    # Create folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    model_file_name = f"{model_type}_model.pt"
    # Save models
    model_path = os.path.join(folder_path, model_file_name)
    torch.save(q_policy_net.state_dict(), model_path)


def save_dict_to_json(dict_obj, file_path):
    """
    Save a dictionary as a JSON file at the given file path.
    """
    with open(file_path, "w") as f:
        json.dump(dict_obj, f)

def save_list_to_json(list_obj, file_path):
    with open(file_path, "w") as f:
        json.dump(list(list_obj), f)
