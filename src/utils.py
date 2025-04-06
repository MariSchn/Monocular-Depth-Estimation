import os
import yaml
import torch


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def target_transform(depth, input_size=(426, 560)):
    # Resize the depth map to match input size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0), 
        size=input_size, 
        mode='bilinear', 
        align_corners=True
    ).squeeze()
    
    # Add channel dimension to match model output
    depth = depth.unsqueeze(0)
    return depth