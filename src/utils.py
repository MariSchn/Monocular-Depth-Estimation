import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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

def create_depth_comparison(model, image, gt, device):
    model.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through the model
        pred = model(image)

    pred = nn.functional.interpolate(
                pred,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )

    # Convert tensors to numpy arrays for visualization
    pred = pred.squeeze().cpu().numpy()
    gt = gt.squeeze()

    min_depth = gt.min()
    max_depth = gt.max()

    # Create the figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(pred, cmap='plasma', vmin=min_depth, vmax=max_depth)
    axs[0].set_title('Predicted Depth')

    axs[0].axis('off')
    axs[1].imshow(gt, cmap='plasma', vmin=min_depth, vmax=max_depth)
    axs[1].set_title('Ground Truth Depth')
    axs[1].axis('off')

    # Render the figure into a numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buffer = fig.canvas.tostring_argb()
    image = np.frombuffer(buffer, dtype=np.uint8)
    image.shape = (h, w, 4) 
    image = image[:, :, 1:4]

    plt.close(fig)

    return image