import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


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

def image_gradients(x: torch.Tensor, convert_to_gray: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the gradient image of the input image.
    To handle multi-channel images, the image is first converted to grayscale. if you do not want this, `set convert_to_gray` to False.

    Shape Variables:
        - B = batch size
        - C_in = input channels
        - C_out = output channels (1 for convert_to_gray=True, C_in for convert_to_gray=False)
        - H = height
        - W = width

    Args:
        x (torch.Tensor): Input image tensor of shape (B, C_in, H, W) or (C_in, H, W) or (H, W).
        convert_to_gray (bool): Whether to convert the image to grayscale before calculating gradients.
    Returns:
        - dx (torch.Tensor): Gradient image in the x-direction of shape (B, C_out, H, W).
        - dy (torch.Tensor): Gradient image in the y-direction of shape (B, C_out, H, W).
    """
    # Make sure the image has all dimensions
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)

    if convert_to_gray:
        x = x.mean(dim=1, keepdim=True)

    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx = right - left
    dy = bottom - top 

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy

def gradient_regularizer(depth: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient regularization term for given depth maps.
    The gradient regularization term is defined as the mean of the absolute values of the gradients in both x and y directions.

    Shape Variables:
        - B = batch size
        - H = height
        - W = width
        
    Args:
        depth (torch.Tensor): Depth map tensor of shape (B, 1, H, W) or (1, H, W) or (H, W).
    Returns:
        torch.Tensor: Value of the gradient regularization term.
    """
    if (depth.ndim == 3 and depth.shape[0] != 1) or (depth.ndim == 4 and depth.shape[1] != 1):
        raise ValueError("Depth map should have just one channel.")

    dx, dy = image_gradients(depth, convert_to_gray=True)
    grad_x = torch.abs(dx)
    grad_y = torch.abs(dy)

    return grad_x.mean() + grad_y.mean()
    
def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient loss between the predicted and ground truth depth maps.
    The gradient loss is defined as the mean of the absolute differences of the gradients in both x and y directions.

    Shape Variables:
        - B = batch size
        - H = height
        - W = width

    Args:
        pred (torch.Tensor): Predicted depth map tensor of shape (B, 1, H, W) or (1, H, W) or (H, W).
        gt (torch.Tensor): Ground truth depth map tensor of shape (B, 1, H, W) or (1, H, W) or (H, W).
    Returns:
        torch.Tensor: Value of the gradient loss.
    """
    if (pred.ndim == 3 and pred.shape[0] != 1) or (pred.ndim == 4 and pred.shape[1] != 1):
        raise ValueError("Predicted depth map should have just one channel.")
    
    if (gt.ndim == 3 and gt.shape[0] != 1) or (gt.ndim == 4 and gt.shape[1] != 1):
        raise ValueError("Ground truth depth map should have just one channel.")

    dx_pred, dy_pred = image_gradients(pred)
    dx_gt, dy_gt = image_gradients(gt)

    grad_x_loss = F.l1_loss(dx_pred, dx_gt)
    grad_y_loss = F.l1_loss(dy_pred, dy_gt)

    return grad_x_loss + grad_y_loss

def create_depth_comparison(model: torch.nn.Module, image: torch.Tensor, gt: torch.Tensor, device: str) -> np.ndarray:
    """
    Creates a comparison figure which visualizes the predicted depth and the ground truth depth.

    Shape Variables:
        - B = batch size
        - C = number of channels
        - H = height
        - W = width

    Args:
        model (torch.nn.Module): The depth estimation model to be used.
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        gt (torch.Tensor): Ground truth depth tensor of shape (B, 1, H, W).
        device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
        np.ndarray: Rendered image of the comparison figure.
    """
    model.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through the model
        pred, _ = model(image)

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

def create_uncertainty_visualization(model: torch.nn.Module, train_image: torch.Tensor, val_image: torch.Tensor, device: str) -> np.ndarray:
    """
    Creates a comparison figure which visualizes the model's uncertainty for both training and validation images.

    Shape Variables:
        - B = batch size
        - C = number of channels
        - H = height
        - W = width

    Args:
        model (torch.nn.Module): The depth estimation model to be used.
        train_image (torch.Tensor): Input image tensor from train set of shape (B, C, H, W).
        val_image (torch.Tensor): Input image tensor from validation set of shape (B, C, H, W).
        device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
        np.ndarray: Rendered image of the comparison figure.
    """
    model.eval()

    if train_image.ndim == 3:
        train_image = train_image.unsqueeze(0).to(device)
    if val_image.ndim == 3:
        val_image = val_image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through the model
        _, train_uncertainty = model(train_image)
        _, val_uncertainty = model(val_image)

    # Convert tensors to numpy arrays for visualization
    train_uncertainty = train_uncertainty.squeeze().cpu().numpy()
    val_uncertainty = val_uncertainty.squeeze().cpu().numpy()

    min_uncertainty = min(train_uncertainty.min(), val_uncertainty.min())
    max_uncertainty = max(train_uncertainty.max(), val_uncertainty.max())

    # Create the figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axs[0].imshow(train_uncertainty, cmap='viridis', vmin=min_uncertainty, vmax=max_uncertainty)
    axs[0].set_title('Train Uncertainty')
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(val_uncertainty, cmap='viridis', vmin=min_uncertainty, vmax=max_uncertainty)
    axs[1].set_title('Validation Uncertainty')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # Render the figure into a numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buffer = fig.canvas.tostring_argb()
    image = np.frombuffer(buffer, dtype=np.uint8)
    image.shape = (h, w, 4) 
    image = image[:, :, 1:4]

    plt.close(fig)

    return image

def create_depth_comparison_postprocess(model: torch.nn.Module, post_processor, image: torch.Tensor, gt: torch.Tensor, device: str) -> np.ndarray:
    model.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through the model
        pred, std = model(image)
    
    target_shape = gt.shape

    # Perform post-processing.
    outputs = post_processor(
        pred.clone(),
        inputs=image,
        resize_to=target_shape[-2:],
        raw_outputs=pred.clone(),
        std=std,
    )

    pred = nn.functional.interpolate(
        pred,
        size=(426, 560),  # Original input dimensions
        mode='bilinear',
        align_corners=True
    )
    outputs = nn.functional.interpolate(
        outputs,
        size=(426, 560),  # Original input dimensions
        mode='bilinear',
        align_corners=True
    )

    # Convert tensors to numpy arrays for visualization
    pred = pred.squeeze().cpu().numpy()
    outputs = outputs.squeeze().cpu().numpy()

    min_depth = gt.min()
    max_depth = gt.max()

    # Create the figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(pred, cmap='plasma', vmin=min_depth, vmax=max_depth)
    axs[0].set_title('Predicted Depth')

    axs[0].axis('off')
    axs[1].imshow(outputs, cmap='plasma', vmin=min_depth, vmax=max_depth)
    axs[1].set_title('Postprocess Depth')
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

def create_depth_postprocess_gt_comparison(model: torch.nn.Module, post_processor, image: torch.Tensor, gt: torch.Tensor, device: str) -> np.ndarray:
    model.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through the model
        pred, std = model(image)

    # Perform post-processing.
    outputs = post_processor(
        pred,
        inputs=image,
        resize_to=gt.shape[-2:],
        raw_outputs=pred,
        std=std,
    )

    outputs = nn.functional.interpolate(
        outputs,
        size=(426, 560),  # Original input dimensions
        mode='bilinear',
        align_corners=True
    )

    # Convert tensors to numpy arrays for visualization
    outputs = outputs.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()

    min_depth = gt.min()
    max_depth = gt.max()
    
    # Create the figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(outputs, cmap='plasma', vmin=min_depth, vmax=max_depth)
    axs[0].set_title('Postprocessed Depth')

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