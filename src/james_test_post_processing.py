import os
import argparse
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from utils import ensure_dir, load_config, target_transform, create_depth_comparison, gradient_regularizer, gradient_loss
from modules import SimpleUNet, UncertaintyDepthAnything
from data import DepthDataset
from train import generate_test_predictions
from post_process import PostProcessor


def evaluate_preds_with_post_processing(post_processor, val_loader, device, output_dir):
    """Evaluate the model and compute metrics on validation set, with post-process pipeline."""
    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0

    total_samples = 0
    target_shape = None

    with torch.no_grad():
        for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            if target_shape is None:
                target_shape = targets.shape

            # Forward pass
            # outputs = model(inputs)

            # Perform post-processing.
            outputs = post_processor(outputs, inputs)

            # Calculate metrics
            abs_diff = torch.abs(outputs - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()

            # Calculate scale-invariant RMSE for each image in the batch
            for i in range(batch_size):
                # Convert tensors to numpy arrays
                pred_np = outputs[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()

                EPSILON = 1e-6

                valid_target = target_np > EPSILON
                if not np.any(valid_target):
                    continue

                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]

                log_target = np.log(target_valid)

                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)

                # Calculate scale-invariant error
                diff = log_pred - log_target
                diff_mean = np.mean(diff)

                # Calculate RMSE for this image
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))

            # Calculate thresholded accuracy
            max_ratio = torch.max(outputs / (targets + 1e-6), targets / (outputs + 1e-6))
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()
            # Free up memory
            del inputs, targets, outputs, abs_diff, max_ratio

        # Clear CUDA cache
        torch.cuda.empty_cache()

    # Calculate final metrics using stored target shape
    total_pixels = target_shape[1] * target_shape[2] * target_shape[3]  # channels * height * width
    mae /= total_samples * total_pixels
    rmse = np.sqrt(rmse / (total_samples * total_pixels))
    rel /= total_samples * total_pixels
    sirmse = sirmse / total_samples
    delta1 /= total_samples * total_pixels
    delta2 /= total_samples * total_pixels
    delta3 /= total_samples * total_pixels

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'siRMSE': sirmse,
        'REL': rel,
        'Delta1': delta1,
        'Delta2': delta2,
        'Delta3': delta3
    }

    return metrics


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    api = wandb.Api()

    # I'd recomment to set this, as some artifacts are quite large and can fill up your home dir (max 20GB)
    os.environ["WANDB_ARTIFACT_DIR"] = f"/work/scratch/{os.getenv('USER')}/.artifacts"


    print("Downloading model artifact")
    model_artifact = api.artifact("MonocularDepthEstimation/MonocularDepthEstimation/best_model:v5")
    model_artifact_dir = model_artifact.download()
    print("Model artifact downloaded to", model_artifact_dir)

    # V5 model has 16 heads :)
    model = UncertaintyDepthAnything(num_heads=16)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    state_dict = torch.load(os.path.join(model_artifact_dir, 'best_model.pth'))

    # I don't really know why this is necessary, but will look into it later
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix (7 characters)
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    # How do we modify the number of heads in the backbone?
