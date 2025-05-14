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
from transformers import AutoImageProcessor
import torchvision.transforms as T

from utils import ensure_dir, load_config, target_transform, create_depth_comparison, gradient_regularizer, gradient_loss
from modules import SimpleUNet, UncertaintyDepthAnything
from data import DepthDataset
from train import generate_test_predictions
from post_process import PostProcessor, load_post_processor_from_config
from guided_filter_pytorch.guided_filter import GuidedFilter

def evaluate_model_with_postprocess(model, post_processor, val_loader, device, output_dir):
    """Evaluate the model and compute metrics on validation set, with post-process pipeline."""
    model.eval()

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
            outputs, std = model(inputs)

            # Perform post-processing.
            smoothed = post_processor(outputs, inputs=inputs, resize_to=target_shape[-2:])

            # Interpolate between the post-process and the targets depending on the std deviation.
            # Normalize the std deviation
            for i in range(batch_size):
                std[i] = (std[i] - std[i].min()) / (std[i].max() - std[i].min() + 1e-6)

            outputs = (1 - std) * outputs + std * smoothed

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

            # Save some sample predictions
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 5)):
                    idx = total_samples - batch_size + i

                    # Convert tensors to numpy arrays
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    output_np = outputs[i].cpu().squeeze().numpy()
                    std_np = std[i].cpu().squeeze().numpy()

                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)

                    # Create visualization
                    plt.figure(figsize=(15, 5))

                    plt.subplot(2, 2, 1)
                    plt.imshow(input_np)
                    plt.title("RGB Input")
                    plt.axis('off')

                    plt.subplot(2, 2, 2)
                    plt.imshow(target_np, cmap='plasma')
                    plt.title("Ground Truth Depth")
                    plt.axis('off')

                    plt.subplot(2, 2, 3)
                    plt.imshow(output_np, cmap='plasma')
                    plt.title("Predicted Depth")
                    plt.axis('off')

                    # print("Min std dev", np.min(std, dim=-1))
                    plt.subplot(2, 2, 4)
                    im = plt.imshow(std_np, cmap='plasma')
                    plt.title("Std dev of model")
                    plt.colorbar(im, fraction=0.046, pad=0.04)
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"sample_{idx}.png"))
                    plt.close()

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
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default.yml", help="The config file from which to load the hyperparameters")
    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)

    assert config["model"]["type"] == "depth_anything"

    eval_results_dir = os.path.join(config["output_dir"], f"{config['logging']['run_name']}_eval_results")
    ensure_dir(eval_results_dir)

    # Define paths
    train_data_dir = os.path.join(config["data_dir"], "train")
    test_data_dir = os.path.join(config["data_dir"], "test")

    train_list_file = os.path.join(config["data_dir"], "train_list.txt")
    test_list_file = os.path.join(config["data_dir"], "test_list.txt")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    api = wandb.Api()

    # ##### LOAD THE MODEL #####

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

    # ##### LOAD THE DATASET #####

    train_transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    test_transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")

    # How do we modify the number of heads in the backbone?
    train_full_dataset = DepthDataset(
        data_dir=train_data_dir,
        list_file=train_list_file,
        transform=train_transform,
        target_transform=target_transform,
        has_gt=True
    )

    # Create test dataset without ground truth
    test_dataset = DepthDataset(
        data_dir=test_data_dir,
        list_file=test_list_file,
        transform=test_transform,
        has_gt=False  # Test set has no ground truth
    )

    # Split training dataset into train and validation
    total_size = len(train_full_dataset)
    train_size = int(0.85 * total_size)   # 85% for training
    val_size = total_size - train_size    # 15% for validation

    # Set a fixed random seed for reproducibility
    torch.manual_seed(config["seed"])

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )

    # Get images to be used as visualizations duringn logging
    train_log_rgb, train_log_depth, _ = train_dataset[0]
    val_log_rgb, val_log_depth, _ = val_dataset[0]

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )

    # ##### TIME TO EVALUATE THE MODEL >:) #####

    # Load the post processor based on the configuration.
    post_processor = load_post_processor_from_config(config)

    # Evaluate the model on validation set
    print("Evaluating model on validation set with post-processing...")
    metrics = evaluate_model_with_postprocess(model, post_processor, val_loader, DEVICE, eval_results_dir)

    # Print metrics
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

"""
More Heads baseline model
Validation Metrics:
MAE: 0.3918
RMSE: 0.5829
siRMSE: 0.1303
REL: 0.1541
Delta1: 0.8012
Delta2: 0.9611
Delta3: 0.9941

More heads w std deviation smoothing, guided filter r=3 eps = 0.001
FAKE!!! I inputted the ground truth
MAE: 0.1656
RMSE: 0.2889
siRMSE: 0.0539
REL: 0.0553
Delta1: 0.9784
Delta2: 0.9983
Delta3: 0.9996

Guided filter with r=3 eps=0.001
Validation Metrics:
MAE: 1.8631
RMSE: 2.1395
siRMSE: 4.4698
REL: 0.7179
Delta1: 0.3079
Delta2: 0.4279
Delta3: 0.5554

# GAUSSIAN KERNEL
MAE: 0.3916
RMSE: 0.5825
siRMSE: 0.1301
REL: 0.1540
Delta1: 0.8013
Delta2: 0.9611
Delta3: 0.9941
"""
