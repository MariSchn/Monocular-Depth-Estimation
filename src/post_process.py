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

from utils import ensure_dir, load_config, target_transform, create_depth_comparison, gradient_regularizer, gradient_loss
from modules import SimpleUNet
from data import DepthDataset

from guided_filter_pytorch.guided_filter import GuidedFilter

class DepthPostProcessor:
    def __init__(self, resize_to=None):
        self.resize_to = resize_to  # (H, W) or None

        # TODO: specify the r and eps parameters with the configuration
        self.filter = GuidedFilter(3, 1e-3)

    def __call__(self, outputs, inputs):
        # Resize if needed
        if self.resize_to is not None:
            out = nn.functional.interpolate(
                outputs, size=self.resize_to, mode='bilinear', align_corners=False
            )

        # Use the original RGB inputs as a guide to perform smoothing.
        filtered = self.filter(out, inputs)
        # Since the guide image is a color image, we may replicate the smoothed depth across the 3 channels.
        filtered = filtered.mean(dim=1, keepdim=True)

        return filtered


def evaluate_model_with_postprocess(model, val_loader, device, output_dir):
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
            # Instantiate a post processor for the inputs.
            post_process = DepthPostProcessor(resize_to=targets.shape[-2:])
            
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            if target_shape is None:
                target_shape = targets.shape
            

            # Forward pass
            outputs = model(inputs)

            # Perform post-processing.
            processed_outputs = post_process(outputs, inputs)

            i = 0
            # Convert tensors to numpy arrays
            input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
            output_np = outputs[i].cpu().squeeze().numpy()
            processed_np = processed_outputs[i].cpu().squeeze().numpy()
            # print("output.shape", output_np.shape, "processed.shape", processed_np.shape)
            
            # Normalize for visualization
            input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(input_np)
            plt.title("RGB Input")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(output_np, cmap='plasma')
            plt.title("Predicted Depth")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(processed_np, cmap='plasma')
            plt.title("Processed Predicted Depth")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"processed_comp.png"))
            plt.close()
            
            assert False
            
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
                    
                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)
                    
                    # Create visualization
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(input_np)
                    plt.title("RGB Input")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(target_np, cmap='plasma')
                    plt.title("Ground Truth Depth")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(output_np, cmap='plasma')
                    plt.title("Predicted Depth")
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

    # Define paths
    train_data_dir = os.path.join(config["data_dir"], "train")
    test_data_dir = os.path.join(config["data_dir"], "test")

    train_list_file = os.path.join(config["data_dir"], "train_list.txt")
    test_list_file = os.path.join(config["data_dir"], "test_list.txt")

    results_dir = os.path.join(config["output_dir"], f"{config['logging']['run_name']}_results")
    predictions_dir = os.path.join(config["output_dir"], f"{config['logging']['run_name']}_predictions")

    # Create output directories
    ensure_dir(results_dir)
    ensure_dir(predictions_dir)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(config["data"]["input_size"]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(config["data"]["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create training dataset with ground truth
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
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Set device
    if torch.cuda.is_available() and config["device"] == "cuda":
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # Clear CUDA cache before model initialization
    torch.cuda.empty_cache()
    
    # Display GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    model = SimpleUNet(hidden_channels=config["model"]["hidden_channels"], dilation=config["model"]["dilation"])
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    # Print memory usage after model initialization
    if torch.cuda.is_available():
        print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    
    # Initialize Weights and Biases for logging
    config.update({
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    })
    
    # Load the best model from the results directory. This assumes you
    # already trained the model.
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    
    # Evaluate the model on validation set
    print("Evaluating model on validation set with post-processing...")
    metrics = evaluate_model_with_postprocess(model, val_loader, DEVICE, results_dir)

    # Print metrics
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    assert False
    
    # Save metrics to file
    with open(os.path.join(results_dir, 'validation_metrics.txt'), 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    
    # Generate predictions for the test set
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE, predictions_dir)
    
    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")
