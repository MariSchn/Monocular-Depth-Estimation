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
import tempfile
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModel

from utils import *
from modules import SimpleUNet, UncertaintyDepthAnything, UNetWithResNet50Backbone
from data import DepthDataset
from create_prediction_csv import process_depth_maps


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, output_dir):
    """Train the model and save the best based on validation metrics"""
    best_val_loss = float('inf')
    best_epoch = 0
    epoch_head_penalty = 0.0
    epoch_grad_loss = 0.0
    epoch_grad_reg = 0.0
    epoch_mean_uncertainty = 0.0
    train_losses = []
    val_losses = []
        
    total_num_steps = num_epochs * len(train_loader)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0

        step = epoch * len(train_loader)
        for inputs, targets, _ in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, std = model(inputs)
            epoch_mean_uncertainty += std.mean().item() * inputs.size(0)

            loss = criterion(outputs, targets)

            if config["model"]["num_heads"] > 1 and config["train"]["head_penalty_weight"][0] > 0:
                head_penalty = model.module.head_penalty() if hasattr(model, 'module') else model.head_penalty()
                head_penalty_weight = np.interp(step, [0, total_num_steps], [config["train"]["head_penalty_weight"][0], config["train"]["head_penalty_weight"][1]])
                loss += head_penalty_weight * head_penalty
                epoch_head_penalty += head_penalty.item() * inputs.size(0)
            if config["train"]["gradient_loss_weight"] > 0:
                grad_loss = gradient_loss(outputs, targets)
                loss += config["train"]["gradient_loss_weight"] * grad_loss
                epoch_grad_loss += grad_loss.item() * inputs.size(0)
            if config["train"]["gradient_regularizer_weight"] > 0:
                grad_reg = gradient_regularizer(outputs)
                loss += config["train"]["gradient_regularizer_weight"] * grad_reg
                epoch_grad_reg += grad_reg.item() * inputs.size(0)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

            # Step-level logging
            if config["logging"]["log_every_k_steps"] > 0 and step % config["logging"]["log_every_k_steps"] == 0:
                log_dict = {
                    "Step/Train Loss": loss.item(),
                    "Step/Step": step,
                    "Step/Epoch": epoch,
                }

                # Log comparisons
                if config["logging"]["log_images"]:
                    train_comparison = create_depth_comparison(model, train_log_rgb, train_log_depth, device)
                    val_comparison = create_depth_comparison(model, val_log_rgb, val_log_depth, device)

                    log_dict.update({
                        "Step/Train Depth Comparison": wandb.Image(train_comparison),
                        "Step/Validation Depth Comparison": wandb.Image(val_comparison),
                    })

                # Log validation loss
                if config["logging"]["val_every_step_log"]:
                    model.eval()
                    step_val_loss = 0.0

                    with torch.no_grad():
                        for inputs_, targets_, _ in val_loader:
                            inputs_, targets_ = inputs_.to(device), targets_.to(device)
                            
                            # Forward pass
                            outputs_, _ = model(inputs_)
                            loss_ = criterion(outputs_, targets_)

                            step_val_loss += loss_.item() * inputs_.size(0)

                    log_dict["Step/Validation Loss"] = step_val_loss / len(val_loader.dataset)
                    model.train()

                wandb.log(log_dict)

            step += 1
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        epoch_head_penalty /= len(train_loader.dataset)
        epoch_grad_loss /= len(train_loader.dataset)
        epoch_grad_reg /= len(train_loader.dataset)
        epoch_mean_uncertainty /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs, std = model(inputs)

                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
                
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Epoch-level logging
        if config["logging"]["log_every_k_epochs"] > 0 and (epoch + 1) % config["logging"]["log_every_k_epochs"] == 0:
            log_dict = {
                "Epoch/Train Loss": train_loss,
                "Epoch/Validation Loss": val_loss,
                "Epoch/Train Head Penalty": epoch_head_penalty,
                "Epoch/Train Gradient Loss": epoch_grad_loss,
                "Epoch/Train Gradient Regularizer": epoch_grad_reg,
                "Epoch/Train Mean Uncertainty": epoch_mean_uncertainty,
            }

            if config["logging"]["log_images"]:
                # Log comparisons
                train_comparison = create_depth_comparison(model, train_log_rgb, train_log_depth, device)
                val_comparison = create_depth_comparison(model, val_log_rgb, val_log_depth, device)

                uncertainty_visualization = create_uncertainty_visualization(model, train_log_rgb, val_log_rgb, device)

                log_dict.update({
                    "Epoch/Train Depth Comparison": wandb.Image(train_comparison),
                    "Epoch/Validation Depth Comparison": wandb.Image(val_comparison),
                    "Epoch/Uncertainty Visualization": wandb.Image(uncertainty_visualization),
                })

            wandb.log(log_dict)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
    
    print(f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")

    # Save the best model
    if config["train"]["num_epochs"] > 0:
        if config["logging"]["upload_to_wandb"]:
            best_model_artifact = wandb.Artifact(name="best_model", type="model", description=f"Best model at epoch {best_epoch+1}")
            best_model_artifact.add_file(os.path.join(output_dir, 'best_model.pth'), name="best_model.pth")
            wandb.log_artifact(best_model_artifact, aliases=[config["logging"]["run_name"].replace(" ", "_").replace("/", "").replace(":", "").lower()])
        
        # Load the best model
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    
    return model

def evaluate_model(model, val_loader, device, output_dir):
    """Evaluate the model and compute metrics on validation set"""
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

    with tempfile.TemporaryDirectory() as temp_dir:
        with torch.no_grad():
            for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating"):
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                if target_shape is None:
                    target_shape = targets.shape
                
                # Forward pass
                outputs, std = model(inputs)
                
                # Resize outputs to match target dimensions
                outputs = nn.functional.interpolate(
                    outputs,
                    size=targets.shape[-2:],  # Match height and width of targets
                    mode='bilinear',
                    align_corners=True
                )

                # Save predictions into temp_dir
                for i in range(batch_size):
                    filename = filenames[i]
                    depth_pred = outputs[i].cpu().squeeze().numpy()
                    
                    # Save depth map prediction as numpy array
                    np.save(os.path.join(temp_dir, f"{filename}"), depth_pred)
                
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
        
        # Upload validation images to Weights and Biases
        if config["logging"]["log"] and config["logging"]["upload_to_wandb"]:
            val_artifact = wandb.Artifact(name=f"validation_images", type="predictions", description="Best model validation depth predictions")
            val_artifact.add_dir(temp_dir)
            wandb.log_artifact(val_artifact, aliases=[config["logging"]["run_name"].replace(" ", "_").replace("/", "").replace(":", "").lower()])
            val_artifact.wait()

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

def generate_test_predictions(model, test_loader, device, output_dir):
    """Generate predictions for the test set without ground truth"""
    model.eval()
    
    # Ensure predictions directory exists
    ensure_dir(output_dir)
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # Forward pass
            outputs, std = model(inputs)
            
            # Resize outputs to match original input dimensions (426x560)
            outputs = nn.functional.interpolate(
                outputs,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )
            
            # Save all test predictions
            for i in range(batch_size):
                # Get filename without extension
                filename = filenames[i].split(' ')[1]
                
                # Save depth map prediction as numpy array
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(output_dir, f"{filename}"), depth_pred)
            
            # Clean up memory
            del inputs, outputs
        
        # Clear cache after test predictions
        torch.cuda.empty_cache()

        # Upload test images to Weights and Biases
        if config["logging"]["log"] and config["logging"]["upload_to_wandb"]:
            test_artifact = wandb.Artifact(name="test_images", type="predictions", description="Test depth predictions")
            test_artifact.add_dir(output_dir)
            wandb.log_artifact(test_artifact, aliases=[config["logging"]["run_name"].replace(" ", "_").replace("/", "").replace(":", "").lower()])
            test_artifact.wait()

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
    if config["model"]["type"] == "u_net":
        train_transform = transforms.Compose([
            transforms.Resize((426, 560)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((426, 560)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif config["model"]["type"] == "depth_anything":
        train_transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
        test_transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")

    
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

    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True, 
        num_workers=config["data"]["num_workers"], 
        pin_memory=config["data"]["pin_memory"],
        drop_last=True,
        persistent_workers=True
    )
    
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
    
    
    if config["model"]["type"] == "u_net":
        model = SimpleUNet(hidden_channels=config["model"]["hidden_channels"], dilation=config["model"]["dilation"], num_heads=config["model"]["num_heads"])
    elif config["model"]["type"] == "depth_anything":
        model = UncertaintyDepthAnything(num_heads=config["model"]["num_heads"], include_pretrained_head=config["model"]["include_pretrained_head"])
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
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
    wandb.init(
        entity=config["logging"]["entity"],
        project=config["logging"]["project_name"],
        name=config['logging']["run_name"],
        config=config,
        mode="online" if config["logging"]["log"] else "disabled",
        save_code=True,
    )
    
    # Train the model
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, config["train"]["num_epochs"], DEVICE, results_dir)
            
    # Evaluate the model on validation set
    print("Evaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, DEVICE, results_dir)

    # Log metrics to Weights and Biases
    wandb.log({
        "Metrics/MAE": metrics['MAE'],
        "Metrics/RMSE": metrics['RMSE'],
        "Metrics/siRMSE": metrics['siRMSE'],
        "Metrics/REL": metrics['REL'],
        "Metrics/Delta1": metrics['Delta1'],
        "Metrics/Delta2": metrics['Delta2'],
        "Metrics/Delta3": metrics['Delta3']
    })
    
    # Print metrics
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Save metrics to file
    with open(os.path.join(results_dir, 'validation_metrics.txt'), 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    
    # Generate predictions for the test set
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE, predictions_dir)
    
    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")

    # Process depth maps and save to CSV
    print("Processing depth maps and saving to CSV...")
    process_depth_maps(test_list_file, predictions_dir, os.path.join(predictions_dir, 'predictions.csv'))

    if config["logging"]["log"] and config["logging"]["upload_to_wandb"]:
        csv_artifact = wandb.Artifact(name="predictions_csv", type="submission", description="Test depth predictions CSV")
        csv_artifact.add_file(os.path.join(predictions_dir, 'predictions.csv'), name="predictions.csv")
        wandb.log_artifact(csv_artifact, aliases=[config["logging"]["run_name"].replace(" ", "_").replace("/", "").replace(":", "").lower()])
        csv_artifact.wait()

    wandb.finish()