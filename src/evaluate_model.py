import os
import argparse
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from transformers import AutoImageProcessor
import torchvision.transforms as T


from utils import ensure_dir, load_config, target_transform, create_depth_comparison, gradient_regularizer, gradient_loss
from modules import SimpleUNet, UNetWithDinoV2Backbone, UncertaintyDepthAnything
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
            raw_outputs, std = model(inputs)

            # Perform post-processing.
            outputs = post_processor(
                raw_outputs,
                inputs=inputs,
                resize_to=target_shape[-2:],
                raw_outputs=raw_outputs,
                std=std,
            )

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
                    raw_output_np = raw_outputs[i].cpu().squeeze().numpy()
                    output_np = outputs[i].cpu().squeeze().numpy()
                    std_np = std[i].cpu().squeeze().numpy()

                    # Assume img1 and img2 are (H, W) or (H, W, C), normalized to [0, 1]
                    diff = np.abs(raw_output_np - output_np)
                    diff_gray = diff if diff.ndim == 2 else diff.mean(axis=2)

                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)

                    # Create visualization
                    fig = plt.figure(figsize=(15, 5))

                    all_data = np.concatenate([target_np.flatten(), raw_output_np.flatten(), output_np.flatten()])
                    vmin, vmax = all_data.min(), all_data.max()

                    plt.subplot(3, 2, 1)
                    plt.imshow(input_np)
                    plt.title("RGB Input")
                    plt.axis('off')

                    plt.subplot(3, 2, 2)
                    im = plt.imshow(target_np, cmap='plasma', vmin=vmin, vmax=vmax)
                    plt.title("Ground Truth Depth")
                    plt.colorbar(im, fraction=0.046, pad=0.04)
                    plt.axis('off')

                    plt.subplot(3, 2, 3)
                    plt.imshow(raw_output_np, cmap='plasma', vmin=vmin, vmax=vmax)
                    plt.title("Raw Prediction")
                    plt.axis('off')

                    plt.subplot(3, 2, 4)
                    plt.imshow(output_np, cmap='plasma', vmin=vmin, vmax=vmax)
                    plt.title("Post processed Prediction")
                    plt.axis('off')


                    plt.subplot(3, 2, 5)
                    plt.imshow(diff_gray, cmap='hot')
                    plt.colorbar(label="Absolute Difference")
                    plt.title("Difference between raw and post processed Heatmap")
                    plt.axis('off')
                    plt.show()

                    plt.subplot(3, 2, 6)
                    im = plt.imshow(std_np, cmap='plasma')
                    plt.title("Std dev of model")
                    plt.colorbar(im, fraction=0.046, pad=0.04)
                    plt.axis('off')

                    # Dump a post processor text to the bottom of the plot.
                    fig.text(0.5, 0.01, str(post_processor), ha='center', fontsize=10)

                    # plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"sample_{idx}.png"))
                    plt.close()

            # Free up memory
            del inputs, targets, raw_outputs, outputs, std, abs_diff, max_ratio

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

    wandb_artifact_fullname = config["model"].get("wandb_artifact_fullname")
    assert wandb_artifact_fullname is not None, "Specify `config.wandb_artifact_fullname` to tell this script what to download."

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

    wandb.init(
        entity=config["logging"]["entity"],
        project=config["logging"]["project_name"],
        name=config['logging']["run_name"],
        config=config,
        mode="online" if config["logging"]["log"] else "disabled",
        group="evaluation",
        save_code=True,
    )

    api = wandb.Api()

    # ##### LOAD THE MODEL #####

    # I'd recomment to set this, as some artifacts are quite large and can fill up your home dir (max 20GB)
    os.environ["WANDB_ARTIFACT_DIR"] = f"/work/scratch/{os.getenv('USER')}/.artifacts"

    print("Downloading model artifact")
    model_artifact = api.artifact(wandb_artifact_fullname)
    model_artifact_dir = model_artifact.download()
    print("Model artifact downloaded to", model_artifact_dir)

    # V5 model has 16 heads :)
    if config["model"]["type"] == "u_net":
        model = SimpleUNet(
            hidden_channels=config["model"]["hidden_channels"],
            dilation=config["model"]["dilation"],
            num_heads=config["model"]["num_heads"],
            conv_transpose=config["model"]["conv_transpose"]
        )
    elif config["model"]["type"] == "depth_anything":
        model = UncertaintyDepthAnything(
            num_heads=config["model"]["num_heads"],
            include_pretrained_head=config["model"]["include_pretrained_head"]
        )
    elif config["model"]["type"] == "dinov2_backboned_unet":
        model = UNetWithDinoV2Backbone(
            num_heads=config["model"]["num_heads"],
            image_size=(config["data"]["input_size"][0], config["data"]["input_size"][1]),
            conv_transpose=config["model"]["conv_transpose"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    state_dict = torch.load(os.path.join(model_artifact_dir, 'best_model.pth'))

    # I don't really know why this is necessary, but will look into it later
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     if key.startswith('module.'):
    #         new_key = key[7:]  # Remove 'module.' prefix (7 characters)
    #     else:
    #         new_key = key
    #     new_state_dict[new_key] = value

    model.load_state_dict(state_dict)

    # ##### LOAD THE DATASET #####
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
    elif config["model"]["type"] == "dinov2_backboned_unet":
        train_transform = transforms.Compose([
            transforms.Resize((426, 560)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((426, 560)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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

    # ONLY DO 32 indices to try out different filtering :)
    # val_dataset = Subset(val_dataset, indices=range(128))
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

    # Log metrics to Weights and Biases
    wandb.run.summary.update({
        "Metrics/MAE": metrics['MAE'],
        "Metrics/RMSE": metrics['RMSE'],
        "Metrics/siRMSE": metrics['siRMSE'],
        "Metrics/REL": metrics['REL'],
        "Metrics/Delta1": metrics['Delta1'],
        "Metrics/Delta2": metrics['Delta2'],
        "Metrics/Delta3": metrics['Delta3']
    })

    wandb.finish()
