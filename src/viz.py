import hashlib
import math
import os
import argparse
from typing import List
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


from config import load_config_from_yaml
from utils import ensure_dir, load_config, target_transform, create_depth_comparison, gradient_regularizer, gradient_loss
from modules import SimpleUNet, UNetWithDinoV2Backbone, UncertaintyDepthAnything
from data import DepthDataset
from train import generate_test_predictions
from post_process import BoxFilter, GaussianBlurStep, GuidedFilteringStep, NormalizedStdInterpolation, PostProcessor, SigmoidStdInterpolation, load_post_processor_from_config
from guided_filter_pytorch.guided_filter import GuidedFilter


def visualize_all_post_processing_output(target_np, raw_output_np, output_np, output_dir: str, img_idx: int):
    CMAP = "plasma"
    MAX_COLS = 4  # max number of columns before wrapping

    # Total number of images: ground truth depth + Raw + one per post processor
    num_images = 2 + len(post_processors)

    # Compute rows and columns needed
    n_cols = min(num_images, MAX_COLS)
    n_rows = math.ceil(num_images / MAX_COLS)

    # Figure size scales with number of columns and rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), constrained_layout=True)

    # Flatten axs in case it's 2D
    axs = np.array(axs).reshape(-1)

    # Gather all data to compute consistent color scale
    all_data = np.concatenate([
        target_np.flatten(),
        raw_output_np.flatten(),
    ] + [o.flatten() for o in output_np])
    vmin, vmax = all_data.min(), all_data.max()

    # Plot Ground Truth
    im = axs[0].imshow(target_np, cmap=CMAP, vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground Truth Depth")
    axs[0].axis('off')

    # Plot Raw Prediction
    im = axs[1].imshow(raw_output_np, cmap=CMAP, vmin=vmin, vmax=vmax)
    axs[1].set_title("Raw Prediction")
    axs[1].axis('off')

    # Plot post-processed predictions
    for i, (processor, output) in enumerate(zip(post_processors, output_np)):
        ax = axs[2 + i]
        im = ax.imshow(output, cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(str(processor))
        ax.axis('off')

    # Turn off any unused axes (e.g., if grid is bigger than image count)
    for j in range(2 + len(post_processors), len(axs)):
        axs[j].axis('off')

    # Shared colorbar (linked to last plotted image)
    fig.colorbar(im, ax=axs[:2 + len(post_processors)], fraction=0.02, pad=0.04)

    # Save or display
    plt.savefig(os.path.join(output_dir, f"post_processing_table_{img_idx}.png"))
    plt.close()

class DummyConstantPostProcessor:
    def __call__(self, outputs, **kwargs):
        return torch.zeros_like(outputs)

    def __str__(self):
        return ""

def visualize_uncertainty_interpolation(target_np, raw_outputs, std, output_dir: str, img_idx: int):
    uncertainty_interpolations = [
        NormalizedStdInterpolation(),
        SigmoidStdInterpolation(),
    ]
    # This is a way to visualize the effect of the std interpolation.
    # We have the raw model output be all zeros.
    # We have the smooth model output be all ones.
    # We get a [0, 1] with all the interpolation values of the uncertainty interpolation.
    ones = torch.ones_like(raw_outputs)
    post_processors = [PostProcessor([DummyConstantPostProcessor(), u]) for u in uncertainty_interpolations]
    processed = [
        p(
            ones,
            raw_outputs=ones,
            std=std,
        ) for p in post_processors
    ]
    output_np = [outputs.cpu().squeeze().numpy() for outputs in processed]
    for o in output_np:
        print("uncertainty min and max", o.min(), o.max())
    std_np = std.cpu().squeeze().numpy()

    CMAP = "plasma"
    MAX_COLS = 4  # max number of columns before wrapping

    # Total number of images: ground truth depth + Raw uncertainty + one per uncertainty interp
    num_images = 2 + len(post_processors)

    # Compute rows and columns needed
    n_cols = min(num_images, MAX_COLS)
    n_rows = math.ceil(num_images / MAX_COLS)

    # Figure size scales with number of columns and rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), constrained_layout=True)

    # Flatten axs in case it's 2D
    axs = np.array(axs).reshape(-1)

    # Gather all data to compute consistent color scale
    # Plot Ground Truth
    im = axs[0].imshow(target_np, cmap=CMAP)
    axs[0].set_title("Ground Truth Depth")
    axs[0].axis('off')
    fig.colorbar(im, ax=axs[0], fraction=0.02, pad=0.04)

    # Plot Raw uncertainty map
    im = axs[1].imshow(std_np, cmap=CMAP)
    axs[1].set_title("Raw Uncertainty")
    axs[1].axis('off')
    fig.colorbar(im, ax=axs[1], fraction=0.02, pad=0.04)

    # Plot post-processed predictions
    for i, (processor, output) in enumerate(zip(post_processors, output_np)):
        ax = axs[2 + i]
        im = ax.imshow(output, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(str(processor))
        ax.axis('off')

    # # Turn off any unused axes (e.g., if grid is bigger than image count)
    for j in range(2 + len(post_processors), len(axs)):
        axs[j].axis('off')

    # # Shared colorbar (linked to last plotted image)
    fig.colorbar(im, ax=axs[:2 + len(post_processors)], fraction=0.02, pad=0.04)

    # Save or display
    plt.savefig(os.path.join(output_dir, f"uncertainty_vis_{img_idx}.png"))
    plt.close()


def visualize_post_processing(model, post_processors: List[PostProcessor], val_loader, device, output_dir):
    """Visualizing post processsing on model predictions."""
    model.eval()

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
            processed = [
                p(
                    raw_outputs,
                    inputs=inputs,
                    resize_to=target_shape[-2:],
                    raw_outputs=raw_outputs,
                    std=std,
                ) for p in post_processors
            ]

            # Save some sample predictions
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 1)):
                    idx = total_samples - batch_size + i

                    # Convert tensors to numpy arrays
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    raw_output_np = raw_outputs[i].cpu().squeeze().numpy()
                    output_np = [outputs[i].cpu().squeeze().numpy() for outputs in processed]

                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)

                    visualize_all_post_processing_output(target_np, raw_output_np, output_np, output_dir, idx)
                    visualize_uncertainty_interpolation(target_np, raw_outputs[i], std[i], output_dir, idx)

            # Free up memory
            del inputs, targets, raw_outputs, processed, std

        # Clear CUDA cache
        torch.cuda.empty_cache()

    return None


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default.yml", help="The config file from which to load the hyperparameters")
    args = parser.parse_args()

    # Load config file
    config = load_config_from_yaml(args.config)

    wandb_artifact_fullname = config.model.wandb_artifact_fullname
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
            conv_transpose=config["model"]["conv_transpose"],
            weight_initialization=config["model"]["weight_initialization"],
            depth_before_aggregate=config["model"]["depth_before_aggregate"],
        )
    elif config["model"]["type"] == "depth_anything":
        model = UncertaintyDepthAnything(
            num_heads=config["model"]["num_heads"],
            include_pretrained_head=config["model"]["include_pretrained_head"],
            weight_initialization=config["model"]["weight_initialization"],
        )
    elif config["model"]["type"] == "dinov2_backboned_unet":
        model = UNetWithDinoV2Backbone(
            num_heads=config["model"]["num_heads"],
            image_size=(config["data"]["input_size"][0], config["data"]["input_size"][1]),
            conv_transpose=config["model"]["conv_transpose"],
            weight_initialization=config["model"]["weight_initialization"],
            depth_before_aggregate=config["model"]["depth_before_aggregate"],
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    state_dict = torch.load(os.path.join(model_artifact_dir, 'best_model.pth'))
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
        train_full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config["seed"])
    )
    val_indices = val_dataset.indices
    val_hash = hashlib.sha256(torch.tensor(val_indices).numpy().tobytes()).hexdigest()
    print(f"Validation indices hash: {val_hash}")

    # Please do not comment this out unless you have a good reason to.
    # Models trained after Friday May 23 should upload a hash of the val dataset indices
    # to WandB. We verify it here to make sure we are evaluating
    # the model on the same validation set.
    # Moels will likely be better on data that they are trained on :)
    with open(os.path.join(model_artifact_dir, "val_dataset_indices_hash.txt")) as f:
        speal = """Hash of validation dataset indices not the same!
        This is bad for evaluation because the model may be evaluated on data it trained on.
        Indices might be out of order. Check the val dataset and the random seed."
        """
        assert val_hash == f.read().strip(), speal

    # Let's only work on a Subset of the validation dataset.
    val_dataset = Subset(val_dataset, range(config["train"]["batch_size"] * 2))

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
    post_processors = [
        PostProcessor([GuidedFilteringStep()]),
        PostProcessor([BoxFilter()]),
        PostProcessor([GaussianBlurStep()]),
        PostProcessor([BoxFilter()]),
        PostProcessor([BoxFilter()]),
    ]

    # Evaluate the model on validation set
    print("Evaluating model on validation set with post-processing...")
    metrics = visualize_post_processing(model, post_processors, val_loader, DEVICE, eval_results_dir)
