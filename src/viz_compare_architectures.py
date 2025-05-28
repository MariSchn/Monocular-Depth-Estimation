import glob
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
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import yaml
from transformers import AutoImageProcessor
import torchvision.transforms as T


from config import load_config_from_yaml
from utils import ensure_dir, load_config, target_transform, create_depth_comparison, gradient_regularizer, gradient_loss
from modules import SimpleUNet, UNetWithDinoV2Backbone, UncertaintyDepthAnything
from data import DepthDataset
from train import generate_test_predictions

def compare_models_on_sample(raw_outputs_per_model, inputs, target, output_dir: str, img_idx: int, model_names: List[str]):
    CMAP = "plasma"
    MAX_COLS = 4  # max number of columns before wrapping

    input_np = inputs.cpu().permute(1, 2, 0).numpy()
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)

    # Total number of images: raw image + ground truth depth + one per model
    num_images = 2 + len(raw_outputs_per_model)

    # Compute rows and columns needed
    n_cols = min(num_images, MAX_COLS)
    n_rows = math.ceil(num_images / MAX_COLS)

    # Figure size scales with number of columns and rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Flatten axs in case it's 2D
    axs = np.array(axs).reshape(-1)
    target_np = target.cpu().squeeze().numpy()

    # Gather all data to compute consistent color scale
    # Plot Raw image
    im = axs[0].imshow(input_np)
    # axs[0].set_title("Ground Truth Depth")
    axs[0].text(0.5, -0.1, "Raw Input", transform=axs[0].transAxes, ha='center', fontsize=10)
    axs[0].axis('off')

    output_np_per_model = [raw_output.cpu().squeeze().numpy() for raw_output in raw_outputs_per_model]
    all_data = np.concatenate([
        target_np.flatten(),
    ] + [o.flatten() for o in output_np_per_model])
    vmin, vmax = all_data.min(), all_data.max()

    # Plot ground truth
    im = axs[1].imshow(target_np, cmap=CMAP, vmin=vmin, vmax=vmax)
    axs[1].text(0.5, -0.1, "Ground Truth Depth", transform=axs[1].transAxes, ha='center', fontsize=10)
    axs[1].axis('off')
    # Create a divider for the existing axes
    divider = make_axes_locatable(axs[1])
    # Append a colorbar axes to the right of the image axes with the same height
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # Create the colorbar in the new axes
    fig.colorbar(im, cax=cax)

    # Plot raw Prediction for each model
    for i, output_np in enumerate(output_np_per_model):
        ax = axs[2 + i]
        im = ax.imshow(output_np, cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.text(0.5, -0.1, model_names[i], transform=ax.transAxes, ha='center', fontsize=10)
        ax.axis('off')
        # Create a divider for the existing axes
        divider = make_axes_locatable(ax)
        # Append a colorbar axes to the right of the image axes with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # Create the colorbar in the new axes
        fig.colorbar(im, cax=cax)

    # Turn off any unused axes (e.g., if grid is bigger than image count)
    for j in range(2 + len(output_np_per_model), len(axs)):
        axs[j].axis('off')

    # # Shared colorbar (linked to last plotted image)
    # divider = make_axes_locatable(axs[2 + len(output_np_per_model) - 1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = fig.colorbar(im, cax=cax)
    # fig.colorbar(im, ax=axs[-(n_rows * n_cols - num_images + 1)], fraction=0.02, pad=0.04)

    # Save or display
    # fig.suptitle("Architecture Prediction Comparison", fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fuck_u_spiderman_{img_idx}.png"), bbox_inches='tight', pad_inches=0.05)
    plt.close()



def compare_models_on_validation_set(models, val_loader, device, output_dir, configs):
    """Visualizing post processsing on model predictions."""
    for model in models:
        model.eval()

    model_names = [conf["logging"]["run_name"] for conf in configs]

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
            raw_outputs_list = [model(inputs) for model in models]

            # Save some sample predictions
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 1)):
                    idx = total_samples - batch_size + i
                    raw_output_on_sample = [r[i] for r, _ in raw_outputs_list]
                    compare_models_on_sample(raw_output_on_sample, inputs[i], targets[i], output_dir, idx, model_names)

            # Free up memory
            for outputs in raw_outputs_list:
                del outputs
            del inputs, targets

        # Clear CUDA cache
        torch.cuda.empty_cache()

    return None


if __name__ == "__main__":
    mpl.rcParams.update({
        'axes.formatter.use_mathtext': True,
        'font.family': 'serif',
        'font.serif': ['cmr10'],  # LaTeX default serif font
    })

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default="./configs/evaluations/architecture", help="Directory from which to search for configs")
    args = parser.parse_args()

    config_paths = glob.glob(os.path.join(args.config_dir, "*.yml"))
    print(f"Found {len(config_paths)} configs.")

    for config_path in config_paths:
        print(config_path)
    configs = [load_config_from_yaml(conf_path) for conf_path in config_paths]
    assert len(configs) > 0, "Please specify a directory with at least one config *.yml file"
    data_config = configs[0]

    # wandb_artifact_fullname = config.model.wandb_artifact_fullname
    # assert wandb_artifact_fullname is not None, "Specify `config.wandb_artifact_fullname` to tell this script what to download."

    eval_results_dir = os.path.join(data_config["output_dir"], f"model_compare_results")
    ensure_dir(eval_results_dir)

    # Define paths
    train_data_dir = os.path.join(data_config["data_dir"], "train")
    test_data_dir = os.path.join(data_config["data_dir"], "test")

    train_list_file = os.path.join(data_config["data_dir"], "train_list.txt")
    test_list_file = os.path.join(data_config["data_dir"], "test_list.txt")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    api = wandb.Api()

    # ##### LOAD THE MODEL #####

    # I'd recomment to set this, as some artifacts are quite large and can fill up your home dir (max 20GB)
    os.environ["WANDB_ARTIFACT_DIR"] = f"/work/scratch/{os.getenv('USER')}/.artifacts"

    models = []
    for conf in configs:
        wandb_artifact_fullname = conf.model.wandb_artifact_fullname
        assert len(wandb_artifact_fullname) > 0, "Specify `config.wandb_artifact_fullname` for conf {} to tell this script what to download.".format(conf.logging.run_name)

        print("Downloading model artifact")
        model_artifact = api.artifact(wandb_artifact_fullname)
        model_artifact_dir = model_artifact.download()
        print("Model artifact downloaded to", model_artifact_dir)

        # V5 model has 16 heads :)
        if conf["model"]["type"] == "u_net":
            model = SimpleUNet(
                hidden_channels=conf["model"]["hidden_channels"],
                dilation=conf["model"]["dilation"],
                num_heads=conf["model"]["num_heads"],
                conv_transpose=conf["model"]["conv_transpose"],
                weight_initialization=conf["model"]["weight_initialization"],
                depth_before_aggregate=conf["model"]["depth_before_aggregate"],
            )
        elif conf["model"]["type"] == "depth_anything":
            model = UncertaintyDepthAnything(
                num_heads=conf["model"]["num_heads"],
                include_pretrained_head=conf["model"]["include_pretrained_head"],
                weight_initialization=conf["model"]["weight_initialization"],
            )
        elif conf["model"]["type"] == "dinov2_backboned_unet":
            model = UNetWithDinoV2Backbone(
                num_heads=conf["model"]["num_heads"],
                image_size=(conf["data"]["input_size"][0], conf["data"]["input_size"][1]),
                conv_transpose=conf["model"]["conv_transpose"],
                weight_initialization=conf["model"]["weight_initialization"],
                depth_before_aggregate=conf["model"]["depth_before_aggregate"],
            )
        else:
            raise ValueError(f"Unknown model type: {conf['model']['type']}")

        model = nn.DataParallel(model)
        model = model.to(DEVICE)
        print(f"Using device: {DEVICE}")

        state_dict = torch.load(os.path.join(model_artifact_dir, 'best_model.pth'))
        model.load_state_dict(state_dict)
        models.append(model)

    # ##### LOAD THE DATASET #####
    # Define transforms
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
    # elif config["model"]["type"] == "depth_anything":
    #     train_transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    #     test_transform = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    # elif config["model"]["type"] == "dinov2_backboned_unet":
    #     train_transform = transforms.Compose([
    #         transforms.Resize((426, 560)),
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])

    #     test_transform = transforms.Compose([
    #         transforms.Resize((426, 560)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])

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
    torch.manual_seed(configs[0]["seed"])

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(configs[0]["seed"])
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
    val_dataset = Subset(val_dataset, range(configs[0]["train"]["batch_size"] * 2))

    # Get images to be used as visualizations duringn logging
    train_log_rgb, train_log_depth, _ = train_dataset[0]
    val_log_rgb, val_log_depth, _ = val_dataset[0]

    val_loader = DataLoader(
        val_dataset,
        batch_size=configs[0]["train"]["batch_size"],
        shuffle=False,
        num_workers=configs[0]["data"]["num_workers"],
        pin_memory=configs[0]["data"]["pin_memory"],
    )

    # ##### TIME TO EVALUATE THE MODEL >:) #####

    # Evaluate the model on validation set
    print("Evaluating model on validation set to produce comparison images")
    compare_models_on_validation_set(models, val_loader, DEVICE, eval_results_dir, configs)
