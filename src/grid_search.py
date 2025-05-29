import os
import glob
import yaml
import torch
import wandb
import torch.nn as nn
from evaluate_model import evaluate_model_with_postprocess
from utils import load_config, ensure_dir, create_depth_comparison_postprocess, create_depth_postprocess_gt_comparison
from post_process import load_post_processor_from_config
from modules import SimpleUNet, UNetWithDinoV2LargeBackbone, UNetWithDinoV2SmallBackbone, UncertaintyDepthAnything, DiUNet
from data import DepthDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoImageProcessor
from config import load_config_from_yaml
import argparse
from train import evaluate_model

# You may choose a different metric for evaluation
BEST_METRIC = "siRMSE"
CONFIG_DIR = "src/configs/postprocess_grid"
BASE_CONFIG_PATH = "src/configs/default.yml"

def build_model(config):
    model_type = config["model"]["type"]

    if model_type == "u_net":
        model = SimpleUNet(
            hidden_channels=config["model"]["hidden_channels"],
            dilation=config["model"]["dilation"],
            num_heads=config["model"]["num_heads"],
            conv_transpose=config["model"]["conv_transpose"]
        )

    elif model_type == "depth_anything":
        model = UncertaintyDepthAnything(
            num_heads=config["model"]["num_heads"],
            include_pretrained_head=config["model"]["include_pretrained_head"]
        )

    elif model_type == "dinov2_large_backboned_unet":
        model = UNetWithDinoV2LargeBackbone(
            num_heads=config["model"]["num_heads"],
            image_size=(config["data"]["input_size"][0], config["data"]["input_size"][1]),
            conv_transpose=config["model"]["conv_transpose"]
        )

    elif model_type == "dinov2_backboned_unet":
        model = UNetWithDinoV2SmallBackbone(
            num_heads=config["model"]["num_heads"],
            image_size=(config["data"]["input_size"][0], config["data"]["input_size"][1]),
            conv_transpose=config["model"]["conv_transpose"]
        )
    elif config["model"]["type"] == "diunet":
        model = DiUNet(
            num_heads=config["model"]["num_heads"],
            image_size=(config["data"]["input_size"][0], config["data"]["input_size"][1]),
            conv_transpose=config["model"]["conv_transpose"],
            weight_initialization=config["model"]["weight_initialization"],
            depth_before_aggregate=config["model"]["depth_before_aggregate"],
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return torch.nn.DataParallel(model)

def build_dataloader(config, subset_size=2048):
    val_transform = transforms.Compose([
        transforms.Resize((426, 560)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define paths
    train_data_dir = os.path.join(config["data_dir"], "train")
    train_list_file = os.path.join(config["data_dir"], "train_list.txt")

    train_full_dataset = DepthDataset(
        data_dir=train_data_dir,
        list_file=train_list_file,
        transform=val_transform,
        target_transform=None,
        has_gt=True,
    )

    # Split training dataset into train and validation
    total_size = len(train_full_dataset)
    train_size = int(0.85 * total_size)   # 85% for training
    val_size = total_size - train_size    # 15% for validation

    # Set a fixed random seed for reproducibility
    torch.manual_seed(config["seed"])

    _, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )
    val_log_rgb, val_log_depth, _ = val_dataset[0]

    # Use a subset to speed up evaluation
    if subset_size == "MAX" or subset_size > len(val_dataset):
        subset_size = len(val_dataset)
    val_dataset = Subset(val_dataset, list(range(subset_size)))

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"]
    )

    return val_log_rgb, val_log_depth, val_loader

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate models with post-processing configurations.")
    parser.add_argument("--config", type=str, required=True, help="Folder inside postprocess_grid to use for evaluation configs")
    args = parser.parse_args()

    config_dir = os.path.join(CONFIG_DIR, args.config)
    # print(CONFIG_DIR)
    config_paths = glob.glob(os.path.join(config_dir, "*.yml"))
    print(f"Found {len(config_paths)} configs.")

    best_config_path = None
    best_metric_value = float("inf")
    all_results = []


    for config_path in config_paths:
        print(f"\nEvaluating config: {config_path}")
        config = load_config_from_yaml(config_path)

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

        os.environ["WANDB_ARTIFACT_DIR"] = f"/work/scratch/{os.getenv('USER')}/.artifacts"

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        wandb_artifact_fullname = config.model.wandb_artifact_fullname
        assert wandb_artifact_fullname is not None, "Specify `config.wandb_artifact_fullname` to tell this script what to download."

        print("Downloading model artifact")
        model_artifact = api.artifact(wandb_artifact_fullname)
        model_artifact_dir = model_artifact.download()
        print("Model artifact downloaded to", model_artifact_dir)

        model = build_model(config)
        model = model.to(device)
        print(f"Using device: {device}")

        state_dict = torch.load(os.path.join(model_artifact_dir, 'best_model.pth'), map_location=device)
        model.load_state_dict(state_dict)

        # Load dataloader
        val_log_rgb, val_log_depth, val_loader = build_dataloader(config, subset_size = "MAX")

        # Load postprocessor
        post_processor = load_post_processor_from_config(config)

        # Output directory for visuals
        eval_output_dir = os.path.join(config["output_dir"], f"{config['logging']['run_name']}_eval")
        ensure_dir(eval_output_dir)

        # Run evaluation
        metrics = evaluate_model_with_postprocess(model, post_processor, val_loader, device, eval_output_dir)
        score = metrics[BEST_METRIC]

        all_results.append((config_path, score))
        print(f"{BEST_METRIC} for {os.path.basename(config_path)}: {score:.4f}")

        # Log metrics
        if config["logging"]["log"]:
            log_dict = {
                'Metrics/MAE': metrics['MAE'],
                'Metrics/RMSE': metrics['RMSE'],
                'Metrics/siRMSE': metrics['siRMSE'],
                'Metrics/REL': metrics['REL'],
                'Metrics/Delta1': metrics['Delta1'],
                'Metrics/Delta2': metrics['Delta2'],
                'Metrics/Delta3': metrics['Delta3'],
            }

            # Log images
            if config["logging"]["log_images"]:
                post_process_comparison = create_depth_comparison_postprocess(model, post_processor, val_log_rgb, val_log_depth, device)
                post_process_gt_comparison = create_depth_postprocess_gt_comparison(model, post_processor, val_log_rgb, val_log_depth, device)
                log_dict.update({
                    "Eval/Postprocess Depth Comparison": wandb.Image(post_process_comparison),
                    "Eval/Ground Truth Depth Comparison": wandb.Image(post_process_gt_comparison),
                })

            wandb.log(log_dict)

        if score < best_metric_value:
            best_metric_value = score
            best_config_path = config_path
    
        wandb.finish()

    print("\n==== Grid Search Summary ====")
    for path, score in sorted(all_results, key=lambda x: x[1]):
        print(f"{os.path.basename(path)} -> {BEST_METRIC}: {score:.4f}")

    if best_config_path:
        print(f"\nBest config: {best_config_path} with {BEST_METRIC} = {best_metric_value:.4f}")
    else:
        print("No valid config evaluated.")

if __name__ == "__main__":
    main()
