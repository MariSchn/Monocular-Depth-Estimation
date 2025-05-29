import os
import itertools
import argparse
import yaml
from copy import deepcopy

# Post-processing parameter grid
POST_PROCESS_GRID = {
    "guided_filter": {
        "r": [2, 8, 16],
        "eps": [1e-2, 1e-3, 1e-4]
    },
    "gaussian_blur": {
        "kernel_size": [3, 5, 7],
        "sigma": [0.5, 1.0, 1.5]
    },
    "box_filter": {
        "kernel_size": [3, 5, 7]
    },
    "normalized_std_interpolation": {}
}

# Model-specific default settings
# Probably not accurate since some params are being tuned rn
MODEL_PRESETS = {
    "u_net": {
        "type": "u_net",
        "hidden_channels": 32,
        "dilation": 1,
        "conv_transpose": True,
        "weight_initialization": "glorot",
        "num_heads": 1,
        "include_pretrained_head": False,
        # "wandb_artifact_fullname": Add artifact name here,
    },
    "depth_anything": {
        "type": "depth_anything",
        "hidden_channels": 32,
        "dilation": 1,
        "conv_transpose": True,
        "weight_initialization": "default",
        "num_heads": 16,
        "include_pretrained_head": True,
        "wandb_artifact_fullname": "MonocularDepthEstimation/MonocularDepthEstimation/best_model:v5",
    },
    "dinov2_backboned_unet": {
        "type": "dinov2_backboned_unet",
        "hidden_channels": 32,
        "dilation": 1,
        "conv_transpose": True,
        "weight_initialization": "glorot",
        "num_heads": 4,
        "include_pretrained_head": False,
        "wandb_artifact_fullname": "MonocularDepthEstimation/MonocularDepthEstimation/best_model:v19",
    },
    "dinov2_large_backboned_unet": {
        "type": "dinov2_backboned_unet",
        "hidden_channels": 32,
        "dilation": 1,
        "conv_transpose": True,
        "weight_initialization": "glorot",
        "num_heads": 4,
        "include_pretrained_head": False,
        # "wandb_artifact_fullname": "MonocularDepthEstimation/MonocularDepthEstimation/best_model:v19",
    },
    "diunet": {
        "type": "diunet",
        "hidden_channels": 32,
        "dilation": 1,
        "conv_transpose": True,
        "weight_initialization": "glorot",
        "num_heads": 4,
        "include_pretrained_head": False,
        # "wandb_artifact_fullname": "MonocularDepthEstimation/MonocularDepthEstimation/best_model:v19",
    },
    "diunet_large": {
        "type": "diunet_large",
        "hidden_channels": 32,
        "dilation": 1,
        "conv_transpose": True,
        "weight_initialization": "glorot",
        "num_heads": 4,
        "include_pretrained_head": False,
        # "wandb_artifact_fullname": "MonocularDepthEstimation/MonocularDepthEstimation/best_model:v19",
    }
}

# Base training/data/logging config
BASE_CONFIG = {
    "device": "cuda",
    "seed": 0,
    "data_dir": "/cluster/courses/cil/monocular_depth/data",
    "output_dir": "./output",
    "train": {
        "batch_size": 8,
    },
    "data": {
        "input_size": [426, 560],
        "num_workers": 2,
        "pin_memory": True,
    },
    "logging": {
        "log": True,
        "log_images": True,
        "val_every_step_log": True,
        "log_every_k_epochs": 1,
        "log_every_k_steps": -1,
        "run_name": "",
        "entity": "MonocularDepthEstimation",
        "project_name": "MonocularDepthEstimation",
        "upload_to_wandb": True,
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate grid search post-process configs")
    parser.add_argument("--output_dir", type=str, default="./src/configs/postprocess_grid", help="Output directory")
    parser.add_argument("--model", type=str, default="dinov2_backboned_unet", choices=list(MODEL_PRESETS.keys()), help="Model type")
    return parser.parse_args()

def get_param_combinations(param_grid):
    if not param_grid:
        return [{}]
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base_config = BASE_CONFIG.copy()
    model_config = MODEL_PRESETS[args.model]

    config_id = 0
    for method, param_grid in POST_PROCESS_GRID.items():
        for param_combo in get_param_combinations(param_grid):
            config = deepcopy(base_config)
            config["model"] = deepcopy(model_config)
            config["logging"]["run_name"] = f"FineTune/id_{config_id}_{method}_model_{args.model}"
            config["post_process"] = {method: param_combo}

            output_path = os.path.join(args.output_dir, f"postprocess_config_{config_id}.yml")
            with open(output_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            config_id += 1

if __name__ == "__main__":
    main()
