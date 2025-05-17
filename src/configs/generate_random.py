import yaml
import os
import random
import argparse

LEARN_RATE_OPTIONS = [1e-3, 5e-3, 1e-4]
WEIGHT_DECAY_OPTIONS = [0, 1e-1, 1e-2, 1e-3, 1e-4]
GRADIENT_LOSS_WEIGHT_OPTIONS = [0, 1, 2, 4, 8]
HEAD_PENALTY_WEIGHT_OPTIONS = [[1, 0], [1, 1], [2, 0], [2, 1]]
HIDDEN_CHANNELS_OPTIONS = [16, 32, 64]
DILATION_OPTIONS = [1, 2, 4]
NUM_HEADS_OPTIONS = [1, 2, 4, 8, 16]

BASE_CONFIG = {
        "device": "cuda",  # or "cpu"
        "seed": 0,

        "data_dir": "/cluster/courses/cil/monocular_depth/data",
        "output_dir": "./output",

        "train": {
            "num_epochs": 15,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "gradient_regularizer_weight": 0,
            "gradient_loss_weight": 1,
            "head_penalty_weight": [1, 0],
        },

        "model": {
            "type": "dinov2_backboned_unet",  # ["u_net", "depth_anything", "dinov2_backboned_unet"]
            "hidden_channels": 32,
            "dilation": 1,
            "conv_transpose": True,
            "weight_initialization": "glorot",  # ["glorot", "default"]
            "num_heads": 8,
            "include_pretrained_head": False,
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
    parser = argparse.ArgumentParser(description="Generate random config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_configs", type=int, default=8, help="Number of configs to generate")
    parser.add_argument("--output_dir", type=str, default="./src/configs/search", help="Output directory for configs")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_configs):
        config = BASE_CONFIG.copy()
        config["logging"]["run_name"] = f"Random Search {i}"

        config["train"]["learning_rate"] = random.choice(LEARN_RATE_OPTIONS)
        config["train"]["weight_decay"] = random.choice(WEIGHT_DECAY_OPTIONS)
        config["train"]["gradient_loss_weight"] = random.choice(GRADIENT_LOSS_WEIGHT_OPTIONS)
        config["train"]["head_penalty_weight"] = random.choice(HEAD_PENALTY_WEIGHT_OPTIONS)
        config["model"]["hidden_channels"] = random.choice(HIDDEN_CHANNELS_OPTIONS)
        config["model"]["dilation"] = random.choice(DILATION_OPTIONS)
        config["model"]["num_heads"] = random.choice(NUM_HEADS_OPTIONS)

        with open(os.path.join(args.output_dir, f"config_{i}.yml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)