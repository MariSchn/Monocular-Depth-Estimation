#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000
#SBATCH --account=cil_jobs
#SBATCH --output=./src/logs/log-%j.out

# Environment setup
. /etc/profile.d/modules.sh
module add cuda/12.6

source ~/.bashrc
conda activate monocular_depth

# Set WandB cache directories to scratch space to save space
export WANDB_DIR=/work/scratch/$USER/wandb
export WANDB_CACHE_DIR=/work/scratch/$USER/.cache/wandb
export WANDB_DATA_DIR=/work/scratch/$USER/.cache/wandb_data
export WANDB_CONFIG_DIR=/work/scratch/$USER/.config/wandb
export WANDB_ARTIFACT_DIR=/work/scratch/$USER/.artifacts

# Run the training script
python src/train.py --config ./src/configs/default.yml