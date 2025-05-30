# Uncertainty-aware Monocular Depth Estimation

## Introduction

This repository contains the code for the report "Uncertainty-aware Monocular Depth Estimation" for the Computational Intelligence Lab (Spring Semester 2025) at ETH Zurich. 
Our work focused on incorporating uncertainty estimation into monocular depth estimation models, and utilize the uncertainty to improve the performance, by appling stronger post-processing in the regions with high uncertainty.

## Setup

To set up the environment, you can use the provided `environment.yml` file to create a conda environment. Simply run the following command:

```bash
conda env create -f environment.yml
```

After the environment is created, you can activate it with:

```bash
conda activate monocular_depth
```

## Reproducability

To reproduce the results from our report, you can use the `run.sh` script to start individual trainings.
The script loads a config `.yml` file from the `src/configs` directory. By default, it will use the `src/configs/default.yml` file.

The configs used to train the models for our final results can be found in the `src/configs/evaluations/architecture` directory. All configs contain a fixed random seed to ensure reproducibility.
To change the config used for training, simply change the `--config` argument in the `run.sh` script.

The configs for the final post-processing results can be generated using the `src/generate_postprocess_configs.py` script. To run the post-processing, the `src/evaluate_model.py.py` script can be used or `src/grid_search.py` to run all the models descibed in the configs from a specified folder in the `src/configs/postprocess_grid`. You can set the folder by using `--config` argument.
