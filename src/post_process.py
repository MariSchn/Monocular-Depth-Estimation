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
import yaml


from utils import ensure_dir, load_config, target_transform, create_depth_comparison, gradient_regularizer, gradient_loss
from modules import SimpleUNet
from data import DepthDataset
from train import generate_test_predictions

from guided_filter_pytorch.guided_filter import GuidedFilter

from typing import List, Callable, Any

# PostProcessor base class, courtesy of Mr.GPT :)
class PostProcessor():
    def __init__(self, steps: List[Callable[..., Any]] = None):
        """
        Initialize the post-processor with an optional list of steps.

        Args:
            steps (List[Callable]): List of callables that manipulate model output.
        """
        self.steps = steps or []

    def add_step(self, step: Callable[..., Any]):
        """Add a single post-processing step."""
        self.steps.append(step)

    def __call__(self, output, **kwargs) -> Any:
        """
        Apply all configured post-processing steps using keyword arguments.

        Args:
            output: Model output to be processed
            kwargs: Arbitrary keyword arguments passed to each step.

        Returns:
            Any: The final output after post-processing.
        """
        for step in self.steps:
            output = step(output, **kwargs)
        return output

    def __str__(self) -> str:
        return ",".join([str(s) for s in self.steps])

class ResizeStep:
    def __call__(self, model_output, **kwargs):
        # Resize to the same dimensions as the input.
        resize_to = kwargs.get("resize_to")
        assert resize_to is not None, "Provide the `resize_to` np.shape parameter to use this step"
        model_output = nn.functional.interpolate(
            model_output, size=resize_to, mode='bilinear', align_corners=False
        )
        return model_output

    def __str__(self):
        return "ResizeStep"

class GuidedFilteringStep:
    def __init__(self, radius=3, epsilon=1e-3):
        self.radius = radius
        self.epsilon = epsilon
        self.filter = GuidedFilter(self.radius, self.epsilon)

    def __call__(self, outputs, **kwargs):
        inputs = kwargs.get("inputs")
        assert inputs is not None, "Provide the `inputs` RGB images to use this step"

        # If the inputs are not the same size as the output, use bilinear interpolation to resize it.
        resized_inputs = nn.functional.interpolate(
            inputs, size=outputs.shape[-2:], mode='bilinear', align_corners=False
        )

        # Use the original RGB inputs as a guide to perform smoothing.
        filtered = self.filter(outputs, resized_inputs)

        # Since the guide image is a color image, we may replicate the smoothed depth across the 3 channels.
        filtered = filtered.mean(dim=1, keepdim=True)

        return filtered

    def __str__(self):
        return f"GuidedFilter(r={self.radius}, epsilon={self.epsilon})"

class GaussianBlurStep:
    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=(self.sigma, self.sigma))

    def __call__(self, outputs, **kwargs):
        smoothed = self.blur(outputs)
        return smoothed

    def __str__(self):
        return f"GaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma})"

class BoxFilter:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __call__(self, outputs, **kwargs):
        weight = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=outputs.device, dtype=outputs.dtype)
        weight /= self.kernel_size ** 2

        # Apply depthwise conv: groups=C to apply same kernel per channel
        filtered = nn.functional.conv2d(outputs, weight, padding=self.kernel_size // 2, groups=1)
        return filtered

    def __str__(self):
        return f"BoxFilter(kernel_size={self.kernel_size})"


class NormalizedStdInterpolation:
    def __call__(self, outputs, **kwargs):
        raw_outputs = kwargs.get("raw_outputs")
        assert raw_outputs is not None, "Provide the `raw_outputs` original predictions to use this step"
        std = kwargs.get("std")
        assert std is not None, "Provide the `std` std deviation across heads to use this step"
        normalized_std = std.clone()
        for i in range(normalized_std.shape[0]):
            normalized_std[i] = (std[i] - std[i].min()) / (std[i].max() - std[i].min() + 1e-6)

        # Combine the processed output with the raw output.
        outputs = (1 - normalized_std) * outputs + normalized_std * raw_outputs

        del normalized_std

        return outputs

    def __str__(self):
        return f"NormalizedStdInterpolation"
    
class SigmoidStdInterpolation:
    def __init__(self, scale=1.0, shift=-2.5):
        self.scale = scale
        self.shift = shift

    def __call__(self, outputs, **kwargs):
        raw_outputs = kwargs.get("raw_outputs")
        assert raw_outputs is not None, "Provide the `raw_outputs` original predictions to use this step"
        std = kwargs.get("std")
        assert std is not None, "Provide the `std` std deviation across heads to use this step"
        normalized_std = std.clone()

        normalized_std = torch.sigmoid(self.scale * (normalized_std + self.shift))

        # Combine the processed output with the raw output.
        outputs = (1 - normalized_std) * outputs + normalized_std * raw_outputs

        del normalized_std

        return outputs

    def __str__(self):
        return f"SigmoidStdInterpolation"

def load_post_processor_from_config(config) -> PostProcessor:
    p = PostProcessor()
    p.add_step(ResizeStep())
    if "post_process" in config:
        if "guided_filter" in config["post_process"]:
            gfilter_conf = config["post_process"]["guided_filter"]
            p.add_step(GuidedFilteringStep(gfilter_conf.get("r", 3), gfilter_conf.get("eps", 1e-3)))
        if "gaussian_blur" in config["post_process"]:
            gblur_conf = config["post_process"]["gaussian_blur"]
            p.add_step(GaussianBlurStep(kernel_size=gblur_conf["kernel_size"], sigma=gblur_conf["sigma"]))
        if "box_filter" in config["post_process"]:
            boxfilter_conf = config["post_process"]["box_filter"]
            p.add_step(BoxFilter(kernel_size=boxfilter_conf.get("kernel_size")))

        # Add interpolation after filtering steps :)
        if "normalized_std_interpolation" in config["post_process"]:
            p.add_step(NormalizedStdInterpolation())
        if "sigmoid_std_interpolation" in config["post_process"]:
            p.add_step(SigmoidStdInterpolation(
                scale=config["post_process"]["sigmoid_std_interpolation"].get("scale", 1.0),
                shift=config["post_process"]["sigmoid_std_interpolation"].get("shift", -2.5)),
            )

    return p
