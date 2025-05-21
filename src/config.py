from dataclasses import dataclass, field
from typing import List, Optional, get_origin, get_args, Union, Any

import yaml

class BaseConfig:
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"{key} is not a valid config field.")
    def __setitem__(self, key: str) -> Any:
        assert False, "Don't set values in code for a config you dingus"

@dataclass
class ResizeStepConfig(BaseConfig):
    pass

@dataclass
class GuidedFilterStepConfig(BaseConfig):
    radius: int = 3
    epsilon: float = 1e-4

@dataclass
class GaussianBlurStepConfig(BaseConfig):
    kernel_size: int = 5
    sigma: float = 1.0

@dataclass
class BoxFilterStepConfig(BaseConfig):
    kernel_size: int = 5

@dataclass
class NormalizedStdInterpolationConfig(BaseConfig):
    pass

@dataclass
class SigmoidStdInterpolationConfig(BaseConfig):
    # Adjust the following parameters based on the uncertainty domain we are working with.
    scale: float = 1.0  # how to scale the inputs to narrow or widen the sigmoid
    shift: float = -2.5  # how to shift the sigmoid curve :)

@dataclass
class PostProcessorConfig(BaseConfig):
    resize: Optional[ResizeStepConfig] = None
    guided_filter: Optional[GuidedFilterStepConfig] = None
    gaussian_blur: Optional[GaussianBlurStepConfig] = None
    box_filter: Optional[BoxFilterStepConfig] = None
    normalized_std_interpolation: Optional[NormalizedStdInterpolationConfig] = None
    sigmoid_std_interpolation: Optional[SigmoidStdInterpolationConfig] = None

@dataclass
class ModelConfig(BaseConfig):
    type: str = "depth_anything"  # The type of model to use. ["u_net", "depth_anything", "dinov2_backboned_unet"]
    hidden_channels: int = 32
    dilation: int = 1

    conv_transpose: bool = True # Whether to use conv transpose layers in the decoder
    weight_initialization: str = "glorot" # The weight initialization method to use. ["glorot", "default"]

    num_heads: int = 16
    include_pretrained_head: bool = False
    num_parameters: int = -1

    wandb_artifact_fullname: str = "MonocularDepthEstimation/MonocularDepthEstimation/best_model:v5"

@dataclass
class TrainConfig(BaseConfig):
    num_epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    gradient_regularizer_weight: int = 0
    gradient_loss_weight: int = 1
    head_penalty_weight: List[int] = field(default_factory=lambda: [1, 0])  # Weight is scheduled from first value to second value over the course of training

@dataclass
class DataConfig(BaseConfig):
    input_size: List[int] = field(default_factory=lambda: [426, 560]) # Size of the input images (height, width).
    num_workers: int = 2         # How many workers to use for parallel data loading
    pin_memory: bool = True

@dataclass
class LoggingConfig(BaseConfig):
    log: bool = True                       # Whether to log the training or not (Useful for debugging to not create a new run every time)
    log_images: bool = True                # Whether to log a comparison figure between predictions and ground truth (at both epoch- and step-level)
    val_every_step_log: bool = True        # Whether to run on validation set at every step-level log
    log_every_k_epochs: int = 1           # How frequent to perform epoch-level logging (set to -1 to disable)
    log_every_k_steps: int = -1           # How frequent to perform step-level logging (set to -1 to disable)
    run_name: str = "Default run - don't waste your compute on this lol"
    entity: str = "MonocularDepthEstimation"       # Which entity/team to log to. There should be no need to change this
    project_name: str = "MonocularDepthEstimation" # Which project to log to. There should be no need to change this
    upload_to_wandb: bool = True                    # Whether to upload the model and predicted depth maps to wandb


@dataclass
class Config(BaseConfig):
    device: str = "cuda" # or "cpu"
    seed: int = 0

    data_dir: str = "/cluster/courses/cil/monocular_depth/data"
    output_dir: str = "./output" # "/work/scratch/smarian/output"  # Replace with your username if you want to save the output to scratch to avoid quota issues

    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    post_process: Optional[PostProcessorConfig] = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def from_dict(cls, data):
    """Recursively convert a dictionary into a dataclass instance.

    Thanks ChatGPT.
    """
    from dataclasses import fields, is_dataclass

    if not is_dataclass(cls):
        return data

    fieldtypes = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for key, val in data.items():
        if key in fieldtypes:
            field_type = fieldtypes[key]

            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    field_type = non_none_args[0]
            if is_dataclass(field_type):
                assert hasattr(field_type, "__getitem__"), "Make sure your config inherits from BaseConfig"
            kwargs[key] = from_dict(field_type, val)
    return cls(**kwargs)


def load_config_from_yaml(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return from_dict(Config, config)

if __name__ == "__main__":
    x = load_config_from_yaml("configs/default.yml")
    print(x["post_process"])
