import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModel
from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingDepthEstimationHead

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class SimpleUNet(nn.Module):
    def __init__(self, hidden_channels=64, dilation=1, num_heads=4, conv_transpose=True, weight_initialization="glorot", depth_before_aggregate=False):
        super(SimpleUNet, self).__init__()
        self.conv_transpose = conv_transpose

        # Encoder blocks
        self.enc1 = UNetBlock(3, hidden_channels, dilation)
        self.enc2 = UNetBlock(hidden_channels, hidden_channels * 2, dilation)
        self.enc3 = UNetBlock(hidden_channels * 2, hidden_channels * 4, dilation)

        # Decoder blocks
        self.upsample1 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = UNetBlock(hidden_channels * 6, hidden_channels * 2, dilation)

        self.upsample2 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = UNetBlock(hidden_channels * 3, hidden_channels, dilation)

        self.dec1 = UNetBlock(hidden_channels, hidden_channels // 2, dilation)

        # Final heads that are combined for depth and uncertainty estimation
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU(),
                nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
            ) for _ in range(num_heads)
        ])

        # Initialize weights
        self.initialize_weights(weight_initialization)

        # Copy the initial state of the heads
        self.initial_head_params = self.get_head_params().clone().detach()

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

        self.depth_before_aggregate = depth_before_aggregate

    def initialize_weights(self, method):
        """
        Initialize the weights of the model, excluding the backbone.
        """
        if method == "default":
            return
        elif method == "glorot":
            modules_to_initialize = [
                self.enc1, self.enc2, self.enc3,
                self.dec3,
                self.dec2, self.upsample2,
                self.dec1, self.upsample1,
                self.heads
            ]
            for module_group in modules_to_initialize:
                for m in module_group.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.ConvTranspose2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")

    def get_head_params(self):
        """
        Returns the parameters of the heads as a flattened tensor.
        This is useful for optimization purposes.
        """
        return torch.cat([
                    torch.cat([
                        p.view(-1) for p in head.parameters() if p.requires_grad
                    ], dim=0) for head in self.heads
                ], dim=0)

    def head_penalty(self):
        """
        Calculates the penalty for the heads based on the distance from the initial parameters.
        """
        if len(self.heads) == 1:
            return torch.tensor(0.0, device=self.initial_head_params.device)

        head_params = self.get_head_params()
        return F.mse_loss(head_params, self.initial_head_params.to(head_params.device))

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)

        enc2 = self.enc2(x)
        x = self.pool(enc2)

        x = self.enc3(x)

        # Decoder with skip connections
        if self.conv_transpose:
            x = self.upsample1(x)

        x = nn.functional.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)

        if self.conv_transpose:
            x = self.upsample2(x)

        x = nn.functional.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)

        x = self.dec1(x)

        # Run inference through all heads
        x = torch.stack([head(x) for head in self.heads], dim=1)

        if self.depth_before_aggregate:
            x = torch.sigmoid(x) * 10

        std = x.std(dim=1)
        x = x.mean(dim=1)

        # Output non-negative depth values
        if not self.depth_before_aggregate:
            x = torch.sigmoid(x) * 10

        return x, std

class UncertaintyDepthAnything(nn.Module):
    """
    This class is a wrapper for the DepthAnything model that uses multiple heads to quantify the uncertainty of the depth estimation.

    The code was written with the Hugging Face library as a base, modifying the original DepthAnything model to add the uncertainty quantification.
    The base code was taken from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/depth_anything/modeling_depth_anything.py
    """
    def __init__(
            self,
            num_heads: int = 1,
            include_pretrained_head: bool = False,
            model_path: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            max_depth: float = 10.0,
            weight_initialization: str = "glorot",
        ):
        super().__init__()

        # Load model and image processor
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path)

        # Extract components
        self.backbone = self.model.backbone
        self.neck = self.model.neck

        # Freeze backbone and neck
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.neck.eval()
        for param in self.neck.parameters():
            param.requires_grad = False

        # Setup custom heads
        self.include_pretrained_head = include_pretrained_head
        self.num_heads = num_heads
        self.config = self.model.config
        if include_pretrained_head:
            # Use the pretrained head if requested
            self.heads = nn.ModuleList([self.model.head] + [DepthAnythingDepthEstimationHead(self.config) for _ in range(num_heads - 1)])
            for i, head in enumerate(self.heads):
                head.train()

                # Keep the pretrained head at the max_depth it was trained on
                if i != 0:
                    head.max_depth = max_depth
        else:
            self.heads = nn.ModuleList([DepthAnythingDepthEstimationHead(self.config) for _ in range(num_heads)])
            for head in self.heads:
                head.train()
                head.max_depth = max_depth

        # Initialize weights
        self.initialize_weights(weight_initialization)

        # Copy the initial state of the heads
        self.initial_head_params = self.get_head_params().clone().detach()

    def initialize_weights(self, method):
        """
        Initialize the weights of the model, excluding the backbone.
        """
        if method == "default":
            return
        elif self.include_pretrained_head and self.num_heads == 1:
            return
        elif method == "glorot":
            print("Using Glorot initialization")
            modules_to_initialize = [
                self.heads if not self.include_pretrained_head else [self.heads[1:]],
            ]
            for module_group in modules_to_initialize:
                for m in module_group.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.ConvTranspose2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")

    def get_head_params(self):
        """
        Returns the parameters of the heads as a flattened tensor.
        This is useful for optimization purposes.
        """
        return torch.cat([
                    torch.cat([
                        p.view(-1) for p in head.parameters() if p.requires_grad
                    ], dim=0) for head in self.heads
                ], dim=0)

    def head_penalty(self):
        """
        Calculates the penalty for the heads based on the distance from the initial parameters.
        """
        if len(self.heads) == 1:
            return torch.tensor(0.0, device=self.initial_head_params.device)

        head_params = self.get_head_params()
        return F.mse_loss(head_params, self.initial_head_params.to(head_params.device))

    def forward(self, x, output_resolution=(426, 560)) -> torch.FloatTensor:
        # Get hidden_states, unfortunately this can not be done as precomputation due to storage quota
        with torch.no_grad():
            # Forward pass through the backbone
            outputs = self.backbone.forward_with_filtered_kwargs(x)
            hidden_states = outputs.feature_maps

            # Forward pass through the neck
            _, _, height, width = x.shape
            patch_size = self.config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

            hidden_states = self.neck(hidden_states, patch_height, patch_width)

        # Forward pass through the heads
        predictions = []
        for head in self.heads:
            outputs = head(hidden_states, patch_height, patch_width)
            outputs = outputs.unsqueeze(1)

            # Interpolate the output to the desired resolution
            outputs = F.interpolate(
                outputs,
                size=output_resolution,
                mode='bilinear',
                align_corners=False
            )

            predictions.append(outputs)
        predictions = torch.stack(predictions, dim=0)

        # Calculate the mean and variance of the predictions
        if self.num_heads > 1:
            x = torch.mean(predictions, dim=0)
            std = torch.std(predictions, dim=0)
        else:
            x = predictions.squeeze(0)
            std = torch.zeros_like(x)

        return x, std

class DiUNet(nn.Module):
    def __init__(self, hidden_channels=64, dilation=1, num_heads=4, image_size=(426, 560), conv_transpose=True, weight_initialization="glorot", depth_before_aggregate=False):
        super().__init__()
        self.image_size = image_size
        self.conv_transpose = conv_transpose

        # Encoder blocks
        self.enc1_output_size = hidden_channels
        self.enc2_output_size = hidden_channels * 2
        self.enc3_output_size = hidden_channels * 4

        self.enc1 = UNetBlock(3, self.enc1_output_size, dilation)
        self.enc2 = UNetBlock(self.enc1_output_size, self.enc2_output_size, dilation)
        self.enc3 = UNetBlock(self.enc2_output_size, self.enc3_output_size, dilation)

        self.upsampling_amount = 3
        self.upsampling_factor = 2

        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-small')

        self.channel_dim = self.backbone.config.hidden_size

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Decoder blocks
        self.dec3 = UNetBlock(self.channel_dim + self.enc3_output_size, hidden_channels * 2, dilation)
        self.upsample3 = nn.ConvTranspose2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec2 = UNetBlock(hidden_channels * 2 + self.enc2_output_size, hidden_channels, dilation)
        self.upsample2 = nn.ConvTranspose2d(
            hidden_channels,
            hidden_channels,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec1 = UNetBlock(hidden_channels + self.enc1_output_size, hidden_channels // 2, dilation)
        self.upsample1 = nn.ConvTranspose2d(
            hidden_channels // 2,
            hidden_channels // 2,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        # Final heads that are combined for depth and uncertainty estimation
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU(),
                nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
            ) for _ in range(num_heads)
        ])

        # Initialize weights
        self.initialize_weights(weight_initialization)

        # Copy the initial state of the heads
        self.initial_head_params = self.get_head_params().clone().detach()

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

        self.depth_before_aggregate = depth_before_aggregate

    def initialize_weights(self, method):
        """
        Initialize the weights of the model, excluding the backbone.
        """
        if method == "default":
            return
        elif method == "glorot":
            print("Using Glorot initialization")
            modules_to_initialize = [
                self.dec3, self.upsample3,
                self.dec2, self.upsample2,
                self.dec1, self.upsample1,
                self.heads
            ]
            for module_group in modules_to_initialize:
                for m in module_group.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.ConvTranspose2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")

    def get_head_params(self):
        """
        Returns the parameters of the heads as a flattened tensor.
        This is useful for optimization purposes.
        """
        return torch.cat([
                    torch.cat([
                        p.view(-1) for p in head.parameters() if p.requires_grad
                    ], dim=0) for head in self.heads
                ], dim=0)

    def head_penalty(self):
        """
        Calculates the penalty for the heads based on the distance from the initial parameters.
        """
        if len(self.heads) == 1:
            return torch.tensor(0.0, device=self.initial_head_params.device)

        head_params = self.get_head_params()
        return F.mse_loss(head_params, self.initial_head_params.to(head_params.device))


    def forward(self, x):
        # Encoder
        encoder_input = x
        # print(f"Input shape: {encoder_input.shape}")

        enc1_output = self.enc1(encoder_input)
        encoder_input = self.pool(enc1_output)
        # print(f"After enc1 -> pool1: {encoder_input.shape}")

        enc2_output = self.enc2(encoder_input)
        encoder_input = self.pool(enc2_output)
        # print(f"After enc2 -> pool2: {encoder_input.shape}")

        enc3_output = self.enc3(encoder_input)
        # print(f"After enc3: {enc3_output.shape}")
        # print("")
        # print(f"input x shape: {x.shape}")

        images = [x_i.detach().cpu().permute(1, 2, 0).numpy() for x_i in x]
        inputs = self.processor(images=images, return_tensors="pt", do_resize=False, do_rescale=True).to(x.device)

        with torch.no_grad():
            outputs = self.backbone(**inputs)

        tokens = outputs.last_hidden_state[:, 1:, :]
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        feature_map = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        # print(f"Feature map shape: {feature_map.shape}")

        # Resize feature map to fit original aspect ratio
        total_scale_factor = self.upsampling_factor ** self.upsampling_amount
        patch_size = (self.image_size[0] // total_scale_factor, self.image_size[1] // total_scale_factor)
        out = nn.functional.interpolate(feature_map, size=patch_size, mode='bilinear', align_corners=False)

        # add first skip connection
        skip1 = nn.functional.interpolate(enc3_output, size=out.shape[2:], mode='bilinear', align_corners=False)
        # print(f"Skip1 shape: {skip1.shape}")
        out = torch.cat([out, skip1], dim=1)
        # print(f"After concatenating skip1: {out.shape}")

        out = self.dec3(out)
        # print(f"After dec3: {out.shape}")
        if self.conv_transpose:
            out = self.upsample3(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)

        # print(f"After upsample3: {out.shape}")

        # add second skip connection
        skip2 = nn.functional.interpolate(enc2_output, size=out.shape[2:], mode='bilinear', align_corners=False)
        # print(f"Skip2 shape: {skip2.shape}")
        out = torch.cat([out, skip2], dim=1)
        # print(f"After concatenating skip2: {out.shape}")

        out = self.dec2(out)
        # print(f"After dec2: {out.shape}")
        if self.conv_transpose:
            out = self.upsample2(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)
        # print(f"After upsample2: {out.shape}")

        # add third skip connection
        skip3 = nn.functional.interpolate(enc1_output, size=out.shape[2:], mode='bilinear', align_corners=False)
        # print(f"Skip3 shape: {skip3.shape}")
        out = torch.cat([out, skip3], dim=1)
        # print(f"After concatenating skip3: {out.shape}")

        out = self.dec1(out)
        # print(f"After dec1: {out.shape}")
        if self.conv_transpose:
            out = self.upsample1(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)
        # print(f"After upsample1: {out.shape}")

        # print("")

        # resize to original image size
        out = nn.functional.interpolate(out, size=self.image_size, mode='bilinear', align_corners=False)

        # Run inference through all heads
        out = torch.stack([head(out) for head in self.heads], dim=1)

        if self.depth_before_aggregate:
            out = torch.sigmoid(out) * 10

        std = out.std(dim=1)
        out = out.mean(dim=1)

        # Output non-negative depth values
        if not self.depth_before_aggregate:
            out = torch.sigmoid(out) * 10

        return out, std
    
class DiUNetLarge(nn.Module):
    def __init__(self, hidden_channels=64, dilation=1, num_heads=4, image_size=(426, 560), conv_transpose=True, weight_initialization="glorot", depth_before_aggregate=False):
        super().__init__()
        self.image_size = image_size
        self.conv_transpose = conv_transpose

        # Encoder blocks
        self.enc1_output_size = hidden_channels // 2
        self.enc2_output_size = hidden_channels * 2
        self.enc3_output_size = hidden_channels * 8
        self.enc4_output_size = hidden_channels * 16

        self.enc1 = UNetBlock(3, self.enc1_output_size, dilation)
        self.enc2 = UNetBlock(self.enc1_output_size, self.enc2_output_size, dilation)
        self.enc3 = UNetBlock(self.enc2_output_size, self.enc3_output_size, dilation)
        self.enc4 = UNetBlock(self.enc3_output_size, self.enc4_output_size, dilation)

        self.upsampling_amount = 4
        self.upsampling_factor = 2

        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-large')

        self.channel_dim = self.backbone.config.hidden_size

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Decoder blocks
        self.dec3 = UNetBlock(self.channel_dim + self.enc4_output_size, hidden_channels * 8, dilation)
        self.upsample3 = nn.ConvTranspose2d(
            hidden_channels * 8,
            hidden_channels * 8,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )


        self.dec2 = UNetBlock(2 * self.enc3_output_size, hidden_channels * 2, dilation)
        self.upsample2 = nn.ConvTranspose2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec1 = UNetBlock(2 * self.enc2_output_size, hidden_channels // 2, dilation)
        self.upsample1 = nn.ConvTranspose2d(
            hidden_channels  // 2,
            hidden_channels  // 2,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec0 = UNetBlock(2 * self.enc1_output_size, hidden_channels // 4, dilation)
        self.upsample0 = nn.ConvTranspose2d(
            hidden_channels // 4,
            hidden_channels // 4,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        # Final heads that are combined for depth and uncertainty estimation
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels // 4, hidden_channels // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels // 4),
                nn.ReLU(),
                nn.Conv2d(hidden_channels // 4, 1, kernel_size=1)
            ) for _ in range(num_heads)
        ])

        # Initialize weights
        self.initialize_weights(weight_initialization)

        # Copy the initial state of the heads
        self.initial_head_params = self.get_head_params().clone().detach()

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

        self.depth_before_aggregate = depth_before_aggregate

    def initialize_weights(self, method):
        """
        Initialize the weights of the model, excluding the backbone.
        """
        if method == "default":
            return
        elif method == "glorot":
            print("Using Glorot initialization")
            modules_to_initialize = [
                self.dec3, self.upsample3,
                self.dec2, self.upsample2,
                self.dec1, self.upsample1,
                self.heads
            ]
            for module_group in modules_to_initialize:
                for m in module_group.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.ConvTranspose2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")

    def get_head_params(self):
        """
        Returns the parameters of the heads as a flattened tensor.
        This is useful for optimization purposes.
        """
        return torch.cat([
                    torch.cat([
                        p.view(-1) for p in head.parameters() if p.requires_grad
                    ], dim=0) for head in self.heads
                ], dim=0)

    def head_penalty(self):
        """
        Calculates the penalty for the heads based on the distance from the initial parameters.
        """
        if len(self.heads) == 1:
            return torch.tensor(0.0, device=self.initial_head_params.device)

        head_params = self.get_head_params()
        return F.mse_loss(head_params, self.initial_head_params.to(head_params.device))

    def forward(self, x):
        # Encoder
        encoder_input = x
        print(f"Input shape: {encoder_input.shape}")

        enc1_output = self.enc1(encoder_input)
        encoder_input = self.pool(enc1_output)
        print(f"After enc1 -> pool1: {encoder_input.shape}")

        enc2_output = self.enc2(encoder_input)
        encoder_input = self.pool(enc2_output)
        print(f"After enc2 -> pool2: {encoder_input.shape}")

        enc3_output = self.enc3(encoder_input)
        encoder_input = self.pool(enc3_output)
        print(f"After enc2 -> pool2: {encoder_input.shape}")

        enc4_output = self.enc4(encoder_input)
        print(f"After enc4: {enc4_output.shape}")
        print("")
        print(f"input x shape: {x.shape}")

        images = [x_i.detach().cpu().permute(1, 2, 0).numpy() for x_i in x]
        inputs = self.processor(images=images, return_tensors="pt", do_resize=False, do_rescale=True).to(x.device)

        with torch.no_grad():
            outputs = self.backbone(**inputs)

        tokens = outputs.last_hidden_state[:, 1:, :]
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        feature_map = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        print(f"Feature map shape: {feature_map.shape}")

        # Resize feature map to fit original aspect ratio
        total_scale_factor = self.upsampling_factor ** self.upsampling_amount
        patch_size = (self.image_size[0] // total_scale_factor, self.image_size[1] // total_scale_factor)
        out = nn.functional.interpolate(feature_map, size=patch_size, mode='bilinear', align_corners=False)
        print(f"After initial interpolation: {out.shape}")

        # add first skip connection
        skip1 = nn.functional.interpolate(enc4_output, size=out.shape[2:], mode='bilinear', align_corners=False)
        print(f"Skip1 shape: {skip1.shape}")
        out = torch.cat([out, skip1], dim=1)
        print(f"After concatenating skip1: {out.shape}")

        out = self.dec3(out)
        print(f"After dec3: {out.shape}")
        if self.conv_transpose:
            out = self.upsample3(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)

        print(f"After upsample3: {out.shape}")

        # add second skip connection
        skip2 = nn.functional.interpolate(enc3_output, size=out.shape[2:], mode='bilinear', align_corners=False)
        print(f"Skip2 shape: {skip2.shape}")
        out = torch.cat([out, skip2], dim=1)
        print(f"After concatenating skip2: {out.shape}")

        out = self.dec2(out)
        print(f"After dec2: {out.shape}")
        if self.conv_transpose:
            out = self.upsample2(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)
        print(f"After upsample2: {out.shape}")

        # add third skip connection
        skip3 = nn.functional.interpolate(enc2_output, size=out.shape[2:], mode='bilinear', align_corners=False)
        print(f"Skip3 shape: {skip3.shape}")
        out = torch.cat([out, skip3], dim=1)
        print(f"After concatenating skip3: {out.shape}")

        out = self.dec1(out)
        print(f"After dec1: {out.shape}")
        if self.conv_transpose:
            out = self.upsample1(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)
        print(f"After upsample1: {out.shape}")

        # add forth skip connection
        skip4 = nn.functional.interpolate(enc1_output, size=out.shape[2:], mode='bilinear', align_corners=False)
        print(f"Skip4 shape: {skip4.shape}")
        out = torch.cat([out, skip4], dim=1)
        print(f"After concatenating skip4: {out.shape}")

        out = self.dec0(out)
        print(f"After dec0: {out.shape}")
        if self.conv_transpose:
            out = self.upsample0(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)
        print(f"After upsample0: {out.shape}")

        print("")

        # resize to original image size
        out = nn.functional.interpolate(out, size=self.image_size, mode='bilinear', align_corners=False)

        # Run inference through all heads
        out = torch.stack([head(out) for head in self.heads], dim=1)

        if self.depth_before_aggregate:
            out = torch.sigmoid(out) * 10

        std = out.std(dim=1)
        out = out.mean(dim=1)

        # Output non-negative depth values
        if not self.depth_before_aggregate:
            out = torch.sigmoid(out) * 10

        return out, std

class UNetWithDinoV2LargeBackbone(nn.Module):
    def __init__(self, hidden_channels=64, dilation=1, num_heads=4, image_size=(426, 560), conv_transpose=True, weight_initialization="glorot", depth_before_aggregate=False):
        super().__init__()
        self.image_size = image_size
        self.conv_transpose = conv_transpose

        self.upsampling_amount = 4
        self.upsampling_factor = 2

        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-large')

        self.channel_dim = self.backbone.config.hidden_size

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Decoder blocks
        self.dec3 = UNetBlock(self.channel_dim, hidden_channels * 8, dilation)
        self.upsample3 = nn.ConvTranspose2d(
            hidden_channels * 8,
            hidden_channels * 8,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec2 = UNetBlock(hidden_channels * 8, hidden_channels * 4, dilation)
        self.upsample2 = nn.ConvTranspose2d(
            hidden_channels * 4,
            hidden_channels * 4,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec1 = UNetBlock(hidden_channels * 4, hidden_channels, dilation)
        self.upsample1 = nn.ConvTranspose2d(
            hidden_channels,
            hidden_channels,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec0 = UNetBlock(hidden_channels, hidden_channels // 2, dilation)
        self.upsample0 = nn.ConvTranspose2d(
            hidden_channels // 2,
            hidden_channels // 2,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        # Final heads that are combined for depth and uncertainty estimation
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU(),
                nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
            ) for _ in range(num_heads)
        ])

        # Initialize weights
        self.initialize_weights(weight_initialization)

        # Copy the initial state of the heads
        self.initial_head_params = self.get_head_params().clone().detach()

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

        self.depth_before_aggregate = depth_before_aggregate

    def initialize_weights(self, method):
        """
        Initialize the weights of the model, excluding the backbone.
        """
        if method == "default":
            return
        elif method == "glorot":
            print("Using Glorot initialization")
            modules_to_initialize = [
                self.dec3, self.upsample3,
                self.dec2, self.upsample2,
                self.dec1, self.upsample1,
                self.heads
            ]
            for module_group in modules_to_initialize:
                for m in module_group.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.ConvTranspose2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")

    def get_head_params(self):
        """
        Returns the parameters of the heads as a flattened tensor.
        This is useful for optimization purposes.
        """
        return torch.cat([
                    torch.cat([
                        p.view(-1) for p in head.parameters() if p.requires_grad
                    ], dim=0) for head in self.heads
                ], dim=0)

    def head_penalty(self):
        """
        Calculates the penalty for the heads based on the distance from the initial parameters.
        """
        if len(self.heads) == 1:
            return torch.tensor(0.0, device=self.initial_head_params.device)

        head_params = self.get_head_params()
        return F.mse_loss(head_params, self.initial_head_params.to(head_params.device))


    def forward(self, x):
        images = [x_i.detach().cpu().permute(1, 2, 0).numpy() for x_i in x]
        inputs = self.processor(images=images, return_tensors="pt", do_resize=False, do_rescale=True).to(x.device)

        with torch.no_grad():
            outputs = self.backbone(**inputs)

        tokens = outputs.last_hidden_state[:, 1:, :]
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        feature_map = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        # print(f"tokens shape: {tokens.shape}")
        # print(f"Feature map shape: {feature_map.shape}")

        # Resize feature map to fit original aspect ratio
        total_scale_factor = self.upsampling_factor ** self.upsampling_amount
        patch_size = (self.image_size[0] // total_scale_factor, self.image_size[1] // total_scale_factor)
        out = nn.functional.interpolate(feature_map, size=patch_size, mode='bilinear', align_corners=False)

        out = self.dec3(out)
        # print(f"After dec3: {out.shape}")
        if self.conv_transpose:
            out = self.upsample3(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)

        # print(f"After upsample3: {out.shape}")

        out = self.dec2(out)
        # print(f"After dec2: {out.shape}")
        if self.conv_transpose:
            out = self.upsample2(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)
        # print(f"After upsample2: {out.shape}")


        out = self.dec1(out)
        # print(f"After dec1: {out.shape}")
        if self.conv_transpose:
            out = self.upsample1(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)
        # print(f"After upsample1: {out.shape}")

        out = self.dec0(out)
        # print(f"After dec0: {out.shape}")
        if self.conv_transpose:
            out = self.upsample0(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)
        # print(f"After upsample0: {out.shape}")

        # resize to original image size
        out = nn.functional.interpolate(out, size=self.image_size, mode='bilinear', align_corners=False)

        # Run inference through all heads
        out = torch.stack([head(out) for head in self.heads], dim=1)

        if self.depth_before_aggregate:
            out = torch.sigmoid(out) * 10

        std = out.std(dim=1)
        out = out.mean(dim=1)

        # Output non-negative depth values
        if not self.depth_before_aggregate:
            out = torch.sigmoid(out) * 10

        return out, std
    
class UNetWithDinoV2SmallBackbone(nn.Module):
    def __init__(self, hidden_channels=64, dilation=1, num_heads=4, image_size=(426, 560), conv_transpose=True, weight_initialization="glorot", depth_before_aggregate=False):
        super().__init__()
        self.image_size = image_size
        self.conv_transpose = conv_transpose

        self.upsampling_amount = 3
        self.upsampling_factor = 2

        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-small')

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Decoder blocks
        self.dec3 = UNetBlock(384, hidden_channels * 2, dilation)
        self.upsample3 = nn.ConvTranspose2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec2 = UNetBlock(hidden_channels * 2, hidden_channels, dilation)
        self.upsample2 = nn.ConvTranspose2d(
            hidden_channels,
            hidden_channels,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        self.dec1 = UNetBlock(hidden_channels, hidden_channels // 2, dilation)
        self.upsample1 = nn.ConvTranspose2d(
            hidden_channels // 2,
            hidden_channels // 2,
            kernel_size=2 * self.upsampling_factor,
            stride=self.upsampling_factor,
            padding=self.upsampling_factor // 2
        )

        # Final heads that are combined for depth and uncertainty estimation
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU(),
                nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
            ) for _ in range(num_heads)
        ])

        # Initialize weights
        self.initialize_weights(weight_initialization)

        # Copy the initial state of the heads
        self.initial_head_params = self.get_head_params().clone().detach()

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

        self.depth_before_aggregate = depth_before_aggregate

    def initialize_weights(self, method):
        """
        Initialize the weights of the model, excluding the backbone.
        """
        if method == "default":
            return
        elif method == "glorot":
            print("Using Glorot initialization")
            modules_to_initialize = [
                self.dec3, self.upsample3,
                self.dec2, self.upsample2,
                self.dec1, self.upsample1,
                self.heads
            ]
            for module_group in modules_to_initialize:
                for m in module_group.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.ConvTranspose2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")

    def get_head_params(self):
        """
        Returns the parameters of the heads as a flattened tensor.
        This is useful for optimization purposes.
        """
        return torch.cat([
                    torch.cat([
                        p.view(-1) for p in head.parameters() if p.requires_grad
                    ], dim=0) for head in self.heads
                ], dim=0)

    def head_penalty(self):
        """
        Calculates the penalty for the heads based on the distance from the initial parameters.
        """
        if len(self.heads) == 1:
            return torch.tensor(0.0, device=self.initial_head_params.device)

        head_params = self.get_head_params()
        return F.mse_loss(head_params, self.initial_head_params.to(head_params.device))


    def forward(self, x):
        images = [x_i.detach().cpu().permute(1, 2, 0).numpy() for x_i in x]
        inputs = self.processor(images=images, return_tensors="pt", do_resize=False, do_rescale=True).to(x.device)

        with torch.no_grad():
            outputs = self.backbone(**inputs)

        tokens = outputs.last_hidden_state[:, 1:, :]
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        feature_map = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        # Resize feature map to fit original aspect ratio
        total_scale_factor = self.upsampling_factor ** self.upsampling_amount
        patch_size = (self.image_size[0] // total_scale_factor, self.image_size[1] // total_scale_factor)
        out = nn.functional.interpolate(feature_map, size=patch_size, mode='bilinear', align_corners=False)

        out = self.dec3(out)
        if self.conv_transpose:
            out = self.upsample3(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)

        out = self.dec2(out)
        if self.conv_transpose:
            out = self.upsample2(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)

        out = self.dec1(out)
        if self.conv_transpose:
            out = self.upsample1(out)
        else:
            out = nn.functional.interpolate(out, scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False)

        # resize to original image size
        out = nn.functional.interpolate(out, size=self.image_size, mode='bilinear', align_corners=False)

        # Run inference through all heads
        out = torch.stack([head(out) for head in self.heads], dim=1)

        if self.depth_before_aggregate:
            out = torch.sigmoid(out) * 10

        std = out.std(dim=1)
        out = out.mean(dim=1)

        # Output non-negative depth values
        if not self.depth_before_aggregate:
            out = torch.sigmoid(out) * 10

        return out, std