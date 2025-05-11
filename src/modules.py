import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


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
    def __init__(self, hidden_channels=64, dilation=1):
        super(SimpleUNet, self).__init__()
        
        # Encoder blocks
        self.enc1 = UNetBlock(3, hidden_channels, dilation)
        self.enc2 = UNetBlock(hidden_channels, hidden_channels * 2, dilation)
        self.enc3 = UNetBlock(hidden_channels * 2, hidden_channels * 4, dilation)
        
        # Decoder blocks
        self.dec3 = UNetBlock(hidden_channels * 6, hidden_channels * 2, dilation)
        self.dec2 = UNetBlock(hidden_channels * 3, hidden_channels, dilation)
        self.dec1 = UNetBlock(hidden_channels, hidden_channels // 2, dilation)
        
        # Final layer
        self.final = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)

        x = self.enc3(x)
        
        # Decoder with skip connections
        x = nn.functional.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)

        x = nn.functional.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)
        
        x = self.dec1(x)
        x = self.final(x)
        
        # Output non-negative depth values
        x = torch.sigmoid(x)*10
        
        return x
    
class UNetWithResNet50Backbone(nn.Module):
    def __init__(self, hidden_channels=64, dilation=1):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-small')


        # Decoder blocks
        self.dec3 = UNetBlock(384, 256, dilation)
        self.dec2 = UNetBlock(256, 128, dilation)
        self.dec1 = UNetBlock(128, 64, dilation)
        
        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        images = [x_i.detach().cpu().permute(1, 2, 0).numpy() for x_i in x] 
        inputs = self.processor(images=images, return_tensors="pt", do_resize=False, do_rescale=True).to(x.device)

        outputs = self.backbone(**inputs)

        tokens = outputs.last_hidden_state[:, 1:, :]
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        feature_map = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        print(feature_map.shape)

        # Decoder with skip connections
        out = nn.functional.interpolate(feature_map, scale_factor=4, mode='bilinear', align_corners=False)
        # print(out.shape)
        out = self.dec3(out)
        # print(out.shape)

        out = nn.functional.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        # print(out.shape)
        out = self.dec2(out)
        # print(out.shape)
        
        out = self.dec1(out)
        # print(out.shape)
        out = self.final(out)
        # print(out.shape)
        
        
        out = nn.functional.interpolate(out, size=(426, 560), mode='bilinear', align_corners=False)
        # print(out.shape)
        out = torch.sigmoid(out) * 10
        return out