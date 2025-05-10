import torch
import torch.nn as nn
from transformers import ResNetModel, ResNetConfig
import timm


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

        config = ResNetConfig.from_pretrained("microsoft/resnet-50", output_hidden_states=True)
        self.backbone = ResNetModel.from_pretrained("microsoft/resnet-50", config=config)

        self.dec4 = UNetBlock(2048 + 1024, 1024, dilation)
        self.dec3 = UNetBlock(1024 + 512, 512, dilation)
        self.dec2 = UNetBlock(512 + 256, 256, dilation)
        self.dec1 = UNetBlock(256 + 64, hidden_channels, dilation)

        self.final = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        feats = outputs.hidden_states  

        x = feats[4]  

        x = torch.nn.functional.interpolate(x, size=feats[3].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, feats[3]], dim=1)
        x = self.dec4(x)

        x = torch.nn.functional.interpolate(x, size=feats[2].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, feats[2]], dim=1)
        x = self.dec3(x)

        x = torch.nn.functional.interpolate(x, size=feats[1].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, feats[1]], dim=1)
        x = self.dec2(x)

        x = torch.nn.functional.interpolate(x, size=feats[0].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, feats[0]], dim=1)
        x = self.dec1(x)

        x = self.final(x)
        x = torch.sigmoid(x) * 10  

        x = torch.nn.functional.interpolate(x, size=(426, 560), mode='bilinear', align_corners=False)

        return x