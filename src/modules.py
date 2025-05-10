import torch
import torch.nn as nn
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
    def __init__(self, freeze_encoder=True):
        super(UNetWithResNet50Backbone, self).__init__()
        
        # Encoder blocks
        self.encoder = timm.create_model('resnet50', pretrained=True, features_only=True)

        # Freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        encoder_channels = self.encoder.feature_info.channels() 

        # Decoder blocks
        self.dec4 = UNetBlock(encoder_channels[4], encoder_channels[3])
        self.dec3 = UNetBlock(encoder_channels[3] + encoder_channels[3], encoder_channels[2])
        self.dec2 = UNetBlock(encoder_channels[2] + encoder_channels[2], encoder_channels[1])
        self.dec1 = UNetBlock(encoder_channels[1] + encoder_channels[1], encoder_channels[0])
        self.dec0 = UNetBlock(encoder_channels[0] + 3, 64)

        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # printf"x: {x.shape}")

        # Encoder
        feats = self.encoder(x)

        x5 = feats[4]
        x4 = feats[3]
        x3 = feats[2]
        x2 = feats[1]
        x1 = feats[0]
    
        # printf"x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}, x5: {x5.shape}")
    
        d4 = nn.functional.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        # print(f"d4 (after upsample x5): {d4.shape}")
        d4 = self.dec4(d4)
        # printf"d4 (after dec4): {d4.shape}")
    
        d3 = torch.cat([d4, x4], dim=1)
        # printf"d3 (after concat with x4): {d3.shape}")
        d3 = nn.functional.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        # printf"d3 (after upsample): {d3.shape}")
        d3 = self.dec3(d3)
    
        d2 = torch.cat([d3, x3], dim=1)
        # printf"d2 (after concat with x3): {d2.shape}")
        d2 = nn.functional.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        # printf"d2 (after upsample): {d2.shape}")
        d2 = self.dec2(d2)
    
        d1 = torch.cat([d2, x2], dim=1)
        # printf"d1 (after concat with x2): {d1.shape}")
        d1 = nn.functional.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        # printf"d1 (after upsample): {d1.shape}")
        d1 = self.dec1(d1)
    
        d0 = torch.cat([d1, x1], dim=1)
        # printf"d0 (after concat with x1): {d0.shape}")
        d0 = self.dec0(d0)
        # printf"d0 (after dec0): {d0.shape}")
    
        out = self.final(d0)
        # printf"out (final): {out.shape}")
    
        out = torch.sigmoid(out) * 10
        return out