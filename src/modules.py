import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, hidden_channels=64, dilation=1, num_heads=4):
        super(SimpleUNet, self).__init__()
        
        # Encoder blocks
        self.enc1 = UNetBlock(3, hidden_channels, dilation)
        self.enc2 = UNetBlock(hidden_channels, hidden_channels * 2, dilation)
        self.enc3 = UNetBlock(hidden_channels * 2, hidden_channels * 4, dilation)
        
        # Decoder blocks
        self.dec3 = UNetBlock(hidden_channels * 6, hidden_channels * 2, dilation)
        self.dec2 = UNetBlock(hidden_channels * 3, hidden_channels, dilation)
        self.dec1 = UNetBlock(hidden_channels, hidden_channels // 2, dilation)
        
        # Final heads that are combined for depth and uncertainty estimation
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1),
                nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
            ) for _ in range(num_heads)
        ])

        # Copy the initial state of the heads
        self.initial_head_params = self.get_head_params().clone().detach()

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

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
        x = nn.functional.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)

        x = nn.functional.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)
        
        x = self.dec1(x)

        # Run inference through all heads
        x = torch.stack([head(x) for head in self.heads], dim=1)

        std = x.std(dim=1)
        x = x.mean(dim=1)

        # Output non-negative depth values
        x = torch.sigmoid(x)*10
        
        return x, std