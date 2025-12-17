"""
Feature encoders for hieroglyphic image processing.

Contains various CNN architectures for extracting global and local features
from hieroglyphic images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNetEncoder(nn.Module):
    """ResNet18-based encoder for global and local feature extraction."""
    
    def __init__(self, global_dim=128, local_channels=64):
        """
        Initialize ResNet encoder.
        
        Args:
            global_dim: Dimension of global embedding vector
            local_channels: Number of channels in local feature map (not used, 
                          kept for interface compatibility)
        """
        super().__init__()
        self.global_dim = global_dim

        # Load ResNet18 without pretrained weights
        self.resnet = resnet18(weights=None)

        # Modify first conv layer for grayscale input (1 channel)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Extract feature layers (remove classifier)
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,  # Output: [B, 512, H, W]
        )

        # Local feature dimension from ResNet18
        local_feat_dim = 512

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_global = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(local_feat_dim, global_dim)
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 1, H, W]
            
        Returns:
            global_emb: Global embeddings [B, global_dim]
            local_feat: Local feature maps [B, 512, H', W']
        """
        local_feat = self.features(x)  # [B, 512, H', W']
        pooled = self.global_pool(local_feat)  # [B, 512, 1, 1]
        global_emb = self.fc_global(pooled.squeeze(-1).squeeze(-1))  # [B, global_dim]

        return global_emb, local_feat


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        """
        Initialize residual block.
        
        Args:
            in_ch: Input channels
            out_ch: Output channels
            stride: Stride for first convolution
            downsample: Downsample module for skip connection
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SimplifiedMultiScaleCNN(nn.Module):
    """
    Lightweight multi-scale CNN encoder for small datasets.
    
    Designed to prevent overfitting on limited training data while
    capturing hierarchical features at multiple scales.
    """
    
    def __init__(self, in_channels=1, global_dim=128, local_channels=64):
        """
        Initialize multi-scale CNN.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            global_dim: Dimension of global embedding
            local_channels: Channels in final local feature map
        """
        super().__init__()
        self.global_dim = global_dim
        self.local_channels = local_channels

        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout2d(0.2)
        )

        # Progressive downsampling layers
        self.layer1 = self._make_layer(32, 32, 1)
        self.layer2 = self._make_layer(32, 64, 1, stride=2)
        self.layer3 = self._make_layer(64, local_channels, 1, stride=2)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_global = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(local_channels, global_dim)
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(self._residual_block(in_ch, out_ch, stride))
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _residual_block(self, in_ch, out_ch, stride=1):
        """Create a single residual block with optional downsampling."""
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

        return ResidualBlock(in_ch, out_ch, stride, downsample)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            global_emb: Global embeddings [B, global_dim]
            local_feat: Local feature maps [B, local_channels, H', W']
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        local_feat = self.layer3(x)

        global_pooled = self.global_pool(local_feat)
        global_emb = self.fc_global(global_pooled.squeeze(-1).squeeze(-1))

        return global_emb, local_feat

