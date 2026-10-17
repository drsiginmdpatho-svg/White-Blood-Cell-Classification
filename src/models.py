"""
src/models.py

Neural network model architectures for White Blood Cell (WBC) classification:
1. BaselineCNN: Custom 4-stage convolutional neural network with BatchNorm and Dropout.
2. TransferLearningModel: Wrapper for pretrained architectures (MobileNetV2, ResNet50).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any


class BaselineCNN(nn.Module):
    """
    Custom 4-block Convolutional Neural Network baseline for 5-class WBC classification.
    
    Architecture:
    - Conv Block 1: Conv(3->32, 3x3), BN, ReLU, MaxPool(2x2) -> (32, 112, 112)
    - Conv Block 2: Conv(32->64, 3x3), BN, ReLU, MaxPool(2x2) -> (64, 56, 56)
    - Conv Block 3: Conv(64->128, 3x3), BN, ReLU, MaxPool(2x2) -> (128, 28, 28)
    - Conv Block 4: Conv(128->256, 3x3), BN, ReLU, AdaptiveAvgPool(4x4) -> (256, 4, 4)
    - Classifier: Flatten -> Linear(4096->256), BN, ReLU, Dropout(0.4) -> Linear(256->5)
    """
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.4):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


def get_transfer_model(
    model_name: str = "mobilenet_v2", 
    num_classes: int = 5, 
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Instantiates a transfer learning model with modified classification head.
    
    Args:
        model_name: 'mobilenet_v2' or 'resnet50'
        num_classes: Number of target categories (5)
        pretrained: If True, loads ImageNet pretrained weights
        freeze_backbone: If True, freezes feature extraction layers
    """
    if model_name.lower() == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
    elif model_name.lower() == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Use 'mobilenet_v2' or 'resnet50'.")
        
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Returns total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }
