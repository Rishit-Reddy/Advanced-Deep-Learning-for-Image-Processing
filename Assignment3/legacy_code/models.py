import torch
import torch.nn as nn
from torchvision import models

class LearnableNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial normalization layer for the 6 input channels
        self.input_norm = nn.BatchNorm2d(6)
        
        # Base DenseNet121 architecture (no pretrained weights)
        self.base_model = models.densenet121(weights=None)
        
        # Replace the first convolutional layer to accept 6 channels
        old_conv = self.base_model.features.conv0
        self.base_model.features.conv0 = nn.Conv2d(6, old_conv.out_channels, 
                                                  kernel_size=old_conv.kernel_size, 
                                                  stride=old_conv.stride, 
                                                  padding=old_conv.padding, 
                                                  bias=False)
        
        # Classification head for binary output
        self.base_model.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        # Apply learnable normalization first
        x = self.input_norm(x)
        return self.base_model(x)

def get_model(ckpt_path=None):
    return LearnableNormModel()
