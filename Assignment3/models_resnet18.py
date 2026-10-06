import os

import torch
import torch.nn as nn
from torchvision import models


class ResNet18DualModal(nn.Module):
    def __init__(self, freeze_early_blocks=True, dropout_p=0.5):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)

        old_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            6,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )

        with torch.no_grad():
            pretrained_weight = old_conv1.weight.data
            backbone.conv1.weight.copy_(torch.cat([pretrained_weight, pretrained_weight], dim=1) / 2.0)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

        # Track which submodules are frozen so we can keep their BN layers in
        # eval mode even when the parent module is set to train().
        self._frozen_modules = []
        if freeze_early_blocks:
            self._freeze_early_blocks()

    def _freeze_early_blocks(self):
        self._frozen_modules = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
        ]
        for module in self._frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # BatchNorm running stats and affine params in frozen blocks must stay
        # fixed — otherwise BN keeps adapting even though weights are frozen.
        for module in self._frozen_modules:
            for sub in module.modules():
                if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                    sub.eval()
        return self

    def forward(self, x):
        return self.backbone(x)


def get_model(ckpt_path=None):
    model = ResNet18DualModal()

    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model
