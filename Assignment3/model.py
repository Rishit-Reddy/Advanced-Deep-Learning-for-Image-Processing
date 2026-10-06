"""6-channel ResNet18 builder + freezing helper.

Pretrained 3-ch conv1 weights are duplicated and halved so the initial
activation magnitudes roughly match the original ImageNet pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

import config


class ResNet18DualModal(nn.Module):
    def __init__(
        self,
        freeze_early_blocks: bool = config.FREEZE_EARLY_BLOCKS,
        dropout_p: float = config.DROPOUT_P,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)

        # 3->6 channel conv1; preserve other hyperparams.
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=config.NUM_CHANNELS,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            w = old_conv1.weight.data  # [64,3,7,7]
            new_conv1.weight.copy_(torch.cat([w, w], dim=1) / 2.0)
        backbone.conv1 = new_conv1

        # Dropout + single-logit head.
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1),
        )

        self.backbone = backbone
        self.freeze_early_blocks = freeze_early_blocks
        self._frozen_modules: list[nn.Module] = []
        if freeze_early_blocks:
            self._apply_freezing()

    # --- freezing -------------------------------------------------------
    def _apply_freezing(self):
        self._frozen_modules = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
        ]
        for module in self._frozen_modules:
            for p in module.parameters():
                p.requires_grad = False
        self._set_frozen_bn_eval()

    def _set_frozen_bn_eval(self):
        for module in self._frozen_modules:
            for sub in module.modules():
                if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                    sub.eval()

    def train(self, mode: bool = True):
        # super().train() flips ALL BN to train. Re-pin frozen BN to eval
        # so running stats and affine params stay fixed.
        super().train(mode)
        if self.freeze_early_blocks:
            self._set_frozen_bn_eval()
        return self

    # --- forward --------------------------------------------------------
    def forward(self, x):
        return self.backbone(x)


def build_model() -> ResNet18DualModal:
    return ResNet18DualModal()


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
