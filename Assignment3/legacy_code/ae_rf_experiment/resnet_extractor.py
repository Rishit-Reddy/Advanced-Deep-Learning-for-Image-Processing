"""
Dual-stream ResNet-50 feature extractor for BF + FL cell images.

Architecture
------------
BF image (3 ch) ──┐
                   ├─► shared frozen ResNet-50 backbone (×2 calls)
FL image (3 ch) ──┘         │
                             ▼
                    GAP → 2048-dim each
                             │
                    concat → 4096-dim feature vector
                             │
                        Random Forest

Why frozen?
-----------
Fine-tuning 25M ResNet parameters on 8 training patients will overfit to
patient-level staining/illumination artefacts far faster than it learns
cancer biology. Frozen ImageNet features + RF is safer and often better
with this dataset size.

Why dual-stream?
----------------
BF and FL look very different (brightfield vs fluorescence). Processing them
through the same frozen backbone separately preserves both modalities without
losing ImageNet weights by forcing a 6-channel first convolution.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        # Pretrained ResNet-50 with the final FC layer removed.
        # Output of backbone: (B, 2048, 1, 1) after the built-in GAP.
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze all weights — no training needed, no overfitting risk.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (B, 6, H, W)  —  channels 0-2 = BF, channels 3-5 = FL
        bf_feat = self.backbone(x[:, :3]).flatten(1)          # (B, 2048)
        fl_feat = self.backbone(x[:, 3:]).flatten(1)          # (B, 2048)
        return torch.cat([bf_feat, fl_feat], dim=1)           # (B, 4096)
