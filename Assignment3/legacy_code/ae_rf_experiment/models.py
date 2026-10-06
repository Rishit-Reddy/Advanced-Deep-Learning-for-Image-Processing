"""
DenseNet-121 supervised classifier — STUB, not wired into the pipeline.

This is approach (a) from the assignment: a supervised DenseNet-121 that
accepts 6-channel (BF + FL) input. The project description mentions loading
RadImageNet weights into the first dense block, but that requires downloading
the RadImageNet checkpoint separately and copying matching parameter tensors —
that loading code is not implemented here.

Until a training script for this model is added, treat this file as a
structural reference only. Do not import it from train.py or helper.py.
"""

import torch.nn as nn
from torchvision import models


class DenseNet121Classifier(nn.Module):
    """6-channel (BF+FL) DenseNet-121 for binary cancer classification."""

    def __init__(self):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(6)

        base = models.densenet121(weights=None)  # RadImageNet loading not implemented
        old  = base.features.conv0
        base.features.conv0 = nn.Conv2d(
            6, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=False,
        )
        base.classifier = nn.Linear(1024, 1)
        self.base = base

    def forward(self, x):
        return self.base(self.input_norm(x))
