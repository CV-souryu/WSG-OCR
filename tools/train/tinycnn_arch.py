"""PyTorch TinyCNN used for training (Conv + BatchNorm + ReLU).

The runtime model has no BatchNorm: ``tools/train/export_model.py`` folds every
BN into the preceding convolution's weight/bias, so the WGSL and numpy
inference paths only need Conv + ReLU.
"""

from __future__ import annotations

import torch
from torch import nn


class TinyCNNBN(nn.Module):
    """Mirror of the runtime TinyCNN with BatchNorm after every conv."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.dw1 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.bn_dw1 = nn.BatchNorm2d(8)
        self.pw1 = nn.Conv2d(8, 16, 1, stride=2)
        self.bn_pw1 = nn.BatchNorm2d(16)
        self.dw2 = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.bn_dw2 = nn.BatchNorm2d(16)
        self.pw2 = nn.Conv2d(16, 32, 1, stride=2)
        self.bn_pw2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn_dw1(self.dw1(x)))
        x = torch.relu(self.bn_pw1(self.pw1(x)))
        x = torch.relu(self.bn_dw2(self.dw2(x)))
        x = torch.relu(self.bn_pw2(self.pw2(x)))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def fold_bn_into_conv(
    conv: nn.Conv2d, bn: nn.BatchNorm2d
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold ``y = gamma * (x - mean)/sqrt(var+eps) + beta`` into conv.

    Returns folded ``(weight, bias)``; the result is bit-identical to
    running the conv followed by BN in eval mode (within float tolerance).
    """

    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    shift = bn.bias - bn.running_mean * scale
    shape = [1] * (conv.weight.dim() - 1)
    w = conv.weight * scale.view(-1, *shape)
    b = conv.bias * scale + shift
    return w, b


def export_folded_weights(model: TinyCNNBN) -> dict[str, torch.Tensor]:
    """Return runtime-format tensors (BN folded) for one module."""
    tensors: dict[str, torch.Tensor] = {}
    for name, conv, bn, shape in (
        ("conv1", model.conv1, model.bn1, (8, 1, 3, 3)),
        ("dw1", model.dw1, model.bn_dw1, (8, 3, 3)),
        ("pw1", model.pw1, model.bn_pw1, (16, 8)),
        ("dw2", model.dw2, model.bn_dw2, (16, 3, 3)),
        ("pw2", model.pw2, model.bn_pw2, (32, 16)),
    ):
        w, b = fold_bn_into_conv(conv, bn)
        tensors[f"{name}.weight"] = w.reshape(shape)
        tensors[f"{name}.bias"] = b
    tensors["fc.weight"] = model.fc.weight
    tensors["fc.bias"] = model.fc.bias
    return tensors
