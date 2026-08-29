"""
src/model.py

A small CNN for MNIST digit classification, sized deliberately for an edge-deployment
demonstration: few channels, few layers -- the point of this assignment is the compression
pipeline (pruning/quantization/export/benchmarking), not squeezing out the last fraction of a
percent of accuracy with a large architecture.

Architecture: 2 conv blocks (with BatchNorm + ReLU, fusable for quantization) + 2 FC layers.
Uses Conv2d/Linear (not exotic ops) specifically because PyTorch's eager-mode static
quantization path supports exactly these layer types cleanly.
"""
import torch
import torch.nn as nn


class EdgeDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # QuantStub/DeQuantStub mark where the tensor enters/leaves quantized precision --
        # required by PyTorch's eager-mode static quantization workflow (Section 3.3).
        self.quant = torch.ao.quantization.QuantStub()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2)  # 14x14 -> 7x7

        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(32, num_classes)

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def fusable_layer_groups(self):
        """Layer name groups eligible for conv+bn+relu fusion before static quantization --
        fusing reduces the number of quantization boundary points and is standard practice
        ahead of PyTorch static PTQ."""
        return [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
