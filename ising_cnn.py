"""Define convolutional neural-network models for Ising configurations."""

import torch
from torch import nn


class IsingCNN(nn.Module):
    """Implement a shared CNN backbone with a configurable output head.

    The network expects tensors with shape ``(batch_size, 1, 20, 20)``.

    Args:
        output_size: Number of output values produced by the final layer.
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return model outputs for a batch of spin configurations.

        Args:
            inputs: Batch of channel-first Ising configurations.

        Returns:
            Output tensor produced by the network.
        """
        return self.classifier(self.features(inputs))


class IsingClassifier(IsingCNN):
    """Classify configurations into low- and high-temperature phases.

    Args:
        num_classes: Number of phase classes produced by the classifier.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__(output_size=num_classes)


class IsingRegressor(IsingCNN):
    """Estimate the absolute magnetization of a spin configuration."""

    def __init__(self) -> None:
        super().__init__(output_size=1)
