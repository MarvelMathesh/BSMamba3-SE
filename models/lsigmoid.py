"""
Learnable Sigmoid Activations
══════════════════════════════
Bounded non-linearity with learnable slope for magnitude masking.
Reference: MP-SENet (Lu et al., 2023)
"""

import torch
import torch.nn as nn


class LearnableSigmoid1D(nn.Module):
    """Per-feature learnable sigmoid for 1D outputs (discriminator)."""

    def __init__(self, in_features: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2D(nn.Module):
    """
    Per-frequency learnable sigmoid for 2D spectral outputs (magnitude mask).
    
    With beta=2.0, the mask output range is [0, 2.0], allowing the model to
    AMPLIFY weak speech components — impossible with tanh which caps at 1.0.
    This is critical for recovering quiet consonants and weak harmonics.
    """

    def __init__(self, in_features: int, beta: float = 2.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)
