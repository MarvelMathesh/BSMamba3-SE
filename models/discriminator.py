"""
MetricGAN Discriminator
═══════════════════════
Learns to predict PESQ scores, enabling adversarial perceptual optimization.
Reference: MetricGAN+ (Fu et al., 2021), MP-SENet (Lu et al., 2023)
"""

import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from models.lsigmoid import LearnableSigmoid1D


def pesq_loss(clean, noisy, sr=16000):
    """Compute WB-PESQ for a single utterance pair."""
    try:
        return pesq(sr, clean, noisy, 'wb')
    except Exception:
        return -1


def batch_pesq(clean_list, noisy_list, sr=16000, n_jobs=4):
    """
    Parallel PESQ computation across a batch.
    Returns normalized PESQ scores in [0, 1] or None if any utterance fails.
    """
    scores = Parallel(n_jobs=n_jobs)(
        delayed(pesq_loss)(c, n, sr)
        for c, n in zip(clean_list, noisy_list)
    )
    scores = np.array(scores)
    if -1 in scores:
        return None
    # Normalize: PESQ range [1, 4.5] → [0, 1]
    scores = (scores - 1) / 3.5
    return torch.FloatTensor(scores)


class MetricDiscriminator(nn.Module):
    """
    Spectral-norm Conv2D discriminator trained on real PESQ scores.
    
    Input: stacked (clean_mag, enhanced_mag) → [B, 2, F, T]
    Output: scalar quality score per utterance
    
    Training:
      D(clean, clean) → 1.0
      D(clean, enhanced) → normalized_pesq_score
    Generator pushes D(clean, enhanced) → 1.0
    """

    def __init__(self, dim: int = 16, in_channel: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True),
            nn.PReLU(dim * 2),
            nn.utils.spectral_norm(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True),
            nn.PReLU(dim * 4),
            nn.utils.spectral_norm(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True),
            nn.PReLU(dim * 8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim * 8, dim * 4)),
            nn.Dropout(0.3),
            nn.PReLU(dim * 4),
            nn.utils.spectral_norm(nn.Linear(dim * 4, 1)),
            LearnableSigmoid1D(1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: reference magnitude [B, F, T]
            y: enhanced magnitude [B, F, T]
        Returns:
            score: [B] quality prediction
        """
        xy = torch.stack((x, y), dim=1)  # [B, 2, F, T]
        return self.layers(xy)
