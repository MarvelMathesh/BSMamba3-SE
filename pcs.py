"""
Perceptual Contrast Stretching (PCS)
═════════════════════════════════════
Training-time target modification based on SII critical band importance weights.

PCS applies frequency-dependent gamma correction to the target magnitude spectrum:
  S_pcs[f] = S[f]^γ(f)
where γ(f) is derived from the SII (Speech Intelligibility Index) importance function.

This is a TARGET modification, not a loss weight. The model learns to weight
perceptually important frequency bands more strongly. Zero inference overhead.

Expected gain: +0.10-0.14 WB-PESQ (from SEMamba ablation: 3.55 → 3.69).
Reference: Chao et al. 2022 (PCS technique), SEMamba implementation.

D6: PCS is the highest-gain single trick for Mamba SE. Not using it would
leave 0.14 PESQ on the table for free.
"""

import numpy as np
import torch
from typing import Optional


class PCSTargetTransform:
    """
    Perceptual Contrast Stretching applied to training targets.
    
    NOT an nn.Module — applied in DataLoader, zero GPU cost.
    
    SII critical band importance weights (ANSI S3.5-1997):
      21 critical bands from 150 Hz to 8500 Hz.
      Higher importance for speech-critical bands (1000-4000 Hz).
    
    Implementation:
      1. Map STFT bins to SII critical bands
      2. Compute per-bin gamma = SII_weight^pcs_gamma
      3. Apply: target_mag_pcs = target_mag ^ gamma
    """
    
    # SII Critical Band center frequencies and importance weights
    # From ANSI S3.5-1997, Table 3 (Band Importance Function for average speech)
    SII_CENTER_FREQ = np.array([
        150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370,
        1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500
    ], dtype=np.float64)
    
    SII_IMPORTANCE = np.array([
        0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711,
        0.0818, 0.0844, 0.0882, 0.0898, 0.0868, 0.0844, 0.0771, 0.0527,
        0.0364, 0.0185, 0.0000, 0.0000, 0.0000
    ], dtype=np.float64)
    
    def __init__(
        self,
        n_fft: int = 512,
        sr: int = 16000,
        pcs_gamma: float = 0.3,
        enabled: bool = True,
    ):
        """
        Args:
            n_fft: FFT size (determines number of freq bins)
            sr: sample rate
            pcs_gamma: exponent for SII importance → gamma mapping.
                       0.3 is conservative (from SEMamba paper).
                       Do NOT use 1.0+ (causes fricative degradation, D6 failure mode 4).
            enabled: set False for ablation A7
        """
        self.n_fft = n_fft
        self.sr = sr
        self.pcs_gamma = pcs_gamma
        self.enabled = enabled
        
        # Compute per-bin gamma weights
        self._compute_gamma_weights()
    
    def _compute_gamma_weights(self):
        """
        Map STFT frequency bins to SII importance weights,
        then compute gamma correction exponents.
        """
        n_bins = self.n_fft // 2 + 1  # 257 for n_fft=512
        bin_freqs = np.linspace(0, self.sr / 2, n_bins)  # [0, 8000] Hz
        
        # Interpolate SII importance to STFT bin frequencies
        sii_weights = np.interp(
            bin_freqs,
            self.SII_CENTER_FREQ,
            self.SII_IMPORTANCE,
            left=self.SII_IMPORTANCE[0],
            right=0.0
        )
        
        # Normalize to [0, 1] range
        sii_max = sii_weights.max()
        if sii_max > 0:
            sii_weights_norm = sii_weights / sii_max
        else:
            sii_weights_norm = np.ones_like(sii_weights)
        
        # Gamma = 1 - sii_weight^pcs_gamma * 0.5
        # This means: high-importance bands get gamma < 1 (expansion),
        # low-importance bands get gamma ≈ 1 (no change)
        # The stretching emphasizes perceptually critical frequencies.
        self.gamma_weights = 1.0 - (sii_weights_norm ** self.pcs_gamma) * 0.5
        
        # Ensure gamma stays in reasonable range [0.3, 1.0]
        self.gamma_weights = np.clip(self.gamma_weights, 0.3, 1.0)
        
        # Convert to torch tensor
        self.gamma_tensor = torch.from_numpy(
            self.gamma_weights.astype(np.float32)
        )  # [257]
    
    def apply(self, target_mag: torch.Tensor) -> torch.Tensor:
        """
        Apply PCS gamma correction to target magnitude spectrum.
        
        Args:
            target_mag: [B, T, F] or [B, F, T] magnitude spectrum (positive values)
        
        Returns:
            pcs_mag: same shape, with PCS-modified magnitudes
        """
        if not self.enabled:
            return target_mag
        
        gamma = self.gamma_tensor.to(target_mag.device)
        
        # Handle both [B, T, F] and [B, F, T] layouts
        if target_mag.shape[-1] == len(gamma):
            # [B, T, F] layout
            return target_mag.clamp(min=1e-10) ** gamma.unsqueeze(0).unsqueeze(0)
        elif target_mag.shape[1] == len(gamma):
            # [B, F, T] layout
            return target_mag.clamp(min=1e-10) ** gamma.unsqueeze(0).unsqueeze(-1)
        else:
            # Fallback: no PCS (dimension doesn't match)
            return target_mag
    
    def apply_to_complex_stft(
        self,
        target_stft: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply PCS to complex STFT by modifying magnitude while preserving phase.
        
        Args:
            target_stft: [B, F, T] complex STFT
        
        Returns:
            pcs_stft: [B, F, T] complex STFT with PCS-modified magnitudes
        """
        if not self.enabled:
            return target_stft
        
        mag = target_stft.abs()          # [B, F, T]
        phase = target_stft.angle()       # [B, F, T]
        
        pcs_mag = self.apply(mag)         # [B, F, T]
        
        # Reconstruct complex with modified magnitude and original phase
        return pcs_mag * torch.exp(1j * phase)

    def get_loss_weights(self) -> torch.Tensor:
        """
        Get per-frequency loss weights derived from PCS importance.

        For CRM-based models, PCS should be implemented as frequency-dependent
        LOSS WEIGHTING, NOT target modification. This avoids the train/eval
        mismatch where modified targets don't match PESQ evaluation references.

        Higher weight for perceptually important bands (1-4 kHz speech formants).
        Normalized so mean = 1.0 (no change to total loss magnitude).

        Returns:
            weights: [n_fft//2 + 1] float32 tensor, mean-normalized
        """
        n_bins = self.n_fft // 2 + 1
        if not self.enabled:
            return torch.ones(n_bins)

        # Inverse gamma: lower gamma = more important = higher weight
        weights = 1.0 / np.clip(self.gamma_weights, 0.3, 1.0)
        weights = weights / weights.mean()  # Normalize to mean=1
        return torch.from_numpy(weights.astype(np.float32))
