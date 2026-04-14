"""
BSMamba3-SE Loss Functions
══════════════════════════
Multi-objective training loss with 7 components:

  L = λ_metric·L_metric + λ_mag·L_mag + λ_phase·L_phase
    + λ_complex·L_complex + λ_time·L_time
    + λ_consistency·L_consistency + λ_tc·L_tc

Loss rationale:
  L_mag:         L2 magnitude in compressed domain → PESQ sensitivity
  L_phase:       Anti-wrapping phase loss (IP+GD+IAF) → phase recovery
  L_complex:     L2 complex in compressed domain → joint mag+phase
  L_time:        L1 waveform → prevents time-domain artifacts
  L_consistency: STFT round-trip → enforces valid spectrograms
  L_tc:          Temporal coherence → smooth harmonic trajectories (Mamba3-specific)
  L_metric:      MetricGAN adversarial → direct PESQ optimization
"""

import torch
import torch.nn.functional as F
import numpy as np


def anti_wrapping_function(x: torch.Tensor) -> torch.Tensor:
    """
    Anti-wrapping function to handle phase discontinuities at ±π.
    Maps arbitrary phase differences to [0, π].
    """
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(phase_r: torch.Tensor, phase_g: torch.Tensor, n_fft: int):
    """
    Three-component anti-wrapping phase loss from MP-SENet.

    Args:
        phase_r: [B, F, T] reference phase
        phase_g: [B, F, T] generated phase
        n_fft: FFT size (for frequency dimension)

    Returns:
        ip_loss: instantaneous phase loss
        gd_loss: group delay loss (phase derivative across frequency)
        iaf_loss: instantaneous frequency loss (phase derivative across time)
    """
    dim_freq = n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    # Group delay matrix: finite difference across frequency
    gd_matrix = (
        torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1)
        - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2)
        - torch.eye(dim_freq)
    ).to(phase_g.device)

    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    # Instantaneous frequency matrix: finite difference across time
    iaf_matrix = (
        torch.triu(torch.ones(dim_time, dim_time), diagonal=1)
        - torch.triu(torch.ones(dim_time, dim_time), diagonal=2)
        - torch.eye(dim_time)
    ).to(phase_g.device)

    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    # Anti-wrapped losses
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))

    return ip_loss, gd_loss, iaf_loss


def temporal_coherence_loss(mag_pred: torch.Tensor, mag_target: torch.Tensor) -> torch.Tensor:
    """
    Architecture-aligned temporal coherence loss (L_tc).

    Penalizes difference between frame-to-frame magnitude changes in
    prediction vs reference. Specifically motivated by Mamba3's complex
    rotation states: smooth harmonic trajectories are exactly what the
    trapezoidal-discretized complex SSM produces, so L_tc rewards the
    architectural inductive bias.

    Args:
        mag_pred: [B, F, T] predicted magnitude
        mag_target: [B, F, T] target magnitude

    Returns:
        tc_loss: scalar temporal coherence loss
    """
    # First-order temporal difference
    delta_pred = mag_pred[:, :, 1:] - mag_pred[:, :, :-1]
    delta_target = mag_target[:, :, 1:] - mag_target[:, :, :-1]
    return F.l1_loss(delta_pred, delta_target)
