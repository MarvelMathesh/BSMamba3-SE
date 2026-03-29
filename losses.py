"""
BSMamba3-SE Loss Functions
═══════════════════════════
Multi-objective training loss with explicit theoretical justification for each term.

Total Loss:
  L = λ₁·L_mag + λ₂·L_complex + λ₃·L_sisnr + λ₄·L_tc

Each term targets a specific aspect of speech quality:
  L_mag:     Magnitude fidelity → PESQ magnitude sensitivity
  L_complex: Phase recovery     → PESQ phase sensitivity above 3.0
  L_sisnr:   Waveform fidelity  → prevents low-energy artifacts
  L_tc:      Temporal coherence  → smooth harmonic trajectories (Mamba3-specific)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MultiScaleLoss(nn.Module):
    """
    Combined multi-objective loss for speech enhancement training.

    Loss weights:
      λ₁ = 15.0 (L_mag)      — dominant early; PESQ is magnitude-sensitive below 3.0
      λ₂ = 10.0 (L_complex)  — phase recovery for PESQ > 3.0
      λ₃ = 0.5  (L_sisnr)    — waveform anchor, prevents spectral-only optimization
      λ₄ = 5.0  (L_tc)       — temporal coherence, motivated by Mamba3 complex states
    """

    def __init__(
        self,
        lambda_mag: float = 15.0,
        lambda_complex: float = 10.0,
        lambda_sisnr: float = 0.5,
        lambda_tc: float = 5.0,
        stft_scales: Optional[list] = None,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.lambda_mag = lambda_mag
        self.lambda_complex = lambda_complex
        self.lambda_sisnr = lambda_sisnr
        self.lambda_tc = lambda_tc
        self.eps = eps

        # STFT configs for multi-scale magnitude loss
        if stft_scales is None:
            self.stft_scales = [
                (512, 256, 512),
                (256, 128, 256),
                (128, 64, 128),
            ]
        else:
            self.stft_scales = stft_scales

        # Register Hann windows
        for i, (n_fft, _, win_length) in enumerate(self.stft_scales):
            self.register_buffer(f'loss_window_{i}', torch.hann_window(win_length))

    def _compute_stft(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Compute complex STFT for a given scale."""
        n_fft, hop_length, win_length = self.stft_scales[scale_idx]
        window = getattr(self, f'loss_window_{scale_idx}')
        return torch.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True, center=True, pad_mode='reflect',
            normalized=False
        )

    def magnitude_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale log-magnitude L1 loss.

        L_mag = (1/3) Σ_{s∈scales} ||log(|STFT_s(ŝ)|+ε) - log(|STFT_s(s)|+ε)||₁

        WHY L1 not L2: L1 is robust to outliers (transient frames where magnitude
        can change by 30+ dB in a single frame). L2 would over-penalize these.
        WHY log-magnitude: human perception is approximately logarithmic in amplitude.
        """
        total_loss = 0.0
        for i in range(len(self.stft_scales)):
            pred_stft = self._compute_stft(pred, i)    # [B, F, T]
            tgt_stft = self._compute_stft(target, i)   # [B, F, T]

            pred_mag = torch.log(pred_stft.abs() + self.eps)
            tgt_mag = torch.log(tgt_stft.abs() + self.eps)

            total_loss = total_loss + F.l1_loss(pred_mag, tgt_mag)

        return total_loss / len(self.stft_scales)

    def complex_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Complex STFT L1 loss (real and imaginary separately).

        L_complex = ||Re(STFT(ŝ)) - Re(STFT(s))||₁ + ||Im(STFT(ŝ)) - Im(STFT(s))||₁

        WHY: Phase recovery is required for WB-PESQ > 3.0. ITU-T P.862.2
        weights phase errors increasingly at high PESQ scores. Training directly
        on complex STFT provides gradients through both magnitude and phase.
        Reference: MP-SENet parallel mag+phase loss.
        """
        # Use primary scale only (512-pt STFT)
        pred_stft = self._compute_stft(pred, 0)    # [B, 257, T]
        tgt_stft = self._compute_stft(target, 0)   # [B, 257, T]

        loss_real = F.l1_loss(pred_stft.real, tgt_stft.real)
        loss_imag = F.l1_loss(pred_stft.imag, tgt_stft.imag)

        return loss_real + loss_imag

    def si_snr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss.

        SI-SNR = 10 log₁₀(||s_target||² / ||e_noise||²)
        where s_target = (<ŝ, s>/<s, s>) · s  (projection)
              e_noise  = ŝ - s_target

        Return negative SI-SNR for minimization.

        WHY: Anchors waveform-level fidelity. Prevents the model from introducing
        low-energy artifacts that don't register in spectral losses but are
        audible as clicking/buzzing.
        """
        # Zero-mean normalization (scale-invariant)
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        # s_target = projection of pred onto target
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / s_target_energy) * target  # [B, L]

        # e_noise = pred - s_target
        e_noise = pred - s_target  # [B, L]

        # SI-SNR
        si_snr = 10 * torch.log10(
            torch.sum(s_target ** 2, dim=-1) /
            (torch.sum(e_noise ** 2, dim=-1) + self.eps)
            + self.eps
        )  # [B]

        return -si_snr.mean()

    def temporal_coherence_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Temporal coherence loss (first-difference on STFT magnitude).

        L_tc = ||Δ_t|STFT(ŝ)| - Δ_t|STFT(s)|||₁

        WHY SPECIFICALLY FOR MAMBA3: The complex states are designed to track
        oscillatory dynamics across time. This loss rewards smooth harmonic
        trajectories — exactly what the complex rotation state is optimized to
        produce. This is the only loss term specifically motivated by the Mamba3
        complex-state architecture contribution.
        """
        pred_stft = self._compute_stft(pred, 0)    # [B, 257, T]
        tgt_stft = self._compute_stft(target, 0)   # [B, 257, T]

        pred_mag = pred_stft.abs()  # [B, 257, T]
        tgt_mag = tgt_stft.abs()    # [B, 257, T]

        # First difference along time axis
        pred_diff = pred_mag[:, :, 1:] - pred_mag[:, :, :-1]  # [B, 257, T-1]
        tgt_diff = tgt_mag[:, :, 1:] - tgt_mag[:, :, :-1]     # [B, 257, T-1]

        return F.l1_loss(pred_diff, tgt_diff)

    def forward(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        pred_stft: Optional[torch.Tensor] = None,
        target_stft: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with all components.

        Args:
            pred_wav:    [B, L] enhanced waveform
            target_wav:  [B, L] clean target waveform
            pred_stft:   [B, F, T] (optional, for efficiency)
            target_stft: [B, F, T] (optional, for efficiency)

        Returns:
            total_loss: scalar
            loss_dict: {str: float} per-term losses for logging
        """
        # Ensure same length
        min_len = min(pred_wav.shape[-1], target_wav.shape[-1])
        pred_wav = pred_wav[..., :min_len]
        target_wav = target_wav[..., :min_len]

        # Compute individual losses
        l_mag = self.magnitude_loss(pred_wav, target_wav)
        l_complex = self.complex_loss(pred_wav, target_wav)
        l_sisnr = self.si_snr_loss(pred_wav, target_wav)
        l_tc = self.temporal_coherence_loss(pred_wav, target_wav)

        # Weighted sum
        total = (
            self.lambda_mag * l_mag +
            self.lambda_complex * l_complex +
            self.lambda_sisnr * l_sisnr +
            self.lambda_tc * l_tc
        )

        loss_dict = {
            'loss_total': total.item(),
            'loss_mag': l_mag.item(),
            'loss_complex': l_complex.item(),
            'loss_sisnr': l_sisnr.item(),
            'loss_tc': l_tc.item(),
        }

        return total, loss_dict
