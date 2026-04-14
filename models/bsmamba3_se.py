"""
BSMamba3-SE: Band-Split Mamba3 Speech Enhancement
══════════════════════════════════════════════════
Architecture: Compressed Mag+Phase → DenseEncoder → N×TFMamba3Block → MagDecoder+PhaseDecoder → ISTFT

Key innovations over SEMamba:
  1. Mamba3 trapezoidal discretization → zero phase-drift for harmonic tracking
  2. Complex-valued MIMO states → direct oscillatory mode modelling
  3. Bidirectional TF processing with Mamba3 on both time and frequency axes
  4. Architecture-aligned temporal coherence loss (L_tc)
  5. PCS with SII-weighted gamma correction

Engineering recipe from MP-SENet/SEMamba:
  - n_fft=400, hop=100, compress_factor=0.3
  - DenseBlock encoder with dilated convolutions
  - Separate MagDecoder (LearnableSigmoid) + PhaseDecoder (atan2)
  - MetricGAN adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

from models.lsigmoid import LearnableSigmoid2D


# ═══════════════════════════════════════════════════════════════════════════
# STFT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def mag_phase_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, addeps=False):
    """
    Compute magnitude and phase from STFT with optional power-law compression.

    Args:
        y: [B, L] or [B, 1, L] waveform
        compress_factor: power-law compression exponent (0.3 = cube root)

    Returns:
        mag: [B, F, T] compressed magnitude
        pha: [B, F, T] phase
        com: [B, F, T, 2] compressed complex (mag*cos, mag*sin)
    """
    eps = 1e-10
    if y.dim() == 3:
        y = y.squeeze(1)
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window, center=center, pad_mode='reflect',
        normalized=False, return_complex=True
    )

    if not addeps:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + eps)
        pha = torch.atan2(imag_part + eps, real_part + eps)

    # Power-law compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_phase_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    """
    Inverse STFT: decompress magnitude, combine with phase, reconstruct waveform.
    """
    mag = torch.pow(mag, 1.0 / compress_factor)
    # PyTorch torch.complex does not support bfloat16
    real_part = (mag * torch.cos(pha)).float()
    imag_part = (mag * torch.sin(pha)).float()
    com = torch.complex(real_part, imag_part)
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(
        com, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window, center=center
    )
    return wav


# ═══════════════════════════════════════════════════════════════════════════
# DENSE CONVOLUTIONAL BLOCKS
# ═══════════════════════════════════════════════════════════════════════════

def get_padding_2d(kernel_size, dilation=(1, 1)):
    """Calculate same-padding for 2D conv with dilation."""
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


class DenseBlock(nn.Module):
    """
    4-layer dilated dense convolution block with InstanceNorm and PReLU.
    
    Dilations: [1, 2, 4, 8] → effective receptive field of 15 frames.
    Dense connectivity: each layer receives all previous outputs concatenated.
    
    This captures local spectral patterns (formants, harmonics) before
    Mamba3 processes long-range temporal dependencies.
    """

    def __init__(self, hid_feature: int, kernel_size=(3, 3), depth: int = 4):
        super().__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList()

        for i in range(depth):
            dil = 2 ** i  # 1, 2, 4, 8
            dense_conv = nn.Sequential(
                nn.Conv2d(
                    hid_feature * (i + 1), hid_feature, kernel_size,
                    dilation=(dil, 1),
                    padding=get_padding_2d(kernel_size, (dil, 1)),
                ),
                nn.InstanceNorm2d(hid_feature, affine=True),
                nn.PReLU(hid_feature),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    """
    Front-end encoder: Conv2d → DenseBlock → stride Conv2d.
    
    Input: [B, 2, T, F=201] (compressed mag + phase)
    Output: [B, C=64, T, F'=100] (downsampled in frequency)
    """

    def __init__(self, in_channel: int = 2, hid_feature: int = 64):
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, hid_feature, (1, 1)),
            nn.InstanceNorm2d(hid_feature, affine=True),
            nn.PReLU(hid_feature),
        )
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(hid_feature, hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(hid_feature, affine=True),
            nn.PReLU(hid_feature),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_conv_1(x)   # [B, C, T, F=201]
        x = self.dense_block(x)    # [B, C, T, F=201]
        x = self.dense_conv_2(x)   # [B, C, T, F'=100]
        return x


# ═══════════════════════════════════════════════════════════════════════════
# BIDIRECTIONAL MAMBA3 BLOCK
# ═══════════════════════════════════════════════════════════════════════════

class BiMamba3Block(nn.Module):
    """
    Bidirectional Mamba3 processing: forward + reverse pass concatenated.
    
    Mamba3 innovations exploited here:
      1. Trapezoidal (bilinear) discretization → zero phase-drift
      2. Complex-valued states → direct harmonic mode modelling
      3. MIMO rank-R coupling → cross-head information flow
    
    Output is 2× input channels (forward + backward concatenated).
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        headdim: int = 16,
        mimo_rank: int = 4,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.norm_f = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

        from mamba_ssm import Mamba3
        self.mamba3_forward = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=True,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
        )
        self.mamba3_backward = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=True,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
        )

    def _pad_to_chunk(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad sequence length to multiple of chunk_size."""
        L = x.shape[1]
        pad = (self.chunk_size - L % self.chunk_size) % self.chunk_size
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        return x, pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [BN, L, C] where BN is batch*freq or batch*time
        Returns:
            y: [BN, L, 2C] bidirectional output
        """
        L_orig = x.shape[1]

        # Forward direction
        x_f, pad = self._pad_to_chunk(self.norm_f(x))
        y_f = self.mamba3_forward(x_f)
        if pad > 0:
            y_f = y_f[:, :L_orig, :]

        # Backward direction
        x_b = torch.flip(x, [1])
        x_b, pad = self._pad_to_chunk(self.norm_b(x_b))
        y_b = self.mamba3_backward(x_b)
        if pad > 0:
            y_b = y_b[:, :L_orig, :]
        y_b = torch.flip(y_b, [1])

        return torch.cat([y_f, y_b], dim=-1)  # [BN, L, 2C]


# ═══════════════════════════════════════════════════════════════════════════
# TIME-FREQUENCY MAMBA3 BLOCK
# ═══════════════════════════════════════════════════════════════════════════

class TFMamba3Block(nn.Module):
    """
    Dual-axis bidirectional Mamba3 block processing BOTH time and frequency.
    
    1. Time-Mamba3: each frequency bin processed temporally [B*F', T, C] → bidirectional
    2. Freq-Mamba3: each time frame processed spectrally [B*T, F', C] → bidirectional
    
    Both with residual connections via ConvTranspose1d projection (2C → C).
    
    This is the core architectural contribution: Mamba3 (trapezoidal, complex, MIMO)
    replaces Mamba1 (ZOH, real, SISO) on both axes.
    """

    def __init__(self, hid_feature: int = 64, d_state: int = 16,
                 headdim: int = 16, mimo_rank: int = 4, chunk_size: int = 16):
        super().__init__()
        self.hid_feature = hid_feature

        # Bidirectional Mamba3 for temporal axis
        self.time_mamba = BiMamba3Block(
            d_model=hid_feature, d_state=d_state, headdim=headdim,
            mimo_rank=mimo_rank, chunk_size=chunk_size,
        )
        # Bidirectional Mamba3 for frequency axis
        self.freq_mamba = BiMamba3Block(
            d_model=hid_feature, d_state=d_state, headdim=headdim,
            mimo_rank=mimo_rank, chunk_size=chunk_size,
        )

        # Projection from 2C → C (concat of forward+backward)
        self.tlinear = nn.ConvTranspose1d(hid_feature * 2, hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(hid_feature * 2, hid_feature, 1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F'] encoded features
        Returns:
            x: [B, C, T, F'] with TF-Mamba3 processing
        """
        b, c, t, f = x.size()

        # === Temporal Mamba3 ===
        # Reshape: [B, C, T, F'] → [B*F', T, C]
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        # BiMamba3 → [B*F', T, 2C], project → [B*F', T, C], residual
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x

        # === Frequency Mamba3 ===
        # Reshape: [B*F', T, C] → [B, F', T, C] → [B, T, F', C] → [B*T, F', C]
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        # BiMamba3 → [B*T, F', 2C], project → [B*T, F', C], residual
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x

        # Reshape back: [B*T, F', C] → [B, T, F', C] → [B, C, T, F']
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)

        return x


# ═══════════════════════════════════════════════════════════════════════════
# DECODERS
# ═══════════════════════════════════════════════════════════════════════════

class MagDecoder(nn.Module):
    """
    Magnitude mask decoder with LearnableSigmoid activation.
    
    Output range: [0, beta=2.0] — can AMPLIFY weak speech components.
    DenseBlock → ConvTranspose2d (upsample F'→F) → Conv2d → LearnableSigmoid2D
    """

    def __init__(self, hid_feature: int = 64, n_fft: int = 400, beta: float = 2.0):
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, depth=4)
        self.n_fft = n_fft

        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(hid_feature, hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(hid_feature, 1, (1, 1)),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1),
            nn.Conv2d(1, 1, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2D(n_fft // 2 + 1, beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F'=100] encoder features
        Returns:
            mask: [B, 1, T, F=201] magnitude mask via LearnableSigmoid
        """
        x = self.dense_block(x)       # [B, C, T, F'=100]
        x = self.mask_conv(x)          # [B, 1, T, F=201]
        # Apply per-frequency learnable sigmoid
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)   # [B, F, T]
        x = self.lsigmoid(x)                                  # [B, F, T]
        x = rearrange(x, 'b f t -> b t f').unsqueeze(1)       # [B, 1, T, F]
        return x


class PhaseDecoder(nn.Module):
    """
    Phase prediction decoder via atan2(imag, real).
    
    Directly predicts clean phase without wrapping issues.
    DenseBlock → ConvTranspose2d (upsample) → atan2
    """

    def __init__(self, hid_feature: int = 64):
        super().__init__()
        self.dense_block = DenseBlock(hid_feature, depth=4)

        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(hid_feature, hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(hid_feature, affine=True),
            nn.PReLU(hid_feature),
        )
        self.phase_conv_r = nn.Conv2d(hid_feature, 1, (1, 1))
        self.phase_conv_i = nn.Conv2d(hid_feature, 1, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F'=100] encoder features
        Returns:
            phase: [B, 1, T, F=201] predicted phase via atan2
        """
        x = self.dense_block(x)          # [B, C, T, F'=100]
        x = self.phase_conv(x)           # [B, C, T, F=201]
        x_r = self.phase_conv_r(x)       # [B, 1, T, F=201]
        x_i = self.phase_conv_i(x)       # [B, 1, T, F=201]
        phase = torch.atan2(x_i, x_r)   # [B, 1, T, F=201]
        return phase


# ═══════════════════════════════════════════════════════════════════════════
# BSMamba3-SE: FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════

class BSMamba3SE(nn.Module):
    """
    Band-Split Mamba3 Speech Enhancement.

    Pipeline:
      1. Input: noisy_mag [B, F, T], noisy_pha [B, F, T] (pre-compressed)
      2. Concat → [B, 2, T, F] → DenseEncoder → [B, C, T, F']
      3. N × TFMamba3Block (bidirectional Mamba3 on time + freq axes)
      4. MagDecoder → magnitude mask (LearnableSigmoid, range [0, 2.0])
      5. PhaseDecoder → clean phase (atan2)
      6. denoised_mag = mask × noisy_mag
      7. denoised_com = mag * exp(j * phase)
    
    Output: denoised_mag, denoised_pha, denoised_com
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hid = cfg.hid_feature
        num_blocks = cfg.num_tfmamba

        # Encoder
        self.dense_encoder = DenseEncoder(
            in_channel=cfg.input_channel,
            hid_feature=hid,
        )

        # Backbone: N × TFMamba3 blocks
        self.tf_blocks = nn.ModuleList([
            TFMamba3Block(
                hid_feature=hid,
                d_state=cfg.d_state,
                headdim=cfg.headdim,
                mimo_rank=cfg.mimo_rank,
                chunk_size=cfg.chunk_size,
            )
            for _ in range(num_blocks)
        ])

        # Decoders
        self.mask_decoder = MagDecoder(
            hid_feature=hid,
            n_fft=cfg.n_fft,
            beta=cfg.beta,
        )
        self.phase_decoder = PhaseDecoder(hid_feature=hid)

    def forward(
        self, noisy_mag: torch.Tensor, noisy_pha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_mag: [B, F=201, T] compressed noisy magnitude
            noisy_pha: [B, F=201, T] noisy phase

        Returns:
            denoised_mag: [B, F, T] enhanced compressed magnitude
            denoised_pha: [B, F, T] predicted clean phase
            denoised_com: [B, F, T, 2] complex representation (for complex loss)
        """
        # Reshape: [B, F, T] → [B, 1, T, F]
        noisy_mag_in = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)
        noisy_pha_in = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)

        # Concat mag + phase → [B, 2, T, F]
        x = torch.cat((noisy_mag_in, noisy_pha_in), dim=1)

        # Encode
        x = self.dense_encoder(x)  # [B, C, T, F'=100]

        # Backbone: TFMamba3 blocks
        for block in self.tf_blocks:
            x = block(x)

        # Decode magnitude mask
        # mask: [B, 1, T, F] → multiply with noisy_mag → denoised_mag
        mask = self.mask_decoder(x)
        denoised_mag = rearrange(
            mask * noisy_mag_in, 'b c t f -> b f t c'
        ).squeeze(-1)  # [B, F, T]

        # Decode phase
        denoised_pha = rearrange(
            self.phase_decoder(x), 'b c t f -> b f t c'
        ).squeeze(-1)  # [B, F, T]

        # Combine into complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha),
             denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )  # [B, F, T, 2]

        return denoised_mag, denoised_pha, denoised_com

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
