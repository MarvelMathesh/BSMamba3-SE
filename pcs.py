"""
Perceptual Contrast Stretching (PCS)
════════════════════════════════════
Training-time target modification with SII-weighted gamma correction.

Applies frequency-dependent power-law compression to clean magnitude targets,
emphasizing perceptually important bands (1–4 kHz formant region).

Zero inference overhead: PCS only modifies training targets.
"""

import numpy as np
import librosa
import scipy.signal


# SII critical band importance weights (ANSI S3.5-1997)
# 21 bands with center frequencies and importance weights
SII_BANDS = [
    (150, 0.0083), (250, 0.0095), (350, 0.0150), (450, 0.0289),
    (570, 0.0440), (700, 0.0578), (840, 0.0653), (1000, 0.0711),
    (1170, 0.0818), (1370, 0.0844), (1600, 0.0882), (1850, 0.0898),
    (2150, 0.0868), (2500, 0.0844), (2900, 0.0771), (3400, 0.0527),
    (4000, 0.0364), (4800, 0.0185), (5800, 0.0000), (7000, 0.0000),
    (8500, 0.0000),
]


def compute_sii_weights(n_fft: int = 400, sr: int = 16000) -> np.ndarray:
    """
    Interpolate SII importance weights to STFT frequency bins.
    
    Returns:
        weights: [F] array of per-bin SII importance weights
    """
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    
    sii_freqs = np.array([f for f, _ in SII_BANDS])
    sii_weights = np.array([w for _, w in SII_BANDS])
    
    # Linear interpolation to STFT bins
    weights = np.interp(freqs, sii_freqs, sii_weights)
    return weights


def compute_pcs_gamma(n_fft: int = 400, sr: int = 16000, gamma: float = 0.3) -> np.ndarray:
    """
    Compute per-frequency-bin gamma exponent for PCS.
    
    gamma(f) = 1 - (w_SII(f) / max(w_SII))^gamma0 × 0.5
    
    High-importance bands: gamma ≈ 0.5 (square-root expansion)
    Low-importance bands:  gamma ≈ 1.0 (no modification)
    
    Returns:
        gammas: [F] per-bin gamma exponents, clipped to [0.3, 1.0]
    """
    weights = compute_sii_weights(n_fft, sr)
    max_w = weights.max()
    
    if max_w < 1e-10:
        return np.ones(n_fft // 2 + 1)
    
    normalized = (weights / max_w) ** gamma
    gammas = 1.0 - normalized * 0.5
    gammas = np.clip(gammas, 0.3, 1.0)
    
    return gammas


def apply_pcs_to_waveform(
    signal: np.ndarray,
    n_fft: int = 400,
    hop_size: int = 100,
    win_size: int = 400,
    gamma: float = 0.3,
) -> np.ndarray:
    """
    Apply PCS to a clean waveform as training target modification.
    
    Pipeline:
      1. STFT → magnitude + phase
      2. Apply per-bin gamma correction to magnitude
      3. iSTFT → modified waveform
    
    Args:
        signal: clean waveform (numpy array)
        n_fft, hop_size, win_size: STFT parameters
        gamma: PCS exponent (0.3 = conservative)
    
    Returns:
        pcs_signal: PCS-modified waveform (numpy array)
    """
    # Compute per-bin gamma exponents
    gammas = compute_pcs_gamma(n_fft, sr=16000, gamma=gamma)
    
    # STFT
    window = scipy.signal.windows.hann(win_size)
    S = librosa.stft(
        signal, n_fft=n_fft, hop_length=hop_size,
        win_length=win_size, window=window,
    )
    
    mag = np.abs(S)
    phase = np.angle(S)
    
    # Apply per-frequency gamma correction
    # gammas shape: [F], mag shape: [F, T]
    eps = 1e-10
    mag_pcs = np.power(mag + eps, gammas[:, np.newaxis])
    
    # Reconstruct
    S_pcs = mag_pcs * np.exp(1j * phase)
    signal_pcs = librosa.istft(
        S_pcs, hop_length=hop_size, win_length=win_size,
        window=window, length=len(signal),
    )
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(signal_pcs))
    if max_val > 0:
        signal_pcs = signal_pcs / max_val
    
    return signal_pcs.astype(np.float32)
