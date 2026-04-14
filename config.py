"""
BSMamba3-SE Configuration
═════════════════════════
Single-source-of-truth for every hyperparameter.
"""

from dataclasses import dataclass


@dataclass
class BSMamba3Config:
    """
    Complete configuration for BSMamba3-SE.

    STFT Configuration (matched to MP-SENet/SEMamba proven recipe):
      n_fft=400 → F=201 bins, 25ms window
      hop_size=100 → 6.25ms hop, T≈321 frames for 2s segment
      compress_factor=0.3 → power-law magnitude compression

    Mamba3 Configuration:
      hid_feature=64 → CNN channels and Mamba3 d_model
      headdim=16 → n_heads = 64/16 = 4
      mimo_rank=4 → full MIMO coupling
      chunk_size=16 → 64/mimo_rank for bf16
    """

    # ── STFT ──
    sr: int = 16000
    n_fft: int = 400
    hop_size: int = 100
    win_size: int = 400
    compress_factor: float = 0.3  # Power-law magnitude compression (MP-SENet/SEMamba standard)

    # ── Model Architecture ──
    hid_feature: int = 64          # Dense layer channels / Mamba3 d_model
    num_tfmamba: int = 4           # Number of TFMamba3 blocks
    dense_depth: int = 4           # DenseBlock depth (dilations 1,2,4,8)

    # Mamba3-specific
    d_state: int = 16              # Complex-valued state dimension
    headdim: int = 16              # n_heads = hid_feature/headdim = 64/16 = 4
    mimo_rank: int = 4             # MIMO coupling rank (≤ n_heads)
    chunk_size: int = 16           # 64/mimo_rank for bf16

    # Decoder
    beta: float = 2.0              # LearnableSigmoid scale factor
    input_channel: int = 2         # Magnitude + Phase
    output_channel: int = 1        # Single-channel output

    # ── Training Recipe ──
    batch_size: int = 4
    epochs: int = 200
    lr: float = 5e-4
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.99         # ExponentialLR gamma per epoch
    segment_size: int = 32000      # 2 seconds at 16kHz
    grad_clip: float = 1.0         # Critical for complex SSM stability
    precision: str = 'bf16'        # MANDATORY for Mamba3 complex angle accumulation

    # ── Loss Weights ──
    lambda_metric: float = 0.05    # MetricGAN adversarial loss
    lambda_mag: float = 0.9        # L2 magnitude loss (compressed domain)
    lambda_phase: float = 0.3      # Anti-wrapping phase loss (IP+GD+IAF)
    lambda_complex: float = 0.1    # L2 complex loss (compressed domain)
    lambda_time: float = 0.2       # L1 time-domain waveform loss
    lambda_consistency: float = 0.1 # STFT round-trip consistency loss
    lambda_tc: float = 0.1         # Temporal coherence loss (our contribution)

    # ── PCS ──
    use_pcs: bool = True
    pcs_gamma: float = 0.3

    # ── Data ──
    data_dir: str = './VoiceBank_DEMAND_16k'

    # ── Checkpointing & Logging ──
    checkpoint_interval: int = 1000  # Save every N steps
    validation_interval: int = 1000  # Validate every N steps
    stdout_interval: int = 10        # Print every N steps
    summary_interval: int = 100      # Tensorboard every N steps
    out_dir: str = './checkpoints/bsmamba3'

    # ── Reproducibility ──
    seed: int = 1234
    num_workers: int = 4

    def validate(self):
        """Validate all configuration constraints."""
        n_heads = self.hid_feature // self.headdim
        assert self.hid_feature % self.headdim == 0, \
            f"hid_feature={self.hid_feature} must be divisible by headdim={self.headdim}"
        assert self.mimo_rank <= n_heads, \
            f"mimo_rank={self.mimo_rank} must be ≤ n_heads={n_heads}"
        assert self.chunk_size == 64 // self.mimo_rank, \
            f"chunk_size must be 64/mimo_rank={64 // self.mimo_rank} for bf16"
        assert self.precision in ('bf16', 'fp32'), \
            f"precision must be bf16 or fp32, got {self.precision}"
        print(f"[Config] n_fft={self.n_fft}, hop={self.hop_size}, compress={self.compress_factor}")
        print(f"[Config] hid_feature={self.hid_feature}, n_heads={n_heads}, mimo_rank={self.mimo_rank}")
        print(f"[Config] num_blocks={self.num_tfmamba}, d_state={self.d_state}")
        print(f"[Config] All constraints validated ✓")
        return True
