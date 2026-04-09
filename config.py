"""
BSMamba3-SE Configuration
═════════════════════════.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BSMamba3Config:
    """
    Complete configuration for BSMamba3-SE training.
    """
    
    # ── Model Architecture ──
    d_model: int = 256          # D1: n_heads=4 with headdim=64
    d_state: int = 16           # D5: must be multiple of 16 (MMA). 32 exceeds RTX 4050 SMEM
    headdim: int = 64           # D1: standard, well-validated
    mimo_rank: int = 2          # D4: K/R = 8/2 = 4 bands per MIMO unit. Reduced from 4 for larger d_state
    chunk_size: int = 32        # 64/mimo_rank for bf16
    n_blocks: int = 6           # D2: within 0.37s/step budget
    n_bands: int = 8            # D3: acoustically-motivated
    attn_heads: int = 4         # D7: matching n_heads
    attn_window: int = 4        # D7: local, not full K×K
    ffn_expansion: int = 4      # Standard SwiGLU expansion
    use_grad_checkpoint: bool = False # Disabled to prevent TileLang unpack bug in backward pass
    
    # ── Dataset ──
    data_dir: str = './VoiceBank_DEMAND_16k'
    sr: int = 16000
    segment_length: float = 4.0  # seconds → 64000 samples
    
    # ── Training recipe ──
    batch_size: int = 8
    epochs: int = 200           # SOTA systems train 200+ epochs for convergence
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95          # WHY: SSM params have volatile gradients
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip: float = 1.0       # Critical for complex angle stability
    warmup_steps: int = 1000
    eta_min: float = 3e-6        # 100× final ratio
    
    # ── Precision ──
    precision: str = 'bf16'      # MANDATORY for Mamba3 complex angle accumulation
    
    # ── Loss weights ──
    lambda_mag: float = 15.0
    lambda_complex: float = 10.0
    lambda_sisnr: float = 0.1   # Reduced from 0.5: raw SI-SNR (-17dB) was dominating gradients
    lambda_tc: float = 5.0
    
    # ── PCS ──
    use_pcs: bool = True
    pcs_gamma: float = 0.3       # Conservative exponent from SEMamba
    
    # ── Data augmentation ──
    use_remix: bool = True       # +0.03 PESQ
    use_bandmask: bool = True    # SSM regularization
    use_gain_aug: bool = True
    bandmask_ratio: float = 0.15  # Fraction of freq bins to mask (conservative)
    gain_range_db: float = 6.0
    ema_decay: float = 0.999      # EMA for evaluation stability (+0.03-0.08 PESQ)
    
    # ── Checkpointing ──
    validate_every: int = 5      # epochs
    save_every: int = 5          # epochs
    out_dir: str = './checkpoints/bsmamba3_run1'
    
    # ── Reproducibility ──
    seed: int = 42
    num_workers: int = 4
    
    # ── Safety valves ──
    max_step_time: float = 0.35  # seconds — reduce n_blocks if exceeded
    gradient_accumulation_steps: int = 1  # increase to 2 if OOM (effective batch stays 8)
    
    def validate(self):
        """Validate configuration constraints."""
        n_heads = self.d_model // self.headdim
        assert self.d_model % self.headdim == 0, \
            f"d_model={self.d_model} must be divisible by headdim={self.headdim}"
        assert self.mimo_rank <= n_heads, \
            f"mimo_rank={self.mimo_rank} must be ≤ n_heads={n_heads}"
        assert self.chunk_size == 64 // self.mimo_rank, \
            f"chunk_size must be 64/mimo_rank={64 // self.mimo_rank} for bf16"
        assert self.n_bands % self.attn_window == 0, \
            f"n_bands={self.n_bands} must be divisible by attn_window={self.attn_window}"
        assert self.precision in ('bf16', 'fp32'), \
            f"precision must be bf16 or fp32, got {self.precision}"
        print("[Config] All constraints validated ✓")
        return True


@dataclass
class AblationConfig(BSMamba3Config):
    """
    Ablation study configurations.
    N=4 blocks, batch=8, epoch 40 for all ablations.
    Each trainable within 4h on RTX 4050.
    """
    n_blocks: int = 4  # Faster for ablation
    epochs: int = 40   # Enough to see separation
    ablation_name: str = 'A1_full'
    
    @classmethod
    def A1_full(cls):
        """Full BSMamba3-SE (baseline for ablation)."""
        return cls(ablation_name='A1_full')
    
    @classmethod
    def A2_mamba2(cls):
        """Replace Mamba3 → Mamba2 (tests trapezoidal discretization)."""
        return cls(ablation_name='A2_mamba2')
    
    @classmethod
    def A3_siso(cls):
        """SISO mode (mimo_rank=1, tests MIMO contribution)."""
        cfg = cls(ablation_name='A3_siso')
        cfg.mimo_rank = 1
        cfg.chunk_size = 64  # 64/1 for SISO
        return cfg
    
    @classmethod
    def A5_no_attn(cls):
        """Remove InterBandAttention (tests generalization)."""
        return cls(ablation_name='A5_no_attn')
    
    @classmethod
    def A6_fullband(cls):
        """Full-band processing (no band-split, tests band-split value)."""
        cfg = cls(ablation_name='A6_fullband')
        cfg.n_bands = 1
        return cfg
    
    @classmethod
    def A7_no_pcs(cls):
        """Remove PCS (tests PCS contribution)."""
        cfg = cls(ablation_name='A7_no_pcs')
        cfg.use_pcs = False
        return cfg
    
    @classmethod
    def A8_no_tc(cls):
        """Remove temporal coherence loss (tests L_tc)."""
        cfg = cls(ablation_name='A8_no_tc')
        cfg.lambda_tc = 0.0
        return cfg
    
    @classmethod
    def A9_4blocks(cls):
        """N=4 blocks (same as SEMamba depth, tests depth scaling)."""
        return cls(n_blocks=4, ablation_name='A9_4blocks')
