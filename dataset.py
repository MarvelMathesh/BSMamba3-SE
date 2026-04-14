"""
BSMamba3-SE Dataset
═══════════════════
VoiceBank-DEMAND dataloader returning pre-computed compressed STFT.

Returns per sample:
  clean_audio:   [1, L] waveform (for time loss and PESQ)
  clean_mag:     [F, T] compressed magnitude |STFT|^0.3
  clean_pha:     [F, T] phase
  clean_com:     [F, T, 2] compressed complex
  noisy_mag:     [F, T] compressed magnitude
  noisy_pha:     [F, T] phase

Normalization: both clean and noisy are scaled by noisy RMS factor,
ensuring consistent input levels for the discriminator.
"""

import os
import random
import torch
import torch.utils.data
import numpy as np
import soundfile as sf
from models.bsmamba3_se import mag_phase_stft
from pcs import apply_pcs_to_waveform


class VoiceBankDemandDataset(torch.utils.data.Dataset):
    """
    VoiceBank-DEMAND dataset with compressed STFT targets.
    """

    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        scp_file: str = None,
        file_list: list = None,
        sr: int = 16000,
        n_fft: int = 400,
        hop_size: int = 100,
        win_size: int = 400,
        compress_factor: float = 0.3,
        segment_size: int = 32000,
        split: bool = True,
        use_pcs: bool = False,
        pcs_gamma: float = 0.3,
    ):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.segment_size = segment_size
        self.split = split
        self.use_pcs = use_pcs
        self.pcs_gamma = pcs_gamma

        # Build file list
        if file_list is not None:
            self.file_list = file_list
        elif scp_file is not None:
            self.file_list = self._load_scp(scp_file)
        else:
            # Auto-discover from noisy directory
            self.file_list = sorted([
                f for f in os.listdir(noisy_dir)
                if f.endswith('.wav')
            ])

        print(f"[Dataset] Loaded {len(self.file_list)} utterances")
        print(f"  Clean dir: {clean_dir}")
        print(f"  Noisy dir: {noisy_dir}")
        print(f"  STFT: n_fft={n_fft}, hop={hop_size}, compress={compress_factor}")
        print(f"  Split: {split}, segment={segment_size} ({segment_size/sr:.1f}s)")
        print(f"  PCS: {use_pcs}")

    def _load_scp(self, scp_file: str) -> list:
        """Load file list from .scp file."""
        files = []
        with open(scp_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Handle both full paths and basenames
                    files.append(os.path.basename(line))
        return files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # Load audio files
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        clean_audio, _ = sf.read(clean_path, dtype='float32')
        noisy_audio, _ = sf.read(noisy_path, dtype='float32')

        # Apply PCS to clean target if enabled (training-time target modification)
        if self.use_pcs:
            clean_audio = apply_pcs_to_waveform(
                clean_audio, n_fft=self.n_fft, hop_size=self.hop_size,
                win_size=self.win_size, gamma=self.pcs_gamma
            )

        clean_audio = torch.FloatTensor(clean_audio)
        noisy_audio = torch.FloatTensor(noisy_audio)

        # Ensure same length
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        # RMS normalization by noisy audio (consistent input level for discriminator)
        norm_factor = torch.sqrt(len(noisy_audio) / (torch.sum(noisy_audio ** 2.0) + 1e-8))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)  # [1, L]
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)  # [1, L]

        # Random segment cropping for training
        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_start = clean_audio.size(1) - self.segment_size
                start = random.randint(0, max_start)
                clean_audio = clean_audio[:, start:start + self.segment_size]
                noisy_audio = noisy_audio[:, start:start + self.segment_size]
            else:
                # Pad shorter utterances
                pad_len = self.segment_size - clean_audio.size(1)
                clean_audio = F.pad(clean_audio, (0, pad_len))
                noisy_audio = F.pad(noisy_audio, (0, pad_len))

        # Compute compressed STFT representations
        clean_mag, clean_pha, clean_com = mag_phase_stft(
            clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )
        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor
        )

        return (
            clean_audio.squeeze(0),   # [L] waveform for time loss
            clean_mag.squeeze(0),     # [F, T]
            clean_pha.squeeze(0),     # [F, T]
            clean_com.squeeze(0),     # [F, T, 2]
            noisy_mag.squeeze(0),     # [F, T]
            noisy_pha.squeeze(0),     # [F, T]
        )


def create_train_dataset(cfg) -> VoiceBankDemandDataset:
    """Create training dataset from config."""
    return VoiceBankDemandDataset(
        clean_dir=os.path.join(cfg.data_dir, 'clean_trainset_wav'),
        noisy_dir=os.path.join(cfg.data_dir, 'noisy_trainset_wav'),
        scp_file=os.path.join(cfg.data_dir, 'train.scp'),
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_size=cfg.hop_size,
        win_size=cfg.win_size,
        compress_factor=cfg.compress_factor,
        segment_size=cfg.segment_size,
        split=True,
        use_pcs=cfg.use_pcs,
        pcs_gamma=cfg.pcs_gamma,
    )


def create_valid_dataset(cfg) -> VoiceBankDemandDataset:
    """Create validation dataset from config."""
    return VoiceBankDemandDataset(
        clean_dir=os.path.join(cfg.data_dir, 'clean_testset_wav'),
        noisy_dir=os.path.join(cfg.data_dir, 'noisy_testset_wav'),
        scp_file=os.path.join(cfg.data_dir, 'test.scp'),
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_size=cfg.hop_size,
        win_size=cfg.win_size,
        compress_factor=cfg.compress_factor,
        segment_size=cfg.segment_size,
        split=False,  # No cropping for validation
        use_pcs=False,  # No PCS for validation
    )


# Make F.pad available
import torch.nn.functional as F
