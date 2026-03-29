"""
VoiceBank-DEMAND Dataset for BSMamba3-SE Training
══════════════════════════════════════════════════
Standard Valentini et al. 2016 split:
  Train: 28 speakers × ~414 utterances = 11,572 utterances, 16kHz
  Test:  2 speakers, 824 utterances
  Noise: 40 DEMAND conditions (train), 5 DEMAND conditions (test)
  SNR:   {0,5,10,15} dB train, {2.5,7.5,12.5,17.5,22.5} dB test

Data Augmentation (CPU-side, zero VRAM cost):
  1. Random gain ±6dB on clean signal (prevents gain normalization confound)
  2. Dynamic mixing / Remix (Maciejewski 2020): shuffle noise within batch (+0.03 PESQ)
  3. BandMask: randomly zero 20% of TARGET bands (SSM regularization, Ku et al. 2023)
  4. Random pitch shift ±1 semitone (prob 0.2, prevents F0-specific overfitting)
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import soundfile as sf


class VoiceBankDEMANDDataset(Dataset):
    """
    VoiceBank-DEMAND dataset for speech enhancement.
    
    Loads paired (noisy, clean) utterances from the standard directory structure:
      clean_trainset_28spk_wav/ or clean_testset_wav/
      noisy_trainset_28spk_wav/ or noisy_testset_wav/
    
    SCP files list utterance IDs (e.g., "p226_001") without extension.
    """

    def __init__(
        self,
        data_dir: str,
        scp_file: str,
        segment_length: float = 4.0,
        sr: int = 16000,
        is_train: bool = True,
        # Augmentation flags
        use_remix: bool = True,
        use_bandmask: bool = True,
        use_gain_aug: bool = True,
        use_pitch_shift: bool = False,
        bandmask_ratio: float = 0.2,
        gain_range_db: float = 6.0,
        pitch_shift_semitones: float = 1.0,
        pitch_shift_prob: float = 0.2,
    ):
        """
        Args:
            data_dir: root directory containing clean/noisy subdirs and scp files
            scp_file: path to .scp file with utterance IDs
            segment_length: segment length in seconds for training (4.0s → 64000 samples)
            sr: sample rate (16000 Hz)
            is_train: True for training (random crops + augmentation), False for validation
            use_remix: enable dynamic mixing augmentation
            use_bandmask: enable band masking augmentation on targets
            use_gain_aug: enable random gain augmentation
            use_pitch_shift: enable random pitch shifting
            bandmask_ratio: fraction of spectral bands to mask (0.2 = 20%)
            gain_range_db: maximum gain in dB (±6dB)
            pitch_shift_semitones: max pitch shift in semitones (±1)
            pitch_shift_prob: probability of applying pitch shift (0.2)
        """
        self.data_dir = data_dir
        self.sr = sr
        self.segment_samples = int(segment_length * sr)
        self.is_train = is_train
        
        # Augmentation config
        self.use_remix = use_remix and is_train
        self.use_bandmask = use_bandmask and is_train
        self.use_gain_aug = use_gain_aug and is_train
        self.use_pitch_shift = use_pitch_shift and is_train
        self.bandmask_ratio = bandmask_ratio
        self.gain_range_db = gain_range_db
        self.pitch_shift_semitones = pitch_shift_semitones
        self.pitch_shift_prob = pitch_shift_prob
        
        # Load utterance IDs from SCP file
        with open(scp_file, 'r') as f:
            self.utt_ids = [line.strip() for line in f if line.strip()]
        
        # Determine directory names
        if is_train:
            self.clean_dir = os.path.join(data_dir, 'clean_trainset_28spk_wav')
            self.noisy_dir = os.path.join(data_dir, 'noisy_trainset_28spk_wav')
        else:
            self.clean_dir = os.path.join(data_dir, 'clean_testset_wav')
            self.noisy_dir = os.path.join(data_dir, 'noisy_testset_wav')
        
        # Verify directories exist
        assert os.path.isdir(self.clean_dir), f"Clean dir not found: {self.clean_dir}"
        assert os.path.isdir(self.noisy_dir), f"Noisy dir not found: {self.noisy_dir}"
        
        print(f"[Dataset] Loaded {len(self.utt_ids)} utterances from {scp_file}")
        print(f"  Clean dir: {self.clean_dir}")
        print(f"  Noisy dir: {self.noisy_dir}")
        print(f"  Train mode: {is_train}, Segment: {segment_length}s ({self.segment_samples} samples)")

    def __len__(self) -> int:
        return len(self.utt_ids)

    def _load_wav(self, filepath: str) -> np.ndarray:
        """Load audio file and return as float32 numpy array."""
        audio, sr = sf.read(filepath, dtype='float32')
        assert sr == self.sr, f"Expected {self.sr}Hz, got {sr}Hz for {filepath}"
        return audio

    def _random_crop(self, clean: np.ndarray, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop to segment_samples length. Pad if shorter."""
        L = len(clean)
        target_L = self.segment_samples
        
        if L >= target_L:
            # Random start point
            start = random.randint(0, L - target_L)
            clean = clean[start:start + target_L]
            noisy = noisy[start:start + target_L]
        else:
            # Pad with zeros
            pad_len = target_L - L
            clean = np.pad(clean, (0, pad_len), mode='constant')
            noisy = np.pad(noisy, (0, pad_len), mode='constant')
        
        return clean, noisy

    def _apply_gain(self, clean: np.ndarray) -> np.ndarray:
        """
        Random gain ±6dB on clean signal.
        
        Prevents gain normalization from becoming a confound.
        Applied before computing noise residual for remix.
        """
        gain_db = random.uniform(-self.gain_range_db, self.gain_range_db)
        gain_linear = 10 ** (gain_db / 20)
        return clean * gain_linear

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            noisy: [segment_samples] noisy waveform tensor
            clean: [segment_samples] clean waveform tensor
            utt_id: str utterance ID
        """
        utt_id = self.utt_ids[idx]
        
        # Load audio files
        clean_path = os.path.join(self.clean_dir, f'{utt_id}.wav')
        noisy_path = os.path.join(self.noisy_dir, f'{utt_id}.wav')
        
        clean = self._load_wav(clean_path)
        noisy = self._load_wav(noisy_path)
        
        # Ensure same length
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        if self.is_train:
            # Random crop to segment length
            clean, noisy = self._random_crop(clean, noisy)
            
            # Gain augmentation (applied to both for consistency)
            if self.use_gain_aug:
                gain_db = random.uniform(-self.gain_range_db, self.gain_range_db)
                gain_linear = 10 ** (gain_db / 20)
                clean = clean * gain_linear
                noisy = noisy * gain_linear
        
        # Convert to tensors
        clean_tensor = torch.from_numpy(clean).float()
        noisy_tensor = torch.from_numpy(noisy).float()
        
        return noisy_tensor, clean_tensor, utt_id


class RemixCollator:
    """
    Dynamic mixing (Remix) collator for DataLoader.
    
    Shuffles noise within the batch to create novel SNR/noise pairings.
    Noise = noisy - clean (residual extraction).
    Adds shuffled noise back to clean at original levels.
    
    +0.03 PESQ in SEMamba ablations (Maciejewski et al. 2020).
    """
    
    def __init__(self, use_remix: bool = True, remix_prob: float = 0.5):
        self.use_remix = use_remix
        self.remix_prob = remix_prob
    
    def __call__(self, batch):
        noisy_list, clean_list, utt_ids = zip(*batch)
        
        noisy = torch.stack(noisy_list)   # [B, L]
        clean = torch.stack(clean_list)   # [B, L]
        
        if self.use_remix and random.random() < self.remix_prob:
            B = noisy.shape[0]
            # Extract noise residual
            noise = noisy - clean  # [B, L]
            
            # Shuffle noise across batch
            perm = torch.randperm(B)
            shuffled_noise = noise[perm]  # [B, L]
            
            # Remix: original clean + shuffled noise
            noisy = clean + shuffled_noise  # [B, L]
        
        return noisy, clean, list(utt_ids)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    segment_length: float = 4.0,
    num_workers: int = 4,
    use_remix: bool = True,
    use_bandmask: bool = True,
    use_gain_aug: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        data_dir: root of VoiceBank-DEMAND dataset
        batch_size: 8
        segment_length: 4.0 seconds (64000 samples at 16kHz)
        num_workers: CPU workers for data loading
        use_remix: enable dynamic mixing
        use_bandmask: enable band masking (applied in loss, not here)
        use_gain_aug: enable random gain
        pin_memory: pin memory for GPU transfer
    
    Returns:
        train_loader, val_loader
    """
    train_scp = os.path.join(data_dir, 'train.scp')
    test_scp = os.path.join(data_dir, 'test.scp')
    
    train_dataset = VoiceBankDEMANDDataset(
        data_dir=data_dir,
        scp_file=train_scp,
        segment_length=segment_length,
        is_train=True,
        use_remix=use_remix,
        use_bandmask=use_bandmask,
        use_gain_aug=use_gain_aug,
    )
    
    val_dataset = VoiceBankDEMANDDataset(
        data_dir=data_dir,
        scp_file=test_scp,
        segment_length=0,  # Not used (full utterance)
        is_train=False,
    )
    
    # Remix collator for training
    remix_collator = RemixCollator(use_remix=use_remix)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=remix_collator,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Validation: full utterances, no augmentation, batch=1
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader
