"""
BSMamba3-SE Inference
════════════════════
Single-file and batch speech enhancement inference.
"""

import os
import argparse
import torch
import numpy as np
import soundfile as sf

from config import BSMamba3Config
from models.bsmamba3_se import BSMamba3SE, mag_phase_stft, mag_phase_istft


class BSMamba3Enhancer:
    """Wrapper for easy speech enhancement inference."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        self.cfg = BSMamba3Config()

        # Load model
        self.model = BSMamba3SE(self.cfg).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'generator' in ckpt:
            self.model.load_state_dict(ckpt['generator'], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        print(f"[Inference] Model loaded: {self.model.count_parameters() / 1e6:.2f}M params")

    @torch.no_grad()
    def enhance(self, noisy_wav: np.ndarray) -> np.ndarray:
        """
        Enhance a single noisy waveform.
        
        Args:
            noisy_wav: numpy array, shape [L] or [1, L], 16kHz mono
            
        Returns:
            enhanced_wav: numpy array, shape [L], 16kHz mono
        """
        cfg = self.cfg

        # To tensor
        if noisy_wav.ndim == 1:
            noisy_wav = noisy_wav[np.newaxis, :]
        noisy_tensor = torch.FloatTensor(noisy_wav).to(self.device)

        # Compute STFT
        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_tensor, cfg.n_fft, cfg.hop_size, cfg.win_size, cfg.compress_factor
        )

        # Forward pass
        mag_g, pha_g, _ = self.model(noisy_mag, noisy_pha)

        # Reconstruct waveform
        enhanced = mag_phase_istft(
            mag_g, pha_g, cfg.n_fft, cfg.hop_size, cfg.win_size, cfg.compress_factor
        )

        return enhanced.squeeze().cpu().numpy()

    def enhance_file(self, input_path: str, output_path: str):
        """Enhance a single audio file."""
        noisy_wav, sr = sf.read(input_path, dtype='float32')
        assert sr == self.cfg.sr, f"Expected {self.cfg.sr}Hz, got {sr}Hz"

        enhanced = self.enhance(noisy_wav)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        sf.write(output_path, enhanced, sr)
        print(f"  {input_path} → {output_path}")


def main():
    parser = argparse.ArgumentParser(description='BSMamba3-SE Inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True, help='Input wav file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output wav file or directory')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    enhancer = BSMamba3Enhancer(args.checkpoint, args.device)

    if os.path.isfile(args.input):
        enhancer.enhance_file(args.input, args.output)
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        wav_files = sorted([f for f in os.listdir(args.input) if f.endswith('.wav')])
        print(f"[Inference] Processing {len(wav_files)} files...")
        for wav_file in wav_files:
            in_path = os.path.join(args.input, wav_file)
            out_path = os.path.join(args.output, wav_file)
            enhancer.enhance_file(in_path, out_path)
        print(f"[Inference] Done! Enhanced files saved to {args.output}")
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")


if __name__ == '__main__':
    main()
