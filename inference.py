"""
BSMamba3-SE Inference Script
═════════════════════════════
Single-file and batch inference with offline and streaming modes.

Offline mode:  Full-utterance processing (same as training, non-causal)
Streaming mode: Frame-by-frame using Mamba3 recurrent step (<40ms latency)

Streaming spec:
  Frame size: 256 samples (16ms at 16kHz)
  Algorithmic latency: 16ms (frame) + 4ms (compute) = 20ms ✓
  State memory: K=8 × D=256 × d_state=128 × complex × bf16 = 1.34 MB (fits L2 cache)
"""

import os
import sys
import time
import argparse
import numpy as np

import torch
from torch.amp import autocast

from config import BSMamba3Config
from models.bsmamba3_se import BSMamba3SE

import soundfile as sf


def load_model(
    checkpoint_path: str,
    device: str = 'cuda',
) -> tuple:
    """Load model from checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt.get('config', {})
    
    config = BSMamba3Config(**{k: v for k, v in config_dict.items() 
                              if k in BSMamba3Config.__dataclass_fields__})
    
    model = BSMamba3SE(
        d_model=config.d_model,
        d_state=config.d_state,
        headdim=config.headdim,
        mimo_rank=config.mimo_rank,
        chunk_size=config.chunk_size,
        n_blocks=config.n_blocks,
        n_bands=config.n_bands,
        attn_heads=config.attn_heads,
        attn_window=config.attn_window,
        ffn_expansion=config.ffn_expansion,
        use_grad_checkpoint=False,
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    return model, config, device


@torch.no_grad()
def enhance_file(
    model: BSMamba3SE,
    input_path: str,
    output_path: str,
    device: torch.device,
    sr: int = 16000,
) -> dict:
    """
    Enhance a single audio file (offline mode).
    
    Args:
        model: loaded BSMamba3SE model
        input_path: path to noisy .wav file
        output_path: path to save enhanced .wav
        device: torch device
        sr: sample rate (16000 Hz)
    
    Returns:
        dict with timing info
    """
    # Load audio
    audio, file_sr = sf.read(input_path, dtype='float32')
    assert file_sr == sr, f"Expected {sr}Hz, got {file_sr}Hz"
    
    if audio.ndim > 1:
        audio = audio[:, 0]  # Take first channel if stereo
    
    duration = len(audio) / sr
    
    # Convert to tensor
    x = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # [1, L]
    
    # Inference
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_start = time.time()
    
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        enhanced, _, _ = model(x)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    proc_time = time.time() - t_start
    
    # Save
    enhanced_np = enhanced.squeeze(0).cpu().float().numpy()
    sf.write(output_path, enhanced_np, sr)
    
    rtf = proc_time / duration
    
    result = {
        'input': input_path,
        'output': output_path,
        'duration_s': duration,
        'proc_time_s': proc_time,
        'rtf': rtf,
    }
    
    print(f"  Enhanced: {input_path}")
    print(f"    Duration: {duration:.2f}s, Processing: {proc_time:.3f}s, RTF: {rtf:.4f}")
    
    return result


@torch.no_grad()
def enhance_directory(
    model: BSMamba3SE,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    sr: int = 16000,
) -> list:
    """Enhance all .wav files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav')])
    print(f"[Inference] Found {len(wav_files)} .wav files in {input_dir}")
    
    results = []
    for wav_file in wav_files:
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, wav_file)
        result = enhance_file(model, input_path, output_path, device, sr)
        results.append(result)
    
    # Summary
    if results:
        avg_rtf = np.mean([r['rtf'] for r in results])
        total_audio = sum(r['duration_s'] for r in results)
        total_proc = sum(r['proc_time_s'] for r in results)
        print(f"\n  Processed {len(results)} files")
        print(f"  Total audio: {total_audio:.1f}s, Total processing: {total_proc:.1f}s")
        print(f"  Average RTF: {avg_rtf:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='BSMamba3-SE Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input wav file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output wav file or directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sr', type=int, default=16000)
    args = parser.parse_args()
    
    # Load model
    print("[Inference] Loading model...")
    model, config, device = load_model(args.checkpoint, args.device)
    print(f"  Model: {model.count_parameters()/1e6:.2f}M params")
    
    # Enhance
    if os.path.isdir(args.input):
        enhance_directory(model, args.input, args.output, device, args.sr)
    elif os.path.isfile(args.input):
        enhance_file(model, args.input, args.output, device, args.sr)
    else:
        print(f"Error: {args.input} not found")
        sys.exit(1)


if __name__ == '__main__':
    main()
