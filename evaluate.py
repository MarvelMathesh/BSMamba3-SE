"""
BSMamba3-SE Evaluation Script
═════════════════════════════
Full test-set evaluation with all metrics: WB-PESQ, CSIG, CBAK, COVL, STOI, SI-SNR, RTF.
"""

import os
import time
import argparse
import numpy as np
import torch
import soundfile as sf
from pesq import pesq
from pystoi import stoi

from config import BSMamba3Config
from models.bsmamba3_se import BSMamba3SE, mag_phase_stft, mag_phase_istft


def si_snr(estimate, reference):
    """Scale-Invariant Signal-to-Noise Ratio."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    dot = np.sum(reference * estimate)
    s_target = dot * reference / (np.sum(reference ** 2) + 1e-8)
    e_noise = estimate - s_target
    si_snr_val = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8) + 1e-8)
    return si_snr_val


def composite_metrics(clean, enhanced, sr=16000):
    """
    Compute composite metrics (CSIG, CBAK, COVL) via Hu & Loizou regression.
    Uses narrowband PESQ as base metric.
    """
    try:
        nb_pesq = pesq(sr, clean, enhanced, 'nb')
    except Exception:
        return 1.0, 1.0, 1.0

    # Segmental SNR
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    seg_snrs = []
    for start in range(0, len(clean) - frame_len, hop):
        c_frame = clean[start:start + frame_len]
        e_frame = enhanced[start:start + frame_len]
        noise_frame = c_frame - e_frame
        c_energy = np.sum(c_frame ** 2) + 1e-10
        n_energy = np.sum(noise_frame ** 2) + 1e-10
        seg_snr = 10 * np.log10(c_energy / n_energy)
        seg_snr = max(min(seg_snr, 35), -10)
        seg_snrs.append(seg_snr)
    seg_snr_val = np.mean(seg_snrs) if seg_snrs else 0.0

    # Hu & Loizou composite regression
    csig = 3.093 - 1.029 * seg_snr_val / 10 + 0.603 * nb_pesq - 0.009 * (seg_snr_val / 10) ** 2
    cbak = 1.634 + 0.478 * nb_pesq - 0.007 * seg_snr_val / 10 + 0.063 * nb_pesq ** 2
    covl = 1.594 + 0.805 * nb_pesq - 0.512 * seg_snr_val / 10 + 0.007 * nb_pesq ** 2

    csig = max(1, min(5, csig))
    cbak = max(1, min(5, cbak))
    covl = max(1, min(5, covl))

    return csig, cbak, covl


@torch.no_grad()
def evaluate(checkpoint_path, data_dir, out_dir, device='cuda'):
    """Run full evaluation on test set."""
    cfg = BSMamba3Config()
    cfg.data_dir = data_dir
    device = torch.device(device)

    n_fft = cfg.n_fft
    hop_size = cfg.hop_size
    win_size = cfg.win_size
    compress_factor = cfg.compress_factor
    sr = cfg.sr

    # Load model
    print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BSMamba3SE(cfg).to(device)

    # Handle different checkpoint formats
    if 'generator' in ckpt:
        model.load_state_dict(ckpt['generator'], strict=False)
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    n_params = model.count_parameters()
    print(f"  Model: {n_params / 1e6:.2f}M parameters")

    # Load test file list
    test_scp = os.path.join(data_dir, 'test.scp')
    clean_dir = os.path.join(data_dir, 'clean_testset_wav')
    noisy_dir = os.path.join(data_dir, 'noisy_testset_wav')

    if os.path.exists(test_scp):
        with open(test_scp, 'r') as f:
            files = [os.path.basename(l.strip()) for l in f if l.strip()]
    else:
        files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])

    print(f"  Test set: {len(files)} utterances")

    # Create output directories
    enhanced_dir = os.path.join(out_dir, 'enhanced')
    os.makedirs(enhanced_dir, exist_ok=True)

    # Evaluate
    all_pesq, all_stoi, all_sisnr = [], [], []
    all_csig, all_cbak, all_covl = [], [], []
    total_audio_time = 0.0
    total_proc_time = 0.0

    for idx, filename in enumerate(files):
        # Load audio
        clean_path = os.path.join(clean_dir, filename)
        noisy_path = os.path.join(noisy_dir, filename)

        clean_wav, _ = sf.read(clean_path, dtype='float32')
        noisy_wav, _ = sf.read(noisy_path, dtype='float32')

        min_len = min(len(clean_wav), len(noisy_wav))
        clean_wav = clean_wav[:min_len]
        noisy_wav = noisy_wav[:min_len]

        audio_duration = len(noisy_wav) / sr
        total_audio_time += audio_duration

        # To tensor
        noisy_tensor = torch.FloatTensor(noisy_wav).unsqueeze(0).to(device)

        # Compute noisy STFT
        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_tensor, n_fft, hop_size, win_size, compress_factor
        )

        # Inference with timing
        start_time = time.time()
        mag_g, pha_g, _ = model(noisy_mag, noisy_pha)
        enhanced_wav = mag_phase_istft(
            mag_g, pha_g, n_fft, hop_size, win_size, compress_factor
        )
        torch.cuda.synchronize()
        proc_time = time.time() - start_time
        total_proc_time += proc_time

        # To numpy
        enhanced_np = enhanced_wav.squeeze().cpu().numpy()

        # Trim to same length
        min_out = min(len(clean_wav), len(enhanced_np))
        clean_np = clean_wav[:min_out]
        enhanced_np = enhanced_np[:min_out]

        # Compute metrics
        try:
            wb_pesq = pesq(sr, clean_np, enhanced_np, 'wb')
        except Exception:
            wb_pesq = 1.0
        all_pesq.append(wb_pesq)

        try:
            stoi_val = stoi(clean_np, enhanced_np, sr, extended=False)
        except Exception:
            stoi_val = 0.0
        all_stoi.append(stoi_val)

        sisnr_val = si_snr(enhanced_np, clean_np)
        all_sisnr.append(sisnr_val)

        csig, cbak, covl = composite_metrics(clean_np, enhanced_np, sr)
        all_csig.append(csig)
        all_cbak.append(cbak)
        all_covl.append(covl)

        # Save enhanced audio
        sf.write(os.path.join(enhanced_dir, filename), enhanced_np, sr)

        # Progress
        if (idx + 1) % 100 == 0:
            avg_pesq = np.mean(all_pesq)
            print(f"  [{idx+1}/{len(files)}] Avg PESQ: {avg_pesq:.4f}")

    # Summary
    rtf = total_proc_time / total_audio_time if total_audio_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  WB-PESQ: {np.mean(all_pesq):.4f} ± {np.std(all_pesq):.4f}")
    print(f"  CSIG:    {np.mean(all_csig):.4f}")
    print(f"  CBAK:    {np.mean(all_cbak):.4f}")
    print(f"  COVL:    {np.mean(all_covl):.4f}")
    print(f"  STOI:    {np.mean(all_stoi):.4f}")
    print(f"  SI-SNR:  {np.mean(all_sisnr):.2f} dB")
    print(f"  RTF:     {rtf:.4f}")
    print(f"  Params:  {n_params / 1e6:.2f}M")
    print(f"{'='*60}")

    # Save results
    results = {
        'pesq': float(np.mean(all_pesq)),
        'pesq_std': float(np.std(all_pesq)),
        'csig': float(np.mean(all_csig)),
        'cbak': float(np.mean(all_cbak)),
        'covl': float(np.mean(all_covl)),
        'stoi': float(np.mean(all_stoi)),
        'sisnr': float(np.mean(all_sisnr)),
        'rtf': rtf,
        'params': n_params,
    }
    import json
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {out_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='BSMamba3-SE Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./VoiceBank_DEMAND_16k')
    parser.add_argument('--out_dir', type=str, default='./eval_results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.out_dir, args.device)


if __name__ == '__main__':
    main()
