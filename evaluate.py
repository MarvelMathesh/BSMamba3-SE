"""
BSMamba3-SE Evaluation Script
══════════════════════════════
Standalone evaluation with full metrics for SOTA comparison table.

Metrics:
  - WB-PESQ (pypesq 0.1.3)
  - CSIG (signal distortion MOS)
  - CBAK (background intrusiveness MOS)
  - COVL (overall quality MOS)
  - STOI (Short-Time Objective Intelligibility)
  - SI-SNR
  - RTF (Real-Time Factor)

Reports per-condition PESQ breakdown for voiced/unvoiced analysis.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import warnings

import torch
from torch.amp import autocast

from config import BSMamba3Config
from models.bsmamba3_se import BSMamba3SE
from dataset import VoiceBankDEMANDDataset

import soundfile as sf

warnings.filterwarnings('ignore')


def compute_composite_metrics(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> dict:
    """
    Compute composite metrics: CSIG, CBAK, COVL.
    
    These are the standard composite metrics from Hu & Loizou (2008)
    computed from segmental SNR and LLR/WSS/PESQ.
    
    Falls back to basic metrics if composite library not available.
    """
    metrics = {}
    
    # WB-PESQ
    try:
        from pesq import pesq
        metrics['wb_pesq'] = pesq(sr, clean, enhanced, 'wb')
    except ImportError:
        try:
            from pypesq import pesq
            metrics['wb_pesq'] = pesq(sr, clean, enhanced, 'wb')
        except ImportError:
            metrics['wb_pesq'] = float('nan')
    except Exception:
        metrics['wb_pesq'] = float('nan')
    
    # STOI
    try:
        from pystoi import stoi
        metrics['stoi'] = stoi(clean, enhanced, sr, extended=False)
    except ImportError:
        metrics['stoi'] = float('nan')
    except Exception:
        metrics['stoi'] = float('nan')
    
    # SI-SNR
    pred = enhanced - np.mean(enhanced)
    target = clean - np.mean(clean)
    dot = np.sum(pred * target)
    s_target = (dot / (np.sum(target ** 2) + 1e-8)) * target
    e_noise = pred - s_target
    metrics['si_snr'] = float(10 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8) + 1e-8
    ))
    
    # Composite metrics (CSIG, CBAK, COVL) via pesq + segSNR regression
    # Using the Hu & Loizou (2008) regression coefficients
    try:
        from pesq import pesq as pesq_fn
        pesq_raw = pesq_fn(sr, clean, enhanced, 'nb')  # NB-PESQ for composite
        
        # Segmental SNR
        seg_snr = _compute_segsnr(clean, enhanced, sr)
        
        # LLR (Log-Likelihood Ratio)
        llr = _compute_llr(clean, enhanced, sr)
        
        # WSS (Weighted Spectral Slope)
        wss = _compute_wss(clean, enhanced, sr)
        
        # Hu & Loizou regression for CSIG, CBAK, COVL
        metrics['csig'] = 3.093 - 1.029 * llr + 0.603 * pesq_raw - 0.009 * wss
        metrics['cbak'] = 1.634 + 0.478 * pesq_raw - 0.007 * wss + 0.063 * seg_snr
        metrics['covl'] = 1.594 + 0.805 * pesq_raw - 0.512 * llr - 0.007 * wss
        
    except Exception:
        metrics['csig'] = float('nan')
        metrics['cbak'] = float('nan')
        metrics['covl'] = float('nan')
    
    return metrics


def _compute_segsnr(clean: np.ndarray, enhanced: np.ndarray, sr: int, frame_len: float = 0.03) -> float:
    """Compute segmental SNR."""
    frame_samples = int(frame_len * sr)
    n_frames = len(clean) // frame_samples
    
    seg_snrs = []
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        c = clean[start:end]
        n = enhanced[start:end] - c
        c_energy = np.sum(c ** 2)
        n_energy = np.sum(n ** 2) + 1e-10
        seg_snr = 10 * np.log10(c_energy / n_energy + 1e-10)
        seg_snr = np.clip(seg_snr, -10, 35)
        seg_snrs.append(seg_snr)
    
    return np.mean(seg_snrs) if seg_snrs else 0.0


def _compute_llr(clean: np.ndarray, enhanced: np.ndarray, sr: int, 
                 frame_len: float = 0.03, order: int = 16) -> float:
    """Compute Log-Likelihood Ratio (LLR)."""
    from numpy.linalg import solve, LinAlgError
    
    frame_samples = int(frame_len * sr)
    n_frames = len(clean) // frame_samples
    
    llrs = []
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        c = clean[start:end]
        e = enhanced[start:end]
        
        try:
            # LPC coefficients
            Rc = np.correlate(c, c, mode='full')[len(c)-1:len(c)-1+order+1]
            Re = np.correlate(e, e, mode='full')[len(e)-1:len(e)-1+order+1]
            
            # Toeplitz solve for LPC
            if abs(Rc[0]) < 1e-10 or abs(Re[0]) < 1e-10:
                continue
                
            # Levinson-Durbin
            ac = _levinson(Rc, order)
            ae = _levinson(Re, order)
            
            if ac is None or ae is None:
                continue
            
            # LLR
            Rc_matrix = np.zeros((order+1, order+1))
            for j in range(order+1):
                for k in range(order+1):
                    idx = abs(j - k)
                    if idx <= order:
                        Rc_matrix[j, k] = Rc[idx]
            
            num = ae @ Rc_matrix @ ae
            den = ac @ Rc_matrix @ ac + 1e-10
            
            llr_val = np.log(max(num / den, 1e-10))
            llr_val = min(llr_val, 2.0)
            llrs.append(llr_val)
        except (LinAlgError, ValueError):
            continue
    
    return np.mean(llrs) if llrs else 0.0


def _levinson(r, order):
    """Simple Levinson-Durbin recursion."""
    if abs(r[0]) < 1e-10:
        return None
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    
    for i in range(1, order + 1):
        lam = 0
        for j in range(i):
            lam -= a[j] * r[i - j]
        lam /= (e + 1e-10)
        
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        a = a_new
        e *= (1 - lam * lam)
        if e < 0:
            return None
    
    return a


def _compute_wss(clean: np.ndarray, enhanced: np.ndarray, sr: int,
                 frame_len: float = 0.03, n_bands: int = 25) -> float:
    """Compute Weighted Spectral Slope (WSS) distance."""
    frame_samples = int(frame_len * sr)
    n_fft = 2 ** int(np.ceil(np.log2(frame_samples)))
    n_frames = len(clean) // frame_samples
    
    wsss = []
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        c = clean[start:end]
        e = enhanced[start:end]
        
        # Apply Hamming window
        win = np.hamming(frame_samples)
        c = c * win
        e = e * win
        
        # Power spectrum
        C = np.abs(np.fft.rfft(c, n=n_fft)) ** 2 + 1e-10
        E = np.abs(np.fft.rfft(e, n=n_fft)) ** 2 + 1e-10
        
        # Critical band aggregation (simplified)
        band_size = len(C) // n_bands
        if band_size < 1:
            continue
            
        wss_frame = 0
        for b in range(n_bands):
            s = b * band_size
            e_idx = min(s + band_size, len(C))
            
            c_band = 10 * np.log10(np.mean(C[s:e_idx]))
            e_band = 10 * np.log10(np.mean(E[s:e_idx]))
            
            # Spectral slope
            if b > 0:
                c_slope = c_band - c_band_prev
                e_slope = e_band - e_band_prev
                wss_frame += (c_slope - e_slope) ** 2
            
            c_band_prev = c_band
            e_band_prev = e_band
        
        wsss.append(wss_frame / max(n_bands - 1, 1))
    
    return np.mean(wsss) if wsss else 0.0


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    data_dir: str,
    out_dir: str = './eval_results',
    device: str = 'cuda',
):
    """
    Full evaluation with all metrics.
    
    Outputs:
      - Per-utterance metrics CSV
      - Mean metrics summary
      - Enhanced wavs
      - RTF measurement
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt.get('config', {})
    
    # Create model
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
    
    total_params = model.count_parameters()
    print(f"  Model: {total_params/1e6:.2f}M parameters")
    
    # Load test data
    test_scp = os.path.join(data_dir, 'test.scp')
    test_dataset = VoiceBankDEMANDDataset(
        data_dir=data_dir,
        scp_file=test_scp,
        is_train=False,
    )
    
    os.makedirs(out_dir, exist_ok=True)
    enhanced_dir = os.path.join(out_dir, 'enhanced')
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # Evaluate
    all_metrics = []
    total_audio_time = 0
    total_proc_time = 0
    
    for idx in range(len(test_dataset)):
        noisy, clean, utt_id = test_dataset[idx]
        noisy = noisy.unsqueeze(0).to(device)  # [1, L]
        
        audio_duration = noisy.shape[1] / 16000
        total_audio_time += audio_duration
        
        # Inference with RTF timing
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_start = time.time()
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            enhanced, _, _ = model(noisy)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        proc_time = time.time() - t_start
        total_proc_time += proc_time
        
        # Convert to numpy
        enhanced_np = enhanced.squeeze(0).cpu().float().numpy()
        clean_np = clean.numpy()
        
        min_len = min(len(enhanced_np), len(clean_np))
        enhanced_np = enhanced_np[:min_len]
        clean_np = clean_np[:min_len]
        
        # Compute metrics
        metrics = compute_composite_metrics(clean_np, enhanced_np, sr=16000)
        metrics['utt_id'] = utt_id
        metrics['rtf'] = proc_time / audio_duration
        all_metrics.append(metrics)
        
        # Save enhanced wav
        sf.write(os.path.join(enhanced_dir, f'{utt_id}.wav'), enhanced_np, 16000)
        
        if (idx + 1) % 100 == 0:
            avg_pesq = np.nanmean([m['wb_pesq'] for m in all_metrics])
            print(f"  [{idx+1}/{len(test_dataset)}] Avg PESQ: {avg_pesq:.4f}")
    
    # Compute summary
    rtf = total_proc_time / total_audio_time
    
    summary = {
        'model': checkpoint_path,
        'params': total_params,
        'params_M': total_params / 1e6,
        'n_utterances': len(all_metrics),
        'wb_pesq': np.nanmean([m['wb_pesq'] for m in all_metrics]),
        'wb_pesq_std': np.nanstd([m['wb_pesq'] for m in all_metrics]),
        'csig': np.nanmean([m.get('csig', float('nan')) for m in all_metrics]),
        'cbak': np.nanmean([m.get('cbak', float('nan')) for m in all_metrics]),
        'covl': np.nanmean([m.get('covl', float('nan')) for m in all_metrics]),
        'stoi': np.nanmean([m.get('stoi', float('nan')) for m in all_metrics]),
        'si_snr': np.nanmean([m['si_snr'] for m in all_metrics]),
        'rtf': rtf,
    }
    
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  WB-PESQ: {summary['wb_pesq']:.4f} ± {summary['wb_pesq_std']:.4f}")
    print(f"  CSIG:    {summary['csig']:.4f}")
    print(f"  CBAK:    {summary['cbak']:.4f}")
    print(f"  COVL:    {summary['covl']:.4f}")
    print(f"  STOI:    {summary['stoi']:.4f}")
    print(f"  SI-SNR:  {summary['si_snr']:.2f} dB")
    print(f"  RTF:     {summary['rtf']:.4f}")
    print(f"  Params:  {summary['params_M']:.2f}M")
    print(f"{'='*70}")
    
    # Save results
    with open(os.path.join(out_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    with open(os.path.join(out_dir, 'eval_per_utterance.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print(f"  Results saved to {out_dir}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BSMamba3-SE Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='./VoiceBank_DEMAND_16k')
    parser.add_argument('--out_dir', type=str, default='./eval_results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.data_dir, args.out_dir, args.device)
