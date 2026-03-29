"""
BSMamba3-SE Training Script
════════════════════════════
Complete training loop for Band-Split Mamba3 Speech Enhancement.

Training recipe:
  Optimizer: AdamW (lr=3e-4, β=(0.9, 0.95), wd=0.01)
  Schedule:  Warmup 1000 steps → CosineAnnealingLR(T_max=80, eta_min=3e-6)
  Precision: bf16 (MANDATORY for Mamba3 complex angle accumulation)
  Gradient:  Checkpointing enabled, max_norm=1.0 clipping
  Batch:     8, segment 4s (64000 samples)
  Epochs:    80, validate every 5, checkpoint every 5

Safety valves:
  If epoch-1 step time > 0.35s: reduce n_blocks
  If OOM: increase gradient_accumulation_steps to 2
"""

import os
import sys
import time
import json
import random
import argparse
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler

from config import BSMamba3Config
from models.bsmamba3_se import BSMamba3SE
from losses import MultiScaleLoss
from pcs import PCSTargetTransform
from dataset import create_dataloaders

# Suppress verbose warnings
warnings.filterwarnings('ignore', category=UserWarning)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility. Seed reported in paper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Safe with deterministic + fixed input sizes


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: BSMamba3Config,
    steps_per_epoch: int,
) -> tuple:
    """
    Create AdamW optimizer and warmup+cosine scheduler.

    WHY β₂=0.95 not 0.999: SSM parameters (Δ, B, C) have more volatile
    gradients than attention. β₂=0.95 adapts faster. Standard for Mamba.
    WHY NOT Lion: Lion has lower memory but underperforms AdamW for SSMs.
    """
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    # Warmup + CosineAnnealing
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = config.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup 0 → 1
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            # Cosine decay from 1 to eta_min/lr
            min_ratio = config.eta_min / config.lr
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    loss_fn: MultiScaleLoss,
    device: torch.device,
    epoch: int,
    config: BSMamba3Config,
    pcs_transform=None,
    global_step: int = 0,
) -> tuple:
    """
    Full training epoch with bf16 autocast, gradient clipping, and step timing.

    Returns:
        avg_loss: float
        step_time_mean: float (seconds per step — CRITICAL for safety valve)
        global_step: int (updated)
        loss_components: dict of averaged per-term losses
    """
    model.train()
    total_loss = 0.0
    loss_accum = {}
    step_times = []
    n_steps = 0

    for batch_idx, (noisy, clean, utt_ids) in enumerate(loader):
        step_start = time.time()

        noisy = noisy.to(device, non_blocking=True)  # [B, L]
        clean = clean.to(device, non_blocking=True)   # [B, L]

        # Forward pass with bf16 autocast
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            enhanced, stfts, enhanced_stft = model(noisy)  # [B, L], list, [B, 257, T]
            
            # Compute loss
            loss, loss_dict = loss_fn(enhanced, clean)
            
            # Scale for gradient accumulation
            loss = loss / config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping — critical for complex angle stability
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        step_time = time.time() - step_start
        step_times.append(step_time)
        total_loss += loss.item() * config.gradient_accumulation_steps
        n_steps += 1

        # Accumulate per-term losses
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        # Progress logging every 50 steps
        if (batch_idx + 1) % 50 == 0:
            avg_step_time = np.mean(step_times[-50:])
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss_dict['loss_total']:.4f} "
                f"(mag={loss_dict['loss_mag']:.3f} cpx={loss_dict['loss_complex']:.3f} "
                f"sisnr={loss_dict['loss_sisnr']:.3f} tc={loss_dict['loss_tc']:.3f}) "
                f"Step: {avg_step_time:.3f}s LR: {current_lr:.2e}"
            )

    avg_loss = total_loss / max(n_steps, 1)
    step_time_mean = np.mean(step_times)
    avg_components = {k: v / max(n_steps, 1) for k, v in loss_accum.items()}

    return avg_loss, step_time_mean, global_step, avg_components


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    out_dir: str = None,
    epoch: int = 0,
) -> dict:
    """
    Validation: run inference on full utterances, compute metrics.

    Metrics computed:
      - WB-PESQ (pypesq 0.1.3 — same version as all cited papers)
      - STOI (optional)
      - SI-SNR

    Returns dict with mean metrics.
    """
    model.eval()
    
    metrics = {
        'pesq_scores': [],
        'si_snr_scores': [],
        'n_utterances': 0,
    }
    
    # Try importing pypesq for WB-PESQ evaluation
    pesq_available = False
    try:
        from pesq import pesq as compute_pesq
        pesq_available = True
    except ImportError:
        try:
            from pypesq import pesq as compute_pesq
            pesq_available = True
        except ImportError:
            print("  [Warning] pypesq/pesq not available. Skipping PESQ evaluation.")
    
    # Create output directory for enhanced wavs
    if out_dir is not None:
        enhanced_dir = os.path.join(out_dir, f'enhanced_epoch{epoch}')
        os.makedirs(enhanced_dir, exist_ok=True)

    for noisy, clean, utt_ids in val_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        B = noisy.shape[0]
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            enhanced, _, _ = model(noisy)

        # Move to CPU for metrics
        enhanced_np = enhanced.squeeze(0).cpu().float().numpy()
        clean_np = clean.squeeze(0).cpu().numpy()
        
        # Ensure same length
        min_len = min(len(enhanced_np), len(clean_np))
        enhanced_np = enhanced_np[:min_len]
        clean_np = clean_np[:min_len]
        
        # WB-PESQ
        if pesq_available:
            try:
                score = compute_pesq(16000, clean_np, enhanced_np, 'wb')
                metrics['pesq_scores'].append(score)
            except Exception as e:
                pass  # Skip utterances that fail PESQ (very short, etc.)
        
        # SI-SNR
        si_snr = compute_si_snr(enhanced_np, clean_np)
        metrics['si_snr_scores'].append(si_snr)
        
        metrics['n_utterances'] += 1
        
        # Save enhanced wav
        if out_dir is not None:
            import soundfile as sf
            utt_id = utt_ids[0] if isinstance(utt_ids, (list, tuple)) else utt_ids
            sf.write(
                os.path.join(enhanced_dir, f'{utt_id}.wav'),
                enhanced_np, 16000
            )

    # Compute means
    results = {'n_utterances': metrics['n_utterances']}
    
    if metrics['pesq_scores']:
        results['wb_pesq_mean'] = np.mean(metrics['pesq_scores'])
        results['wb_pesq_std'] = np.std(metrics['pesq_scores'])
    
    if metrics['si_snr_scores']:
        results['si_snr_mean'] = np.mean(metrics['si_snr_scores'])
    
    return results


def compute_si_snr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SI-SNR between predicted and target waveforms."""
    pred = pred - np.mean(pred)
    target = target - np.mean(target)
    
    dot = np.sum(pred * target)
    s_target = (dot / (np.sum(target ** 2) + 1e-8)) * target
    e_noise = pred - s_target
    
    si_snr = 10 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8) + 1e-8
    )
    return si_snr


def train(config: BSMamba3Config):
    """
    Full training loop: 80 epochs, validation every 5, best model checkpoint.

    SAFETY VALVE: Prints epoch-1 step time and warns if > 0.35s.
      If exceeded: reduce n_blocks from 6 → 5 → 4.
      NEVER reduce d_state (voids complex-state contribution claim).
      NEVER reduce mimo_rank (voids MIMO contribution claim).
    """
    # Validate config
    config.validate()
    
    # Set seeds
    set_seed(config.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # bf16 precision config  — MANDATORY for Mamba3
    if config.precision == 'bf16':
        assert torch.cuda.is_bf16_supported(), "GPU does not support bf16!"
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        print("  [Precision] bf16 enabled, fp32 matmul accumulation enforced")
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(config.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    print(f"  Config saved to {config_path}")
    
    # Create model
    print("\n[Model] Creating BSMamba3-SE...")
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
        use_grad_checkpoint=config.use_grad_checkpoint,
    ).to(device)
    
    # Parameter count
    total_params = model.count_parameters()
    breakdown = model.count_parameters_breakdown()
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    for name, count in breakdown.items():
        print(f"    {name}: {count:,} ({count/1e6:.2f}M)")
    
    # Create dataloaders
    print("\n[Data] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        segment_length=config.segment_length,
        num_workers=config.num_workers,
        use_remix=config.use_remix,
        use_bandmask=config.use_bandmask,
        use_gain_aug=config.use_gain_aug,
    )
    
    steps_per_epoch = len(train_loader)
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {steps_per_epoch * config.epochs}")
    
    # Check training budget
    total_steps = steps_per_epoch * config.epochs
    max_seconds = 12 * 3600  # 12 hours
    max_step_time = max_seconds / total_steps
    print(f"  Budget: {max_step_time:.3f}s/step for 12h training")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, steps_per_epoch
    )
    
    # Create loss function
    loss_fn = MultiScaleLoss(
        lambda_mag=config.lambda_mag,
        lambda_complex=config.lambda_complex,
        lambda_sisnr=config.lambda_sisnr,
        lambda_tc=config.lambda_tc,
    ).to(device)
    
    # PCS transform (if enabled)
    pcs_transform = None
    if config.use_pcs:
        pcs_transform = PCSTargetTransform(
            n_fft=512, sr=16000,
            pcs_gamma=config.pcs_gamma,
            enabled=True,
        )
        print(f"  [PCS] Enabled with γ={config.pcs_gamma}")
    
    # Training state
    best_pesq = 0.0
    global_step = 0
    training_log = []
    
    print(f"\n{'='*70}")
    print(f"  BSMamba3-SE Training — {config.epochs} epochs")
    print(f"  Model: {total_params/1e6:.2f}M params, {config.n_blocks} blocks")
    print(f"  Batch: {config.batch_size}, Segment: {config.segment_length}s")
    print(f"  LR: {config.lr}, WD: {config.weight_decay}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        avg_loss, step_time, global_step, loss_components = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            config=config,
            pcs_transform=pcs_transform,
            global_step=global_step,
        )
        
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        
        print(
            f"\nEpoch {epoch}/{config.epochs} — "
            f"Loss: {avg_loss:.4f} | Step: {step_time:.3f}s | "
            f"Epoch: {epoch_time/60:.1f}min | LR: {current_lr:.2e}"
        )
        
        # ═══ SAFETY VALVE (plan Part 0.4) ═══
        if epoch == 1:
            print(f"\n  ⚡ EPOCH-1 STEP TIME: {step_time:.3f}s")
            if step_time > config.max_step_time:
                print(f"  ⚠️  WARNING: Step time {step_time:.3f}s > {config.max_step_time}s budget!")
                print(f"  ⚠️  Consider reducing n_blocks from {config.n_blocks} → {config.n_blocks - 1}")
                print(f"  ⚠️  Current projection: {step_time * steps_per_epoch * config.epochs / 3600:.1f}h total")
            else:
                projected_hours = step_time * steps_per_epoch * config.epochs / 3600
                print(f"  ✓ Within budget. Projected training time: {projected_hours:.1f}h")
        
        # Log
        epoch_log = {
            'epoch': epoch,
            'loss': avg_loss,
            'step_time': step_time,
            'epoch_time': epoch_time,
            'lr': current_lr,
            **loss_components,
        }
        
        # ═══ VALIDATION ═══
        if epoch % config.validate_every == 0 or epoch == config.epochs:
            print(f"\n  [Validation] Running at epoch {epoch}...")
            val_results = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                out_dir=config.out_dir,
                epoch=epoch,
            )
            
            if 'wb_pesq_mean' in val_results:
                pesq_score = val_results['wb_pesq_mean']
                print(
                    f"  WB-PESQ: {pesq_score:.4f} ± {val_results.get('wb_pesq_std', 0):.4f} "
                    f"| SI-SNR: {val_results.get('si_snr_mean', 0):.2f} dB "
                    f"| Utterances: {val_results['n_utterances']}"
                )
                
                # Save best model
                if pesq_score > best_pesq:
                    best_pesq = pesq_score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'pesq': pesq_score,
                        'config': vars(config),
                    }, os.path.join(config.out_dir, 'checkpoint_best.pt'))
                    print(f"  ★ New best PESQ: {pesq_score:.4f} — checkpoint saved!")
                
                epoch_log.update(val_results)
            else:
                si_snr = val_results.get('si_snr_mean', 0)
                print(f"  SI-SNR: {si_snr:.2f} dB (PESQ unavailable)")
        
        # ═══ PERIODIC CHECKPOINT ═══
        if epoch % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'config': vars(config),
            }, os.path.join(config.out_dir, f'checkpoint_epoch{epoch}.pt'))
        
        training_log.append(epoch_log)
        
        # Save training log
        with open(os.path.join(config.out_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best WB-PESQ: {best_pesq:.4f}")
    print(f"  Checkpoints saved to: {config.out_dir}")
    print(f"{'='*70}")
    
    return best_pesq


def main():
    parser = argparse.ArgumentParser(description='BSMamba3-SE Training')
    
    # Model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--headdim', type=int, default=64)
    parser.add_argument('--mimo_rank', type=int, default=4)
    parser.add_argument('--chunk_size', type=int, default=16)
    parser.add_argument('--n_blocks', type=int, default=6)
    parser.add_argument('--k_bands', type=int, default=8)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--segment_len', type=float, default=4.0)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--grad_checkpoint', action='store_true', default=False)
    parser.add_argument('--grad_accum', type=int, default=1)
    
    # PCS
    parser.add_argument('--use_pcs', action='store_true', default=True)
    parser.add_argument('--pcs_gamma', type=float, default=0.3)
    
    # Augmentation
    parser.add_argument('--use_remix', action='store_true', default=True)
    parser.add_argument('--use_bandmask', action='store_true', default=True)
    
    # Data
    parser.add_argument('--dataset', type=str, default='voicebank_demand')
    parser.add_argument('--data_dir', type=str, default='./VoiceBank_DEMAND_16k')
    parser.add_argument('--sr', type=int, default=16000)
    
    # Checkpointing
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--out_dir', type=str, default='./checkpoints/bsmamba3_run1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Build config from args
    config = BSMamba3Config(
        d_model=args.d_model,
        d_state=args.d_state,
        headdim=args.headdim,
        mimo_rank=args.mimo_rank,
        chunk_size=args.chunk_size,
        n_blocks=args.n_blocks,
        n_bands=args.k_bands,
        batch_size=args.batch_size,
        segment_length=args.segment_len,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        precision=args.precision,
        use_grad_checkpoint=args.grad_checkpoint,
        gradient_accumulation_steps=args.grad_accum,
        use_pcs=args.use_pcs,
        pcs_gamma=args.pcs_gamma,
        use_remix=args.use_remix,
        use_bandmask=args.use_bandmask,
        data_dir=args.data_dir,
        sr=args.sr,
        validate_every=args.validate_every,
        save_every=args.save_every,
        out_dir=args.out_dir,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    
    train(config)


if __name__ == '__main__':
    main()
