"""
BSMamba3-SE Training Script
═══════════════════════════
GAN training with MetricGAN discriminator and 7-component loss.

Training loop:
  1. Forward pass: noisy_mag, noisy_pha → generator → denoised_mag, denoised_pha
  2. iSTFT → enhanced waveform
  3. Discriminator step: train D on real PESQ scores
  4. Generator step: 7 losses (mag + phase + complex + time + consistency + tc + metric)
"""

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pesq import pesq

from config import BSMamba3Config
from models.bsmamba3_se import BSMamba3SE, mag_phase_stft, mag_phase_istft
from models.discriminator import MetricDiscriminator, batch_pesq
from losses import phase_losses, temporal_coherence_loss
from dataset import create_train_dataset, create_valid_dataset


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def pesq_score_batch(clean_list, enhanced_list, sr=16000):
    """Compute mean PESQ across a batch of utterances."""
    scores = []
    for c, e in zip(clean_list, enhanced_list):
        try:
            s = pesq(sr, c.squeeze(), e.squeeze(), 'wb')
            scores.append(s)
        except Exception:
            pass
    return np.mean(scores) if scores else 0.0


def train(cfg: BSMamba3Config):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        raise RuntimeError("BSMamba3-SE requires GPU acceleration")

    set_seed(cfg.seed)
    cfg.validate()

    # Shortcuts
    n_fft = cfg.n_fft
    hop_size = cfg.hop_size
    win_size = cfg.win_size
    compress_factor = cfg.compress_factor

    # ── Models ──
    generator = BSMamba3SE(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)

    n_params = generator.count_parameters()
    print(f"[Model] Generator: {n_params / 1e6:.2f}M parameters")
    print(f"[Model] Discriminator: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M parameters")

    # ── Optimizers ──
    optim_g = optim.AdamW(
        generator.parameters(), lr=cfg.lr,
        betas=(cfg.adam_b1, cfg.adam_b2),
    )
    optim_d = optim.AdamW(
        discriminator.parameters(), lr=cfg.lr,
        betas=(cfg.adam_b1, cfg.adam_b2),
    )

    # ── LR Schedulers ──
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=cfg.lr_decay)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=cfg.lr_decay)

    # ── Datasets ──
    trainset = create_train_dataset(cfg)
    train_loader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )

    validset = create_valid_dataset(cfg)
    valid_loader = DataLoader(
        validset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True,
    )

    # ── Logging ──
    os.makedirs(cfg.out_dir, exist_ok=True)
    sw = SummaryWriter(os.path.join(cfg.out_dir, 'logs'))

    # ── Training state ──
    steps = 0
    best_pesq = 0.0
    best_pesq_step = 0
    one_labels = torch.ones(cfg.batch_size).to(device)

    print(f"\n{'='*60}")
    print(f"  BSMamba3-SE Training")
    print(f"  Epochs: {cfg.epochs}, Batch: {cfg.batch_size}, LR: {cfg.lr}")
    print(f"  STFT: n_fft={n_fft}, hop={hop_size}, compress={compress_factor}")
    print(f"  Device: {device}")
    print(f"  Output: {cfg.out_dir}")
    print(f"{'='*60}\n")

    generator.train()
    discriminator.train()

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        for i, batch in enumerate(train_loader):
            step_start = time.time()

            # Unpack batch
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=True)

            # ── Forward pass ──
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            # Reconstruct waveform for time loss and PESQ
            audio_g = mag_phase_istft(
                mag_g, pha_g, n_fft, hop_size, win_size, compress_factor
            )

            # Match waveform lengths
            min_len = min(audio_g.shape[-1], clean_audio.shape[-1])
            audio_g = audio_g[..., :min_len]
            clean_audio_trimmed = clean_audio[..., :min_len]

            # Compute batch PESQ for discriminator supervision
            audio_list_r = list(clean_audio_trimmed.detach().cpu().numpy())
            audio_list_g = list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(
                audio_list_r, audio_list_g, sr=cfg.sr, n_jobs=cfg.num_workers
            )

            # ═══════════════════════════════════════════════
            # DISCRIMINATOR STEP
            # ═══════════════════════════════════════════════
            optim_d.zero_grad()

            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g.detach())

            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = torch.tensor(0.0, device=device)

            loss_disc = loss_disc_r + loss_disc_g
            loss_disc.backward()
            optim_d.step()

            # ═══════════════════════════════════════════════
            # GENERATOR STEP
            # ═══════════════════════════════════════════════
            optim_g.zero_grad()

            # 1. Magnitude Loss (L2, compressed domain)
            loss_mag = F.mse_loss(clean_mag, mag_g)

            # 2. Phase Loss (anti-wrapping: IP + GD + IAF)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, n_fft)
            loss_phase = loss_ip + loss_gd + loss_iaf

            # 3. Complex Loss (L2, compressed domain)
            loss_complex = F.mse_loss(clean_com, com_g) * 2

            # 4. Time Loss (L1, waveform)
            loss_time = F.l1_loss(clean_audio_trimmed, audio_g)

            # 5. Metric Loss (MetricGAN adversarial)
            metric_g_for_gen = discriminator(clean_mag, mag_g)
            loss_metric = F.mse_loss(metric_g_for_gen.flatten(), one_labels)

            # 6. Consistency Loss (STFT round-trip)
            _, _, rec_com = mag_phase_stft(
                audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True
            )
            loss_consistency = F.mse_loss(com_g, rec_com) * 2

            # 7. Temporal Coherence Loss (L_tc, our contribution)
            loss_tc = temporal_coherence_loss(mag_g, clean_mag)

            # Total generator loss
            loss_gen = (
                cfg.lambda_metric * loss_metric
                + cfg.lambda_mag * loss_mag
                + cfg.lambda_phase * loss_phase
                + cfg.lambda_complex * loss_complex
                + cfg.lambda_time * loss_time
                + cfg.lambda_consistency * loss_consistency
                + cfg.lambda_tc * loss_tc
            )

            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip)
            optim_g.step()

            epoch_loss += loss_gen.item()
            epoch_steps += 1
            step_time = time.time() - step_start

            # ── Logging ──
            if steps % cfg.stdout_interval == 0:
                print(
                    f"Epoch {epoch+1}/{cfg.epochs} [{i+1}/{len(train_loader)}] "
                    f"G:{loss_gen.item():.3f} D:{loss_disc.item():.3f} "
                    f"mag:{loss_mag.item():.3f} pha:{loss_phase.item():.3f} "
                    f"com:{loss_complex.item():.3f} time:{loss_time.item():.3f} "
                    f"con:{loss_consistency.item():.3f} tc:{loss_tc.item():.3f} "
                    f"met:{loss_metric.item():.3f} "
                    f"Step:{step_time:.3f}s LR:{optim_g.param_groups[0]['lr']:.2e}"
                )

            if steps % cfg.summary_interval == 0:
                sw.add_scalar("Train/Gen_Loss", loss_gen.item(), steps)
                sw.add_scalar("Train/Disc_Loss", loss_disc.item(), steps)
                sw.add_scalar("Train/Mag_Loss", loss_mag.item(), steps)
                sw.add_scalar("Train/Phase_Loss", loss_phase.item(), steps)
                sw.add_scalar("Train/Complex_Loss", loss_complex.item(), steps)
                sw.add_scalar("Train/Time_Loss", loss_time.item(), steps)
                sw.add_scalar("Train/Consistency_Loss", loss_consistency.item(), steps)
                sw.add_scalar("Train/TC_Loss", loss_tc.item(), steps)
                sw.add_scalar("Train/Metric_Loss", loss_metric.item(), steps)

            # ── NaN check ──
            if torch.isnan(loss_gen):
                raise ValueError(f"NaN loss at step {steps}!")

            # ── Checkpointing ──
            if steps % cfg.checkpoint_interval == 0 and steps > 0:
                save_path_g = os.path.join(cfg.out_dir, f"g_{steps:08d}.pt")
                torch.save({'generator': generator.state_dict()}, save_path_g)
                save_path_d = os.path.join(cfg.out_dir, f"do_{steps:08d}.pt")
                torch.save({
                    'discriminator': discriminator.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'steps': steps,
                    'epoch': epoch,
                }, save_path_d)

            # ── Validation ──
            if steps % cfg.validation_interval == 0 and steps > 0:
                generator.eval()
                torch.cuda.empty_cache()

                audios_r, audios_g = [], []
                val_mag_err, val_pha_err, val_com_err = 0.0, 0.0, 0.0

                with torch.no_grad():
                    for j, val_batch in enumerate(valid_loader):
                        v_clean_audio, v_clean_mag, v_clean_pha, v_clean_com, \
                            v_noisy_mag, v_noisy_pha = val_batch

                        v_clean_audio = v_clean_audio.to(device)
                        v_clean_mag = v_clean_mag.to(device)
                        v_clean_pha = v_clean_pha.to(device)
                        v_clean_com = v_clean_com.to(device)
                        v_noisy_mag = v_noisy_mag.to(device)
                        v_noisy_pha = v_noisy_pha.to(device)

                        v_mag_g, v_pha_g, v_com_g = generator(v_noisy_mag, v_noisy_pha)

                        v_audio_g = mag_phase_istft(
                            v_mag_g, v_pha_g, n_fft, hop_size, win_size, compress_factor
                        )

                        audios_r += torch.split(v_clean_audio, 1, dim=0)
                        audios_g += torch.split(v_audio_g, 1, dim=0)

                        val_mag_err += F.mse_loss(v_clean_mag, v_mag_g).item()
                        v_ip, v_gd, v_iaf = phase_losses(v_clean_pha, v_pha_g, n_fft)
                        val_pha_err += (v_ip + v_gd + v_iaf).item()
                        val_com_err += F.mse_loss(v_clean_com, v_com_g).item()

                n_val = len(valid_loader)
                val_mag_err /= n_val
                val_pha_err /= n_val
                val_com_err /= n_val

                # Compute PESQ on validation set
                val_pesq = pesq_score_batch(
                    [a.cpu().numpy() for a in audios_r],
                    [a.cpu().numpy() for a in audios_g],
                    sr=cfg.sr,
                )

                print(f"\n  [Validation] Step {steps}: PESQ={val_pesq:.4f} "
                      f"Mag={val_mag_err:.4f} Phase={val_pha_err:.4f}")

                sw.add_scalar("Val/PESQ", val_pesq, steps)
                sw.add_scalar("Val/Mag_Loss", val_mag_err, steps)
                sw.add_scalar("Val/Phase_Loss", val_pha_err, steps)
                sw.add_scalar("Val/Complex_Loss", val_com_err, steps)

                if val_pesq >= best_pesq:
                    best_pesq = val_pesq
                    best_pesq_step = steps
                    # Save best checkpoint
                    torch.save(
                        {'generator': generator.state_dict()},
                        os.path.join(cfg.out_dir, 'checkpoint_best.pt'),
                    )
                    print(f"  ★ New best PESQ: {best_pesq:.4f} at step {best_pesq_step}")

                print(f"  Best PESQ: {best_pesq:.4f} at step {best_pesq_step}\n")

                generator.train()
                discriminator.train()

            steps += 1

        # End of epoch
        scheduler_g.step()
        scheduler_d.step()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch+1}/{cfg.epochs} - Avg Loss: {avg_loss:.4f} "
              f"| Time: {epoch_time/60:.1f}min | LR: {optim_g.param_groups[0]['lr']:.2e}\n")

    # ── Training complete ──
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best WB-PESQ: {best_pesq:.4f} at step {best_pesq_step}")
    print(f"  Checkpoints saved to: {cfg.out_dir}")
    print(f"{'='*60}\n")

    sw.close()


def main():
    parser = argparse.ArgumentParser(description='BSMamba3-SE Training')
    parser.add_argument('--data_dir', type=str, default='./VoiceBank_DEMAND_16k')
    parser.add_argument('--out_dir', type=str, default='./checkpoints/bsmamba3')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--no_pcs', action='store_true')
    args = parser.parse_args()

    cfg = BSMamba3Config()
    cfg.data_dir = args.data_dir
    cfg.out_dir = args.out_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.no_pcs:
        cfg.use_pcs = False

    train(cfg)


if __name__ == '__main__':
    main()
