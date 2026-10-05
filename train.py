"""
Training loop.

Implements every item from the "MOST IMPORTANT REQUIREMENTS" list:
  - mini-batch training with configurable batch size
  - gradient accumulation (effective batch size = batch_size * accum_steps)
  - mixed precision (auto-disabled on CPU)
  - gradient clipping
  - label smoothing
  - Noam-style warmup + inverse-sqrt LR decay (the schedule the
    original Transformer paper used; well suited to training from
    scratch, unlike a flat LR which is what the old notebook had)
  - checkpoint saved after EVERY epoch (checkpoints/latest.pt)
  - best.pt updated whenever validation loss improves
  - full resume: model, optimizer, scaler, scheduler step, epoch number
  - old checkpoints pruned to keep_last_n_checkpoints (best.pt and
    latest.pt are never pruned)

Run:
    python train.py
Resume:
    set CONFIG["resume"] = True (config.py) or edit resume_checkpoint,
    then re-run the same command -- training continues from the next
    epoch, it does not restart at epoch 1.
"""

import os
import glob
import time
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import CONFIG, ensure_dirs, resolve_device
from data_pipeline import load_split_ids, load_sentencepiece
from dataset import build_dataloader
from model import build_model, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NoamScheduler:
    """LR = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5), scaled so
    the peak LR equals cfg['learning_rate']. This is the schedule from
    'Attention Is All You Need' -- warms up linearly then decays, which
    is what lets Transformers train stably from a random init without
    hand-tuning a fixed LR."""

    def __init__(self, optimizer, d_model, warmup_steps, peak_lr):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        natural_peak = (d_model ** -0.5) * (warmup_steps ** -0.5)
        self.scale = peak_lr / natural_peak

    def step(self):
        self.step_num += 1
        lr = self.scale * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5)
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, state):
        self.step_num = state["step_num"]


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                     train_loss, val_loss, cfg):
    tmp_path = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": cfg,
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX -- never leaves a half-written checkpoint


def prune_old_checkpoints(checkpoint_dir, keep_last_n):
    epoch_ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    for p in (epoch_ckpts[:-keep_last_n] if keep_last_n > 0 else []):
        os.remove(p)


def run_epoch(model, dataloader, criterion, device, cfg,
              optimizer=None, scheduler=None, scaler=None, train=True):
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0
    accum = cfg["gradient_accumulation_steps"] if train else 1
    desc = "Train" if train else "Valid"

    pbar = tqdm(dataloader, desc=desc)
    if train:
        optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        decoder_input = tgt[:, :-1]
        decoder_target = tgt[:, 1:]

        with torch.set_grad_enabled(train), torch.autocast(
            device_type=device.type, enabled=(cfg["use_amp"] and device.type == "cuda")
        ):
            output = model(src, decoder_input)
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                decoder_target.contiguous().view(-1),
            )

        n_tokens = (decoder_target != cfg["pad_id"]).sum().item()

        if train:
            loss_to_backprop = loss / accum
            if scaler is not None:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if (step + 1) % accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
                lr = scheduler.step()
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def main(cfg=CONFIG):
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    device = resolve_device()
    print("Device:", device)

    sp_hi, sp_en = load_sentencepiece(cfg)
    train_src, train_tgt = load_split_ids("train", cfg)
    valid_src, valid_tgt = load_split_ids("valid", cfg)

    train_loader = build_dataloader(train_src, train_tgt, cfg, shuffle=True)
    valid_loader = build_dataloader(valid_src, valid_tgt, cfg, shuffle=False)

    model = build_model(cfg, sp_hi.get_piece_size(), sp_en.get_piece_size(), device)
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,} | Trainable: {trainable:,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg["pad_id"], label_smoothing=cfg["label_smoothing"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], betas=(0.9, 0.98), eps=1e-9,
        weight_decay=0.01,
    )
    scheduler = NoamScheduler(optimizer, cfg["d_model"], cfg["warmup_steps"], cfg["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["use_amp"] and device.type == "cuda"))

    start_epoch = 1
    best_val_loss = float("inf")

    if cfg["resume"] and os.path.exists(cfg["resume_checkpoint"]):
        print(f"Resuming from {cfg['resume_checkpoint']}")
        ckpt = torch.load(cfg["resume_checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, best_val_loss so far = {best_val_loss:.4f}")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        print(f"\n{'=' * 70}\nEpoch {epoch}/{cfg['num_epochs']}\n{'=' * 70}")
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, device, cfg,
                                optimizer, scheduler, scaler, train=True)
        val_loss = run_epoch(model, valid_loader, criterion, device, cfg, train=False)

        epoch_time = time.time() - t0
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} "
              f"| Val ppl: {np.exp(val_loss):.2f} | Time: {epoch_time:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # ---- checkpoint after EVERY epoch (never skipped, never optional) ----
        latest_path = os.path.join(cfg["checkpoint_dir"], "latest.pt")
        epoch_path = os.path.join(cfg["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pt")
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, train_loss, val_loss, cfg)
        save_checkpoint(epoch_path, model, optimizer, scheduler, scaler, epoch, train_loss, val_loss, cfg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, train_loss, val_loss, cfg)
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")

        prune_old_checkpoints(cfg["checkpoint_dir"], cfg["keep_last_n_checkpoints"])

    return history


if __name__ == "__main__":
    main()
