"""
Phase 4 - Train the from-scratch Transformer on Multi30k EN->DE.

Implements the training regime of Section 5 of the paper:
  * Adam (beta1=0.9, beta2=0.98, eps=1e-9)
  * Noam learning-rate schedule: lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)   (Eq. 3)
  * Label-smoothed cross-entropy (eps_ls = 0.1), padding positions ignored
  * Residual / embedding dropout (P_drop = 0.1)
  * Teacher forcing with padding + causal masks
Plus: gradient clipping, best-checkpoint saving, reproducible seeds, CPU/GPU support.

Only the train and valid splits are used here. The test split is used exclusively by evaluate.py.

Examples:
  python train.py --config small
  python train.py --config base --epochs 20
  python train.py --config small --subset 2000 --epochs 2 --tag smoke   # quick check
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

import config as C
from data_utils import load_tokenizer, make_loader
from models import Transformer, count_parameters
from plotting import plot_loss_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def noam_lr(step: int, d_model: int, warmup: int, factor: float = 1.0) -> float:
    """Eq. (3) of the paper. `step` starts at 1."""
    step = max(step, 1)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def select_device(requested: str, model_cfg: C.ModelConfig) -> torch.device:
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Device: GPU {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")
        if model_cfg.name == "base" and total_gb < 6:
            print("RECOMMENDATION: < 6 GB GPU memory - consider `--config small` or a smaller "
                  "--batch-size. The base configuration itself is left unchanged.")
        return device
    print(f"Device: CPU ({torch.get_num_threads()} threads) - no CUDA GPU detected")
    if model_cfg.name == "base":
        print("RECOMMENDATION: the base configuration (~48M params with an 8k vocabulary) needs ~5 s per "
              "batch on a laptop CPU (8+ hours for 25 epochs). Use `--config small` for practical CPU "
              "training. The base configuration is left unchanged.")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, grad_clip=None,
              max_steps=None, log_every=50):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.

    Returns (label-smoothed loss per token, plain NLL per token, number of steps).
    """
    training = optimizer is not None
    model.train(training)
    nll_fn = nn.CrossEntropyLoss(ignore_index=C.PAD_ID, reduction="sum")
    total_loss = total_nll = 0.0
    total_tokens = steps = 0
    t0 = time.time()
    with torch.set_grad_enabled(training):
        for src, tgt, _ in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]          # teacher forcing (shift right)
            logits = model(src, tgt_in)
            flat_logits, flat_gold = logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
            loss = criterion(flat_logits, flat_gold)           # mean over non-pad tokens
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")
            n_tokens = int((tgt_out != C.PAD_ID).sum())

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * n_tokens
            with torch.no_grad():
                total_nll += nll_fn(flat_logits, flat_gold).item()
            total_tokens += n_tokens
            steps += 1
            if training and log_every and steps % log_every == 0:
                print(f"    step {steps:5d}/{len(loader)}  loss {total_loss / total_tokens:.4f}  "
                      f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                      f"{total_tokens / (time.time() - t0):,.0f} tok/s", flush=True)
            if max_steps and steps >= max_steps:
                break
    return total_loss / total_tokens, total_nll / total_tokens, steps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", choices=list(C.MODEL_CONFIGS), default="small")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--warmup", type=int, help="override warmup steps")
    p.add_argument("--subset", type=int, help="use only the first N training pairs (debugging)")
    p.add_argument("--max-steps-per-epoch", type=int, help="debugging / benchmarking")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, help="torch CPU threads")
    p.add_argument("--tag", help="run name (default: config name)")
    args = p.parse_args()

    model_cfg, train_cfg = C.get_config(args.config)
    overrides = {k: v for k, v in {"epochs": args.epochs, "batch_size": args.batch_size,
                                   "warmup_steps": args.warmup}.items() if v is not None}
    train_cfg = replace(train_cfg, **overrides)   # architecture is never overridden
    tag = args.tag or args.config

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(train_cfg.seed)
    device = select_device(args.device, model_cfg)

    print("=" * 70)
    print(f"ACTIVE CONFIGURATION: '{model_cfg.name}'  (run tag: {tag})")
    print(C.format_config(model_cfg, train_cfg))
    if overrides:
        print(f"Training overrides from command line: {overrides}")
    print("=" * 70)

    sp = load_tokenizer()
    vocab_size = sp.get_piece_size()
    train_loader = make_loader("train", sp, train_cfg.batch_size, shuffle=True,
                               max_tokens=train_cfg.max_train_tokens, subset=args.subset, seed=train_cfg.seed)
    valid_loader = make_loader("valid", sp, train_cfg.batch_size, shuffle=False)
    print(f"Train pairs: {len(train_loader.dataset)}  Valid pairs: {len(valid_loader.dataset)}  "
          f"Vocab: {vocab_size}  Steps/epoch: {len(train_loader)}")

    model = Transformer.from_config(model_cfg, vocab_size, C.PAD_ID).to(device)
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                 betas=(train_cfg.adam_beta1, train_cfg.adam_beta2), eps=train_cfg.adam_eps)
    # LambdaLR multiplies the base lr (1.0) by noam_lr; scheduler step k -> update number k+1.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda k: noam_lr(k + 1, model_cfg.d_model, train_cfg.warmup_steps, train_cfg.lr_factor))
    criterion = nn.CrossEntropyLoss(ignore_index=C.PAD_ID, label_smoothing=train_cfg.label_smoothing)

    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = C.CHECKPOINT_DIR / f"{tag}_best.pt"
    history = {"epoch": [], "train_loss": [], "train_nll": [], "val_loss": [], "val_nll": [],
               "val_ppl": [], "lr": [], "epoch_time_sec": [], "step": []}
    best_val, bad_epochs, global_step = math.inf, 0, 0
    run_start = time.time()

    for epoch in range(1, train_cfg.epochs + 1):
        train_loader.batch_sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss, train_nll, steps = run_epoch(model, train_loader, criterion, device, optimizer, scheduler,
                                                 train_cfg.grad_clip, args.max_steps_per_epoch)
        global_step += steps
        val_loss, val_nll, _ = run_epoch(model, valid_loader, criterion, device)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(history, [epoch, train_loss, train_nll, val_loss, val_nll, math.exp(val_nll), lr,
                                  dt, global_step]):
            history[k].append(v)

        improved = val_loss < best_val
        print(f"Epoch {epoch:3d}/{train_cfg.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| val NLL {val_nll:.4f} (PPL {math.exp(val_nll):6.2f}) | lr {lr:.2e} | step {global_step} "
              f"| {dt / 60:5.1f} min{'  * best, saved' if improved else ''}", flush=True)

        if improved:
            best_val, bad_epochs = val_loss, 0
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model_cfg.to_dict(),
                "train_config": train_cfg.to_dict(),
                "vocab_size": vocab_size,
                "tokenizer": str(C.SPM_MODEL_PATH.relative_to(C.ROOT_DIR)),
                "epoch": epoch,
                "step": global_step,
                "val_loss": val_loss,
                "val_nll": val_nll,
                "num_parameters": n_params,
                "subset": args.subset,
            }, best_path)
        else:
            bad_epochs += 1

        summary = {"tag": tag, "config": model_cfg.to_dict(), "train_config": train_cfg.to_dict(),
                   "overrides": overrides, "device": str(device), "num_parameters": n_params,
                   "train_pairs": len(train_loader.dataset), "valid_pairs": len(valid_loader.dataset),
                   "best_epoch": history["epoch"][int(np.argmin(history["val_loss"]))],
                   "best_val_loss": best_val, "total_train_time_sec": time.time() - run_start,
                   "history": history}
        (C.METRICS_DIR / f"{tag}_history.json").write_text(json.dumps(summary, indent=2))
        plot_loss_curves(history, C.GRAPHS_DIR / f"{tag}_loss_curve.png",
                         f"Transformer ({model_cfg.name}) on Multi30k EN→DE: training vs validation loss")

        if train_cfg.patience and bad_epochs >= train_cfg.patience:
            print(f"Early stopping: no validation improvement for {train_cfg.patience} epochs")
            break

    total = time.time() - run_start
    print(f"Training finished in {total / 60:.1f} min. Best val loss {best_val:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
