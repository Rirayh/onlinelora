#!/usr/bin/env python
"""Stage 0 smoke test: vanilla LoRA on RoBERTa-base + SST-2, 3 epochs equiv.

Acceptance: SST-2 dev accuracy >= 92.0%.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import evaluate
from src.data import load_glue_three_split, make_loaders, tokenize_splits
from src.model import build_lora_model
from src.utils import dump_yaml, get_logger, load_yaml, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage1_sst2.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override total steps (default: from config or 3 epochs).")
    parser.add_argument("--out", default=str(ROOT / "results/stage0"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("stage0", str(out_dir / "smoke.log"))

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}, gpu_count={torch.cuda.device_count()}")

    # Build model
    model, tok = build_lora_model(
        model_name=cfg["model_name"],
        num_labels=cfg["num_labels"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
    )
    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"trainable params = {n_trainable:,}")

    # Data
    splits = load_glue_three_split(
        task=cfg["task"], diagnostic_ratio=cfg["diagnostic_ratio"], seed=cfg["seed"]
    )
    tok_splits = tokenize_splits(splits, cfg["task"], tok, max_len=cfg["max_seq_len"])
    log.info(
        f"#train_main={len(tok_splits.train_main)}, "
        f"#diag={len(tok_splits.diagnostic)}, "
        f"#test_holdout={len(tok_splits.test_holdout)}"
    )
    train_loader, _, test_loader = make_loaders(
        tok_splits, batch_size=cfg["batch_size"], eval_batch_size=cfg["eval_batch_size"]
    )

    # Optimizer
    total_steps = args.max_steps if args.max_steps else (
        cfg["total_steps"] if not args.smoke else 1500
    )
    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"],
    )
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=cfg["optim"]["warmup_steps"], num_training_steps=total_steps
    )

    # Train
    model.train()
    step = 0
    t0 = time.time()
    running_loss = 0.0
    log_every = 100
    eval_every = max(total_steps // 4, 200)
    best_acc = 0.0
    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            running_loss += float(out.loss.item())
            step += 1
            if step % log_every == 0:
                log.info(f"step={step} loss={running_loss/log_every:.4f} lr={sched.get_last_lr()[0]:.2e}")
                running_loss = 0.0
            if step % eval_every == 0 or step == total_steps:
                loss, acc = evaluate(model, test_loader, device)
                log.info(f"[eval @ step {step}] loss={loss:.4f} acc={acc:.4f}")
                best_acc = max(best_acc, acc)
                model.train()
    elapsed = time.time() - t0

    loss, acc = evaluate(model, test_loader, device)
    log.info(f"[final eval] loss={loss:.4f} acc={acc:.4f}  best={max(best_acc,acc):.4f}  elapsed={elapsed:.1f}s")

    mem_peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    metrics = {
        "task": cfg["task"],
        "model": cfg["model_name"],
        "total_steps": total_steps,
        "final_loss": float(loss),
        "final_acc": float(acc),
        "best_acc": float(max(best_acc, acc)),
        "elapsed_sec": float(elapsed),
        "peak_gpu_mem_gb": float(mem_peak),
        "n_train_main": len(tok_splits.train_main),
        "n_diag": len(tok_splits.diagnostic),
        "n_test_holdout": len(tok_splits.test_holdout),
        "trainable_params": int(n_trainable),
        "pass_threshold": float(acc) >= 0.92,
    }
    write_json(str(out_dir / "smoke.json"), metrics)
    dump_yaml(cfg, str(out_dir / "config.yaml"))
    log.info(f"smoke.json -> {out_dir / 'smoke.json'}")
    return 0 if metrics["pass_threshold"] else 2


if __name__ == "__main__":
    sys.exit(main())
