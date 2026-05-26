#!/usr/bin/env python3
"""S1: Spearman framing test (PI v2 saliency revamp 2026-05-26).

Compute saliency at end-point (W = W0 + DW) vs start-point (W = W0 + eps*DW)
on a trained baseline adapter. Spearman rho across all (layer, comp) pairs
indicates whether endpoint vs trajectory framing matters.

Decision rule (per directive):
  rho >= 0.5  -> A is NOT critical, demote IG in S2 (skip IG axis)
  rho <  0.3  -> A IS critical, must implement IG
  in between  -> ambiguous, default implement IG

Math (eps-scaled-B trick):
  s_eps = <grad_A(W=W0+eps*DW), A>
        = scaling * (eps*B_orig)^T * G(W_eps) * A_orig.sum
        ~ eps * scaling * <B_orig^T * G(W0), A_orig>   for small eps
        = eps * s_start_i.
  Spearman rho is invariant to multiplicative scale -> we compare s_end vs s_eps directly.

Usage:
  CUDA_VISIBLE_DEVICES=7 python scripts/run_s1_framing_test.py \\
    --base_model /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B \\
    --adapter_dir results/exp_v1/qwen3-8b/tulu3-sft/relora_baseline/seed42/adapter \\
    --out_path analysis/results_v3/saliency_framing/spearman_qwen3-8b_tulu3.json \\
    --n_calib 256 --eps 1e-3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scipy.stats import spearmanr

from src.saliency import first_order_saliency
from src.model import get_lora_BA_handles


def _load_stage3():
    """Side-effect-free import of stage3_run.py module body (no main())."""
    sp = importlib.util.spec_from_file_location(
        "stage3_run_mod", str(ROOT / "scripts" / "stage3_run.py"))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


class _DummyLog:
    def info(self, m): print(f"[INFO] {m}", flush=True)
    def warning(self, m): print(f"[WARN] {m}", flush=True)
    def error(self, m): print(f"[ERR] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--dataset", default="tulu3-sft",
                    choices=["tulu3-sft", "gsm8k", "alpaca"])
    ap.add_argument("--n_calib", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    log = _DummyLog()
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    stage3 = _load_stage3()

    log.info(f"loading tokenizer + base model from {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True,
    )
    base.config.use_cache = False
    log.info(f"loading adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=True)
    model.enable_input_require_grads()
    device = torch.device(args.device)
    model = model.to(device)

    handles = get_lora_BA_handles(model)
    n_components = sum(h.r for h in handles)
    log.info(f"#handles={len(handles)} #components={n_components}")

    log.info(f"building {args.dataset} calib (n={args.n_calib})")
    if args.dataset == "tulu3-sft":
        _, val_ds = stage3.build_tulu3(
            tok, args.max_len, log,
            n_train=args.n_calib * 2, n_val=args.n_calib)
    elif args.dataset == "gsm8k":
        _, val_ds = stage3.build_gsm8k(tok, args.max_len, log, val_size=args.n_calib)
    elif args.dataset == "alpaca":
        _, val_ds = stage3.build_alpaca(
            tok, args.max_len, log,
            n_train=args.n_calib * 2, n_val=args.n_calib)
    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=stage3._pad_collate(pad_id),
    )
    max_batches = max(1, args.n_calib // args.batch_size)

    # ---- s_end at current W = W0 + DW (signed) ----
    log.info("computing s_end (signed) at endpoint W = W0+DW...")
    t0 = time.time()
    sal_end = first_order_saliency(
        model, handles, loader, device,
        max_batches=max_batches, signed=True,
    )
    log.info(f"  s_end done in {time.time() - t0:.1f}s")

    # ---- replace B with eps*B_orig, compute s_eps; restore ----
    log.info(f"perturbing B -> eps*B (eps={args.eps}) and recomputing saliency...")
    B_orig = {h.name: h.B.detach().clone() for h in handles}
    with torch.no_grad():
        for h in handles:
            h.B.data.mul_(args.eps)
    t0 = time.time()
    sal_eps = first_order_saliency(
        model, handles, loader, device,
        max_batches=max_batches, signed=True,
    )
    log.info(f"  s_eps done in {time.time() - t0:.1f}s")
    with torch.no_grad():
        for h in handles:
            h.B.data.copy_(B_orig[h.name])
    del B_orig

    # ---- aggregate flat vectors across all (layer, comp) ----
    s_end_list = []
    s_eps_list = []
    layer_idx = []
    for h in handles:
        s_e = sal_end[h.name].numpy().astype(np.float64)
        s_s = sal_eps[h.name].numpy().astype(np.float64)
        s_end_list.append(s_e)
        s_eps_list.append(s_s)
        layer_idx.extend([h.name] * h.r)
    s_end_vec = np.concatenate(s_end_list)
    s_eps_vec = np.concatenate(s_eps_list)

    rho_global, p_global = spearmanr(s_end_vec, s_eps_vec)
    sign_flip = float(np.mean(np.sign(s_end_vec) != np.sign(s_eps_vec)))

    # Top-10% IoU on "most negative" (helpful, candidates for keep)
    n = len(s_end_vec)
    k = max(int(0.1 * n), 1)
    top_end = set(np.argsort(s_end_vec)[:k].tolist())
    top_eps = set(np.argsort(s_eps_vec)[:k].tolist())
    iou_top10 = len(top_end & top_eps) / len(top_end | top_eps)

    # Top-10% IoU on "most positive" (hurtful, candidates for drop)
    bot_end = set(np.argsort(-s_end_vec)[:k].tolist())
    bot_eps = set(np.argsort(-s_eps_vec)[:k].tolist())
    iou_bot10 = len(bot_end & bot_eps) / len(bot_end | bot_eps)

    # Per-layer Spearman
    per_layer = {}
    for h in handles:
        s_e = sal_end[h.name].numpy().astype(np.float64)
        s_s = sal_eps[h.name].numpy().astype(np.float64)
        if np.std(s_e) > 0 and np.std(s_s) > 0 and h.r >= 3:
            r_, p_ = spearmanr(s_e, s_s)
        else:
            r_, p_ = float("nan"), float("nan")
        per_layer[h.name] = {"rho": float(r_), "p": float(p_), "r": int(h.r)}

    if rho_global >= 0.5:
        decision = "A_NOT_CRITICAL_demote_IG_in_S2"
    elif rho_global < 0.3:
        decision = "A_CRITICAL_implement_IG"
    else:
        decision = "AMBIGUOUS_between_0p3_and_0p5_default_implement_IG"

    result = {
        "args": vars(args),
        "n_components": int(n_components),
        "n_layers": len(handles),
        "rho_global": float(rho_global),
        "p_global": float(p_global),
        "sign_flip_rate": sign_flip,
        "top10pct_iou_keep": float(iou_top10),
        "top10pct_iou_drop": float(iou_bot10),
        "per_layer": per_layer,
        "decision": decision,
    }
    with out.open("w") as f:
        json.dump(result, f, indent=2)
    log.info(f"WROTE {out}")
    log.info(f"rho_global={rho_global:.4f}  p={p_global:.3e}")
    log.info(f"sign_flip_rate={sign_flip:.4f}")
    log.info(f"top10pct_iou (keep)={iou_top10:.4f}  (drop)={iou_bot10:.4f}")
    log.info(f"DECISION: {decision}")


if __name__ == "__main__":
    main()
