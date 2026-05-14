"""Stage 3: ReLoRA + diagnostic gate on real 7B/8B pretrained models + SFT data.

Pivot from Stage 2's 11M LM-pretraining (val_loss saturated at random init) to
real pretrained models on real SFT tasks where val_loss carries signal.

Methods (4-way ablation, same as Stage 2):
  - A0 lora_vanilla         : standard LoRA, NO merge events
  - A1 relora_baseline      : vanilla ReLoRA (merge ALL components every K steps)
  - A2 relora_diag_gated_S3pos : drop if S3_fo_val_signed > 0  (PI operational default)
  - A3 relora_diag_gated_S3neg : drop if S3_fo_val_signed < 0  (sign-check insurance arm)

Datasets:
  - gsm8k   (config "main"; 7473 train, 1319 test). Response-only CE loss.
  - alpaca  (yahma/alpaca-cleaned; sample 10k train + 500 val).

Outputs under results/stage3/<model_key>/<dataset>/<method>/:
  config.yaml, train_loss.jsonl, val_loss.jsonl,
  effective_rank.jsonl, condition_number.jsonl,
  saliency_at_merge.jsonl (A1/A2/A3 only), summary.json, run.log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peft import LoraConfig, get_peft_model

try:
    from peft import AdaLoraConfig
    _HAS_ADALORA = True
except ImportError:
    _HAS_ADALORA = False

from src.effective_rank import condition_number, effective_rank
from src.model import LoraHandle, count_lora_components, get_lora_BA_handles
from src.saliency import first_order_saliency
from src.utils import append_jsonl, dump_yaml, get_logger, set_seed, write_json


METHOD_CHOICES = [
    "lora_vanilla",
    "relora_baseline",
    "relora_diag_gated_S3pos",
    "relora_diag_gated_S3neg",
    "dora",
    "adalora",
    "relora_random_drop",
    "relora_train_gated",
]
DATASET_CHOICES = ["gsm8k", "alpaca", "tulu3-sft", "metamathqa-10k"]
LOCAL_TULU3_PATH = "/mnt/cpfs/junlongke/onlinelora/datasets/tulu-3-sft-mixture"
LOCAL_METAMATH_PATH = "/mnt/cpfs/junlongke/onlinelora/datasets/MetaMathQA"
TARGET_MODULES_DEFAULT = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# -----------------------------------------------------------------------------
# Data loaders — produce (input_ids, labels) where labels are -100 on prompt
# -----------------------------------------------------------------------------
class SFTDataset(Dataset):
    """Holds list of (input_ids, labels) tensors, pre-tokenized."""
    def __init__(self, examples: list[dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def _pad_collate(pad_id: int):
    def _fn(batch):
        max_len = max(b["input_ids"].numel() for b in batch)
        ids_out, lab_out, mask_out = [], [], []
        for b in batch:
            n = b["input_ids"].numel()
            pad = max_len - n
            ids_out.append(F.pad(b["input_ids"], (0, pad), value=pad_id))
            lab_out.append(F.pad(b["labels"], (0, pad), value=-100))
            mask_out.append(F.pad(torch.ones(n, dtype=torch.long), (0, pad), value=0))
        return {
            "input_ids": torch.stack(ids_out),
            "labels": torch.stack(lab_out),
            "attention_mask": torch.stack(mask_out),
        }
    return _fn


def _tokenize_pair(tok, prompt: str, response: str, max_len: int) -> dict[str, torch.Tensor]:
    """Tokenize prompt+response, mask prompt tokens (label=-100), truncate to max_len."""
    pr_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    rs_ids = tok(response, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    if eos is not None:
        rs_ids = rs_ids + [eos]
    # truncate; keep response intact if possible
    ids = pr_ids + rs_ids
    if len(ids) > max_len:
        # truncate from the prompt side first
        overflow = len(ids) - max_len
        if overflow < len(pr_ids):
            pr_ids = pr_ids[overflow:]
        else:
            # extreme: prompt itself too long
            pr_ids = []
            rs_ids = rs_ids[-max_len:]
        ids = pr_ids + rs_ids
    labels = [-100] * len(pr_ids) + rs_ids
    assert len(ids) == len(labels)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_gsm8k(tok, max_len: int, log, val_size: int = 250) -> tuple[SFTDataset, SFTDataset]:
    from datasets import load_dataset
    log.info("loading gsm8k (config=main)")
    ds = load_dataset("gsm8k", "main")
    train_raw = list(ds["train"])
    test_raw = list(ds["test"])
    rng = random.Random(42)
    rng.shuffle(train_raw)
    val_split = train_raw[:val_size]
    train_split = train_raw[val_size:]
    log.info(f"gsm8k train={len(train_split)} val={len(val_split)} test={len(test_raw)}")

    def _fmt(ex):
        prompt = f"Question: {ex['question']}\nAnswer:"
        response = " " + ex["answer"]
        return _tokenize_pair(tok, prompt, response, max_len)
    train_ex = [_fmt(e) for e in train_split]
    val_ex = [_fmt(e) for e in val_split]
    log.info(f"gsm8k tokenized: train {len(train_ex)} val {len(val_ex)}")
    return SFTDataset(train_ex), SFTDataset(val_ex)


def build_alpaca(tok, max_len: int, log, n_train: int = 10_000, n_val: int = 500) -> tuple[SFTDataset, SFTDataset]:
    from datasets import load_dataset
    log.info("loading yahma/alpaca-cleaned")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    raw = list(ds)
    rng = random.Random(42)
    rng.shuffle(raw)
    val_split = raw[:n_val]
    train_split = raw[n_val : n_val + n_train]
    log.info(f"alpaca train={len(train_split)} val={len(val_split)}")

    def _fmt(ex):
        instr = ex["instruction"]
        inp = ex.get("input", "") or ""
        if inp.strip():
            prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
        response = ex["output"]
        return _tokenize_pair(tok, prompt, response, max_len)
    train_ex = [_fmt(e) for e in train_split]
    val_ex = [_fmt(e) for e in val_split]
    log.info(f"alpaca tokenized: train {len(train_ex)} val {len(val_ex)}")
    return SFTDataset(train_ex), SFTDataset(val_ex)


def build_tulu3(tok, max_len: int, log, n_train: int = 10_000, n_val: int = 500) -> tuple[SFTDataset, SFTDataset]:
    """Tulu-3 SFT mixture. Local parquet at LOCAL_TULU3_PATH/data/*.parquet.
    Schema: {'id', 'messages': List[{'role','content'}], 'source'}.
    We use the FIRST user turn + FIRST assistant turn (most samples are single-turn)."""
    import pandas as pd
    import glob
    log.info(f"loading tulu-3-sft-mixture from {LOCAL_TULU3_PATH}")
    files = sorted(glob.glob(f"{LOCAL_TULU3_PATH}/data/train-*.parquet"))
    # only load enough for n_train+n_val with a buffer; each shard ~26k samples
    needed_shards = max(1, (n_train + n_val) // 20000 + 1)
    dfs = [pd.read_parquet(f) for f in files[:needed_shards]]
    df = pd.concat(dfs, ignore_index=True)
    raw = df.to_dict("records")
    rng = random.Random(42)
    rng.shuffle(raw)
    val_split = raw[:n_val]
    train_split = raw[n_val : n_val + n_train]
    log.info(f"tulu3-sft train={len(train_split)} val={len(val_split)}")

    def _fmt(ex):
        msgs = ex["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if user is None or asst is None:
            return None
        prompt = f"### Instruction:\n{user}\n\n### Response:\n"
        response = asst
        return _tokenize_pair(tok, prompt, response, max_len)
    train_ex = [r for r in (_fmt(e) for e in train_split) if r is not None]
    val_ex = [r for r in (_fmt(e) for e in val_split) if r is not None]
    log.info(f"tulu3-sft tokenized: train {len(train_ex)} val {len(val_ex)}")
    return SFTDataset(train_ex), SFTDataset(val_ex)


def build_metamathqa(tok, max_len: int, log, n_train: int = 10_000, n_val: int = 500) -> tuple[SFTDataset, SFTDataset]:
    """MetaMathQA-395K local JSON. Schema: {'query','response','type','original_question'}."""
    log.info(f"loading MetaMathQA from {LOCAL_METAMATH_PATH}")
    with open(f"{LOCAL_METAMATH_PATH}/MetaMathQA-395K.json") as f:
        raw = json.load(f)
    rng = random.Random(42)
    rng.shuffle(raw)
    val_split = raw[:n_val]
    train_split = raw[n_val : n_val + n_train]
    log.info(f"metamathqa train={len(train_split)} val={len(val_split)}")

    def _fmt(ex):
        prompt = f"Question: {ex['query']}\nAnswer:"
        response = " " + ex["response"]
        return _tokenize_pair(tok, prompt, response, max_len)
    train_ex = [_fmt(e) for e in train_split]
    val_ex = [_fmt(e) for e in val_split]
    log.info(f"metamathqa tokenized: train {len(train_ex)} val {len(val_ex)}")
    return SFTDataset(train_ex), SFTDataset(val_ex)


# -----------------------------------------------------------------------------
# LoRA helpers (reuse stage2 conventions)
# -----------------------------------------------------------------------------
def wrap_lora(model: nn.Module, r: int, alpha: int, dropout: float,
              target_modules: list[str], method: str = "lora_vanilla",
              total_steps: int = 3000) -> nn.Module:
    if method == "dora":
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=target_modules, bias="none",
                         use_dora=True)
    elif method == "adalora":
        if not _HAS_ADALORA:
            raise RuntimeError("peft AdaLoraConfig not available")
        cfg = AdaLoraConfig(
            init_r=r * 2,
            target_r=r,
            beta1=0.85, beta2=0.85,
            tinit=200,
            tfinal=total_steps - 500,
            deltaT=10,
            lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none",
            total_step=total_steps,
        )
    else:
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=target_modules, bias="none")
    return get_peft_model(model, cfg)


def _find_lora_owner(peft_model: nn.Module, handle_name: str):
    """handle.name = '<module path>.<adapter_key>' -> walk to module."""
    parts = handle_name.rsplit(".", 1)[0]
    mod = peft_model
    for p in parts.split("."):
        if p == "":
            continue
        mod = getattr(mod, p)
    return mod


@torch.no_grad()
def merge_and_reset_lora(peft_model: nn.Module, handles: list[LoraHandle],
                        keep_mask: dict[str, torch.Tensor], log) -> dict[str, Any]:
    merged_total = 0
    kept_per_layer: dict[str, int] = {}
    for h in handles:
        mask = keep_mask[h.name].to(h.A.device)
        owner = _find_lora_owner(peft_model, h.name)
        base_linear = owner.base_layer
        if mask.any():
            r_keep = int(mask.sum().item())
            B_kept = h.B[:, mask].to(torch.float32)
            A_kept = h.A[mask, :].to(torch.float32)
            delta = (B_kept @ A_kept) * h.scaling
            base_linear.weight.data.add_(delta.to(base_linear.weight.dtype))
            merged_total += r_keep
            kept_per_layer[h.name] = r_keep
        else:
            kept_per_layer[h.name] = 0
        nn.init.kaiming_uniform_(h.A, a=math.sqrt(5))
        nn.init.zeros_(h.B)
    return {"merged_total": merged_total, "kept_per_layer": kept_per_layer}


# -----------------------------------------------------------------------------
# Effective rank / condition number — compute on EFFECTIVE weight = base + scaling * B @ A
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_rank_stats(peft_model: nn.Module, sample_layers: int = 8) -> dict[str, Any]:
    """For 7B+ LoRA models there are ~224 layers; SVD on each (4096 x 4096) is expensive.
    We sample a fixed subset stratified by module-type, log mean over the sample.

    sample_layers: how many handles to SVD per call. Default 8 = stratified by module type
    if 7 module types are LoRA'd, we take the FIRST layer of each + one extra random.
    Stratified sampling keeps timing constant + comparable across runs.
    """
    handles = get_lora_BA_handles(peft_model)
    # stratify by module type (q_proj, k_proj, ...)
    by_type: dict[str, list[LoraHandle]] = {}
    for h in handles:
        # name format ends in '...<proj>.default'
        proj = h.name.rsplit(".", 2)[-2]   # e.g. 'q_proj'
        by_type.setdefault(proj, []).append(h)
    picked: list[LoraHandle] = []
    for proj, lst in by_type.items():
        picked.append(lst[0])
        if len(lst) > 1 and len(picked) < sample_layers:
            picked.append(lst[len(lst) // 2])  # mid-layer too
    picked = picked[:sample_layers] if len(picked) > sample_layers else picked

    per_layer: dict[str, dict[str, float]] = {}
    for h in picked:
        owner = _find_lora_owner(peft_model, h.name)
        base_W = owner.base_layer.weight.detach().to(torch.float32)
        delta = (h.B.detach().to(torch.float32) @ h.A.detach().to(torch.float32)) * h.scaling
        W = base_W + delta
        er = float(effective_rank(W))
        cn = float(condition_number(W))
        per_layer[h.name] = {"effective_rank": er, "condition_number": cn}
        del W, base_W, delta
        torch.cuda.empty_cache()

    mean_er = float(np.mean([v["effective_rank"] for v in per_layer.values()]))
    mean_cn = float(np.mean([v["condition_number"] for v in per_layer.values()]))
    return {"per_layer": per_layer, "mean_effective_rank": mean_er,
            "mean_condition_number": mean_cn,
            "sampled_layers": len(per_layer)}


# -----------------------------------------------------------------------------
# Eval (mean loss over response tokens, batched)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_lm(model, loader, device, max_batches: int = 200) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    n_batches = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        total += float(out.loss.item())
        n_batches += 1
    if was_training:
        model.train()
    return total / max(n_batches, 1)


# -----------------------------------------------------------------------------
# Gate predicate
# -----------------------------------------------------------------------------
def build_keep_mask(handles: list[LoraHandle], gate_sign: str,
                    fo_val_signed: dict[str, torch.Tensor],
                    target_drop_rate: float | None = None,
                    rng_seed: int | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """gate_sign='S3pos_drops' -> keep if s<0 (drop if s>0).
       gate_sign='S3neg_drops' -> keep if s>0 (drop if s<0).
       gate_sign='S2train_pos_drops' -> same as S3pos_drops but uses train-gradient saliency.
       gate_sign='random' -> drop uniformly Bernoulli with prob `target_drop_rate`
                              (or 0.5 if None).
    """
    masks = {}
    all_scores = []
    total = 0
    kept = 0
    per_layer_keep = {}
    if gate_sign == "random":
        gen = torch.Generator()
        if rng_seed is not None:
            gen.manual_seed(rng_seed)
        drop_p = target_drop_rate if target_drop_rate is not None else 0.5
        for h in handles:
            r = h.r
            # Bernoulli keep mask
            m = torch.rand(r, generator=gen) >= drop_p
            masks[h.name] = m
            n_kept = int(m.sum().item())
            per_layer_keep[h.name] = n_kept
            total += r
            kept += n_kept
        qs = []
    else:
        for h in handles:
            s = fo_val_signed[h.name]
            if gate_sign in ("S3pos_drops", "S2train_pos_drops"):
                m = s < 0.0
            elif gate_sign == "S3neg_drops":
                m = s > 0.0
            else:
                raise ValueError(gate_sign)
            masks[h.name] = m
            all_scores.extend([float(v) for v in s.tolist()])
            n_kept = int(m.sum().item())
            per_layer_keep[h.name] = n_kept
            total += h.r
            kept += n_kept
        qs = [float(np.quantile(all_scores, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)] if all_scores else []
    return masks, {
        "components_total": total, "components_kept": kept,
        "components_dropped": total - kept,
        "drop_rate": 1.0 - kept / max(total, 1),
        "score_quantiles": qs,
        "per_layer_keep_counts": per_layer_keep,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="local path or HF id")
    p.add_argument("--model_key", required=True, help="short name for output dir")
    p.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    p.add_argument("--method", choices=METHOD_CHOICES, required=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=TARGET_MODULES_DEFAULT)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--merge_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--rank_stat_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diag_batches", type=int, default=8,
                   help="number of val batches to estimate val grad saliency at merge")
    p.add_argument("--alpaca_n_train", type=int, default=10_000)
    p.add_argument("--alpaca_n_val", type=int, default=500)
    p.add_argument("--gsm8k_n_val", type=int, default=250)
    p.add_argument("--abort_factor", type=float, default=1.5,
                   help="abort if post-merge val_loss > first_eval_val_loss * abort_factor")
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--save_adapter", action="store_true",
                   help="save peft adapter to out_root/adapter/ after training")
    p.add_argument("--ckpt_every", type=int, default=50,
                   help="save adapter checkpoint every N steps (0 = disabled)")
    p.add_argument("--smoke", action="store_true",
                   help="quick smoke: total_steps=50 eval_every=25 rank_stat_every=25")
    args = p.parse_args()

    if args.smoke:
        args.total_steps = 50
        args.eval_every = 25
        args.rank_stat_every = 25
        args.merge_every = 25
        args.log_every = 5
        args.warmup_steps = 5
        args.diag_batches = 2
        args.gsm8k_n_val = 64
        args.alpaca_n_val = 64
        args.alpaca_n_train = 500
        args.ckpt_every = 0

    # Method -> gate
    saliency_source = "val"   # default for S3pos/S3neg
    if args.method == "lora_vanilla":
        do_relora = False
        gate_sign = None
    elif args.method == "relora_baseline":
        do_relora = True
        gate_sign = None      # merge ALL
    elif args.method == "relora_diag_gated_S3pos":
        do_relora = True
        gate_sign = "S3pos_drops"
    elif args.method == "relora_diag_gated_S3neg":
        do_relora = True
        gate_sign = "S3neg_drops"
    elif args.method == "relora_random_drop":
        do_relora = True
        gate_sign = "random"
    elif args.method == "relora_train_gated":
        do_relora = True
        gate_sign = "S2train_pos_drops"
        saliency_source = "train"
    elif args.method == "dora":
        do_relora = False    # DoRA = no ReLoRA merges
        gate_sign = None
    elif args.method == "adalora":
        do_relora = False    # AdaLoRA has its own importance-driven rank reduction
        gate_sign = None
    else:
        raise ValueError(args.method)

    # Output dir
    out_root = Path(args.out_root) if args.out_root else (
        ROOT / "results" / "stage3" / args.model_key / args.dataset / args.method
    )
    out_root.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"stage3.{args.model_key}.{args.dataset}.{args.method}",
                     str(out_root / "run.log"))
    log.info(f"START stage3 model={args.model_path} dataset={args.dataset} "
             f"method={args.method} out={out_root}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log.info(f"tokenizer loaded in {time.time()-t0:.1f}s, vocab={tok.vocab_size}")

    # Data
    t0 = time.time()
    if args.dataset == "gsm8k":
        train_ds, val_ds = build_gsm8k(tok, args.seq_len, log, val_size=args.gsm8k_n_val)
    elif args.dataset == "alpaca":
        train_ds, val_ds = build_alpaca(tok, args.seq_len, log,
                                         n_train=args.alpaca_n_train, n_val=args.alpaca_n_val)
    elif args.dataset == "tulu3-sft":
        train_ds, val_ds = build_tulu3(tok, args.seq_len, log,
                                        n_train=args.alpaca_n_train, n_val=args.alpaca_n_val)
    elif args.dataset == "metamathqa-10k":
        train_ds, val_ds = build_metamathqa(tok, args.seq_len, log,
                                             n_train=args.alpaca_n_train, n_val=args.alpaca_n_val)
    else:
        raise ValueError(args.dataset)
    log.info(f"data prepared in {time.time()-t0:.1f}s")
    collate = _pad_collate(tok.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, drop_last=False)
    # diagnostic loader = same val data, shuffled
    diag_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate, drop_last=False)

    # Model
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    log.info(f"model loaded in {time.time()-t0:.1f}s")
    base.gradient_checkpointing_enable()
    model = wrap_lora(base, r=args.lora_r, alpha=args.lora_alpha,
                      dropout=args.lora_dropout, target_modules=args.target_modules,
                      method=args.method, total_steps=args.total_steps)
    model.enable_input_require_grads()
    model = model.to(device)
    handles = get_lora_BA_handles(model)
    n_components = count_lora_components(handles)
    log.info(f"#LoRA layers={len(handles)} #components={n_components}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"#trainable params={n_trainable/1e6:.2f}M")

    # Optim + sched
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps
    )

    # Persist config
    import subprocess
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit_hash = "unknown"
    wall_clock_start = time.time()
    dump_yaml({
        "model_path": args.model_path, "model_key": args.model_key,
        "dataset": args.dataset, "method": args.method,
        "gate_sign": gate_sign, "do_relora": do_relora,
        "saliency_source": saliency_source,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "target_modules": args.target_modules,
        "seq_len": args.seq_len, "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch": args.batch_size * args.grad_accum_steps,
        "total_steps": args.total_steps, "merge_every": args.merge_every,
        "eval_every": args.eval_every, "rank_stat_every": args.rank_stat_every,
        "lr": args.lr, "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "seed": args.seed, "diag_batches": args.diag_batches,
        "abort_factor": args.abort_factor,
        "n_lora_components": n_components, "n_trainable_M": n_trainable / 1e6,
        "commit_hash": commit_hash,
        "wall_clock_start": wall_clock_start,
    }, str(out_root / "config.yaml"))

    # Empty out output files
    train_loss_path = out_root / "train_loss.jsonl"
    val_loss_path = out_root / "val_loss.jsonl"
    er_path = out_root / "effective_rank.jsonl"
    cn_path = out_root / "condition_number.jsonl"
    merge_path = out_root / "saliency_at_merge.jsonl"
    cumrank_path = out_root / "cumulative_rank.jsonl"
    dropped_path = out_root / "dropped_components.jsonl"
    for f in [train_loss_path, val_loss_path, er_path, cn_path, merge_path,
              cumrank_path, dropped_path]:
        if f.exists():
            f.unlink()

    cumulative_merged_total = 0
    cumulative_dropped_total = 0

    # Baseline rank stats
    rs0 = compute_rank_stats(model)
    append_jsonl(str(er_path), {"step": 0, "mean_effective_rank": rs0["mean_effective_rank"],
                                 "per_layer": {k: v["effective_rank"] for k, v in rs0["per_layer"].items()},
                                 "sampled_layers": rs0["sampled_layers"]})
    append_jsonl(str(cn_path), {"step": 0, "mean_condition_number": rs0["mean_condition_number"],
                                 "per_layer": {k: v["condition_number"] for k, v in rs0["per_layer"].items()},
                                 "sampled_layers": rs0["sampled_layers"]})
    log.info(f"[step 0] mean_ER={rs0['mean_effective_rank']:.2f} "
             f"mean_CN={rs0['mean_condition_number']:.2e} (sampled {rs0['sampled_layers']} layers)")

    # Merge schedule
    if do_relora:
        merge_steps = set(range(args.merge_every, args.total_steps + 1, args.merge_every))
    else:
        merge_steps = set()
    log.info(f"merge events scheduled at: {sorted(merge_steps)}")

    # Train loop
    model.train()
    step = 0
    micro_step = 0
    running = 0.0
    n_run = 0
    t_start = time.time()
    first_eval_val_loss: Optional[float] = None
    best_val_loss: float = float("inf")
    best_step: int = -1
    ckpt_dir = out_root / "checkpoints"
    if args.ckpt_every > 0 or True:
        ckpt_dir.mkdir(exist_ok=True)
    aborted = False

    while step < args.total_steps and not aborted:
        for batch in train_loader:
            if step >= args.total_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum_steps
            loss.backward()
            running += float(out.loss.item())
            n_run += 1
            micro_step += 1

            if micro_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    avg = running / max(n_run, 1)
                    lr_now = sched.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    log.info(f"step={step}/{args.total_steps} train_loss={avg:.4f} "
                             f"lr={lr_now:.2e} elapsed={elapsed:.0f}s")
                    append_jsonl(str(train_loss_path),
                                 {"step": step, "train_loss": avg, "lr": lr_now})
                    running = 0.0; n_run = 0

                if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                    step_dir = ckpt_dir / f"step_{step:06d}"
                    step_dir.mkdir(exist_ok=True)
                    model.save_pretrained(str(step_dir))
                    tok.save_pretrained(str(step_dir))
                    log.info(f"periodic ckpt saved: {step_dir}")

                if step % args.eval_every == 0:
                    vl = evaluate_lm(model, val_loader, device,
                                     max_batches=max(50, len(val_loader)))
                    log.info(f"step={step} VAL_LOSS={vl:.4f}")
                    append_jsonl(str(val_loss_path), {"step": step, "val_loss": vl})
                    if first_eval_val_loss is None:
                        first_eval_val_loss = vl
                    if vl < best_val_loss:
                        best_val_loss = vl
                        best_step = step
                        best_dir = ckpt_dir / "best"
                        best_dir.mkdir(exist_ok=True)
                        model.save_pretrained(str(best_dir))
                        tok.save_pretrained(str(best_dir))
                        write_json(str(best_dir / "meta.json"),
                                   {"step": step, "val_loss": vl})
                        log.info(f"best ckpt updated: step={step} val_loss={vl:.4f}")

                if step % args.rank_stat_every == 0 and step > 0:
                    rs = compute_rank_stats(model)
                    append_jsonl(str(er_path), {"step": step,
                                                 "mean_effective_rank": rs["mean_effective_rank"],
                                                 "per_layer": {k: v["effective_rank"] for k, v in rs["per_layer"].items()},
                                                 "sampled_layers": rs["sampled_layers"]})
                    append_jsonl(str(cn_path), {"step": step,
                                                 "mean_condition_number": rs["mean_condition_number"],
                                                 "per_layer": {k: v["condition_number"] for k, v in rs["per_layer"].items()},
                                                 "sampled_layers": rs["sampled_layers"]})
                    log.info(f"step={step} mean_ER={rs['mean_effective_rank']:.2f} "
                             f"mean_CN={rs['mean_condition_number']:.2e}")

                # ----- ReLoRA merge event -----
                if step in merge_steps:
                    event_idx = sorted(merge_steps).index(step) + 1
                    log.info(f"=== MERGE EVENT {event_idx} at step {step} (method={args.method}) ===")
                    # build keep_mask
                    if gate_sign is None:
                        # vanilla ReLoRA: merge all
                        keep_masks = {h.name: torch.ones(h.r, dtype=torch.bool) for h in handles}
                        stats = {"components_total": n_components,
                                 "components_kept": n_components,
                                 "components_dropped": 0, "drop_rate": 0.0,
                                 "score_quantiles": [],
                                 "per_layer_keep_counts": {h.name: h.r for h in handles}}
                    elif gate_sign == "random":
                        # Bernoulli random drop at fixed rate 0.5 (matches median S3pos drop_rate empirically)
                        keep_masks, stats = build_keep_mask(
                            handles, "random", fo_val_signed={},
                            target_drop_rate=0.5,
                            rng_seed=args.seed + event_idx,
                        )
                    else:
                        # gated: compute first-order saliency on val OR train batch
                        sal_loader = diag_loader if saliency_source == "val" else train_loader
                        fo_signed = first_order_saliency(
                            model, handles, sal_loader, device,
                            max_batches=args.diag_batches, signed=True,
                        )
                        keep_masks, stats = build_keep_mask(handles, gate_sign, fo_signed)
                    merge_stats = merge_and_reset_lora(model, handles, keep_masks, log)
                    # reset optimizer (Lialin protocol)
                    optim = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
                    remaining = args.total_steps - step
                    sched = get_cosine_schedule_with_warmup(
                        optim, num_warmup_steps=min(args.warmup_steps, max(remaining // 4, 1)),
                        num_training_steps=max(remaining, 1)
                    )
                    rec = {"step": step, "merge_event": event_idx,
                           "method": args.method, "gate_sign": gate_sign,
                           **{k: stats[k] for k in stats if k != "per_layer_keep_counts"},
                           "merged_total": merge_stats["merged_total"],
                           "per_layer_keep_counts": stats["per_layer_keep_counts"]}
                    append_jsonl(str(merge_path), rec)
                    log.info(f"merge: total={stats['components_total']} "
                             f"kept={stats['components_kept']} drop_rate={stats['drop_rate']:.3f}")

                    cumulative_merged_total += merge_stats["merged_total"]
                    cumulative_dropped_total += stats["components_dropped"]
                    append_jsonl(str(cumrank_path), {
                        "step": step, "merge_event": event_idx,
                        "cumulative_merged_total": cumulative_merged_total,
                        "cumulative_dropped_total": cumulative_dropped_total,
                        "components_kept_this_event": stats["components_kept"],
                        "components_dropped_this_event": stats["components_dropped"],
                    })
                    append_jsonl(str(dropped_path), {
                        "step": step, "merge_event": event_idx,
                        "drop_rate": stats["drop_rate"],
                        "per_layer_keep_counts": {
                            k: int(v) for k, v in stats["per_layer_keep_counts"].items()
                        },
                        "score_quantiles": stats.get("score_quantiles", []),
                    })

                    # post-merge rank stats
                    rs_post = compute_rank_stats(model)
                    append_jsonl(str(er_path), {"step": step,
                                                 "mean_effective_rank": rs_post["mean_effective_rank"],
                                                 "per_layer": {k: v["effective_rank"] for k, v in rs_post["per_layer"].items()},
                                                 "post_merge": True,
                                                 "sampled_layers": rs_post["sampled_layers"]})
                    append_jsonl(str(cn_path), {"step": step,
                                                 "mean_condition_number": rs_post["mean_condition_number"],
                                                 "per_layer": {k: v["condition_number"] for k, v in rs_post["per_layer"].items()},
                                                 "post_merge": True,
                                                 "sampled_layers": rs_post["sampled_layers"]})

                    # post-merge val_loss
                    vl_post = evaluate_lm(model, val_loader, device,
                                          max_batches=max(50, len(val_loader)))
                    append_jsonl(str(val_loss_path), {"step": step, "val_loss": vl_post,
                                                       "post_merge": True})
                    log.info(f"step={step} POST-MERGE VAL_LOSS={vl_post:.4f} "
                             f"(first_eval={first_eval_val_loss})")
                    if vl_post < best_val_loss:
                        best_val_loss = vl_post
                        best_step = step
                        best_dir = ckpt_dir / "best"
                        best_dir.mkdir(exist_ok=True)
                        model.save_pretrained(str(best_dir))
                        tok.save_pretrained(str(best_dir))
                        write_json(str(best_dir / "meta.json"),
                                   {"step": step, "val_loss": vl_post, "post_merge": True})
                        log.info(f"best ckpt updated (post-merge): step={step} val_loss={vl_post:.4f}")

                    # red-line abort
                    if first_eval_val_loss is not None and vl_post > first_eval_val_loss * args.abort_factor:
                        log.warning(f"ABORT: post-merge val_loss={vl_post:.4f} > "
                                    f"first_eval={first_eval_val_loss:.4f} * {args.abort_factor}")
                        with open(out_root / "ABORTED.flag", "w") as f:
                            f.write(f"step={step} val_loss={vl_post:.4f} "
                                    f"first_eval={first_eval_val_loss:.4f} "
                                    f"factor={args.abort_factor}\n")
                        write_json(str(out_root / "summary.json"), {
                            "model_key": args.model_key, "dataset": args.dataset,
                            "method": args.method, "aborted": True,
                            "abort_step": step, "abort_val_loss": vl_post,
                            "first_eval_val_loss": first_eval_val_loss,
                            "elapsed_sec": time.time() - t_start,
                        })
                        aborted = True
                        break

                    model.train()

    elapsed = time.time() - t_start
    log.info(f"training {'ABORTED' if aborted else 'done'} in {elapsed:.1f}s")

    # final val + rank
    vl_final = evaluate_lm(model, val_loader, device, max_batches=max(50, len(val_loader)))
    log.info(f"FINAL VAL_LOSS={vl_final:.4f}")
    append_jsonl(str(val_loss_path), {"step": args.total_steps, "val_loss": vl_final, "final": True})

    rs_final = compute_rank_stats(model)
    write_json(str(out_root / "summary.json"), {
        "model_key": args.model_key, "model_path": args.model_path,
        "dataset": args.dataset, "method": args.method, "gate_sign": gate_sign,
        "total_steps": args.total_steps, "merge_every": args.merge_every,
        "final_val_loss": vl_final,
        "first_eval_val_loss": first_eval_val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "final_mean_effective_rank": rs_final["mean_effective_rank"],
        "final_mean_condition_number": rs_final["mean_condition_number"],
        "sampled_layers": rs_final["sampled_layers"],
        "elapsed_sec": elapsed,
        "wall_clock_start": wall_clock_start,
        "wall_clock_end": time.time(),
        "commit_hash": commit_hash,
        "cumulative_merged_total": cumulative_merged_total,
        "cumulative_dropped_total": cumulative_dropped_total,
        "n_trainable_M": n_trainable / 1e6,
        "n_lora_components": n_components,
        "aborted": aborted,
    })
    log.info("summary.json written")

    if args.save_adapter and not aborted:
        adapter_dir = out_root / "adapter"
        adapter_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tok.save_pretrained(str(adapter_dir))
        log.info(f"adapter saved to {adapter_dir}")

    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
