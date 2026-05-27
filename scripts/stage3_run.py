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
import gc
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

# Local Muon optimizer (vendored from KellerJordan/Muon, MIT).
import sys as _sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)
from muon import Muon, split_params_for_muon  # noqa: E402


class OptimizerEnsemble(torch.optim.Optimizer):
    """Combine multiple optimizers behind a single Optimizer interface.

    HF's `get_cosine_schedule_with_warmup` walks `optimizer.param_groups`
    and assigns `group["lr"]` per step. By exposing the union of all child
    optimizers' param_groups, the scheduler updates LR for every group of
    every child uniformly.

    `.step()` / `.zero_grad()` / `.state_dict()` are forwarded.
    """

    def __init__(self, children: list[torch.optim.Optimizer]):
        assert len(children) > 0
        self._children = children
        # Don't call super().__init__ (it expects params/defaults).
        self.defaults = children[0].defaults

    @property
    def param_groups(self):  # noqa: D401
        groups = []
        for opt in self._children:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        # HF schedulers assign back via index; instead they only mutate items in
        # the list returned above (in-place). So we don't need to handle reassign.
        # But satisfy any setter callers by re-distributing.
        idx = 0
        for opt in self._children:
            n = len(opt.param_groups)
            opt.param_groups = value[idx:idx + n]
            idx += n

    @property
    def state(self):  # union view (read-only-ish)
        merged: dict = {}
        for opt in self._children:
            merged.update(opt.state)
        return merged

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self._children:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._children:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self._children)}

    def load_state_dict(self, sd):
        for i, opt in enumerate(self._children):
            opt.load_state_dict(sd[f"opt_{i}"])


def build_optimizer(model: nn.Module, args, log) -> torch.optim.Optimizer:
    """Build optimizer per --optimizer choice.

    adamw: single AdamW over all trainable params (legacy behaviour).
    muon : Muon for 2D LoRA matrices (lora_A, lora_B) + AdamW for the rest
           (1D biases, scaling, embeddings, head).
    """
    if args.optimizer == "adamw":
        trainable = [p for p in model.parameters() if p.requires_grad]
        return AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                     betas=(0.9, 0.95))
    elif args.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model.named_parameters())
        n_muon = sum(p.numel() for p in muon_p)
        n_adamw = sum(p.numel() for p in adamw_p)
        log.info(f"[muon] partitioned trainable params: muon={n_muon/1e6:.2f}M (2D LoRA) "
                 f"adamw={n_adamw/1e6:.2f}M (rest)")
        if not muon_p:
            log.warning("[muon] no 2D LoRA params found; falling back to AdamW only.")
            return AdamW(adamw_p, lr=args.lr, weight_decay=args.weight_decay,
                         betas=(0.9, 0.95))
        children = [
            Muon(muon_p, lr=args.muon_lr, momentum=0.95, nesterov=True,
                 ns_steps=args.muon_ns_steps, weight_decay=args.weight_decay),
        ]
        if adamw_p:
            children.append(AdamW(adamw_p, lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95)))
        return OptimizerEnsemble(children)
    else:
        raise ValueError(f"unknown --optimizer={args.optimizer}")

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
    "relora_diag_gated_S3pos_keepB",
    "dora",
    "adalora",
    "relora_random_drop",
    "relora_train_gated",
    "cola",
]
DATASET_CHOICES = ["gsm8k", "alpaca", "tulu3-sft", "metamathqa-10k"]
LOCAL_TULU3_PATH = "/mnt/cpfs/junlongke/onlinelora/datasets/tulu-3-sft-mixture"
LOCAL_METAMATH_PATH = "/mnt/cpfs/junlongke/onlinelora/datasets/MetaMathQA"
TARGET_MODULES_DEFAULT = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Per-event drop_rate schedules (PI 2026-05-26 v2 saliency revamp, S2.5).
# Each schedule lists drop probabilities per merge event, in event order.
# Default has 6 entries (matches merge_every=500, total_steps=3000).
DROP_SCHEDULE_REGISTRY: dict[str, list[float]] = {
    "const_0p25":         [0.25] * 6,
    "const_0p5":          [0.5] * 6,
    "const_0p75":         [0.75] * 6,
    "anneal_down":        [0.75, 0.65, 0.55, 0.45, 0.35, 0.25],
    "anneal_up":          [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    "triangle_up_down":   [0.25, 0.45, 0.65, 0.65, 0.45, 0.25],
    "triangle_down_up":   [0.75, 0.55, 0.35, 0.35, 0.55, 0.75],
    "early_burst":        [0.9, 0.5, 0.5, 0.5, 0.5, 0.5],
    "late_burst":         [0.5, 0.5, 0.5, 0.5, 0.5, 0.9],
    "bookend_burst":      [0.9, 0.3, 0.3, 0.3, 0.3, 0.9],
    "extreme_alternate":  [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
}


def parse_drop_schedule(spec: str, n_events: int) -> list[float] | None:
    """Resolve --drop_schedule spec to a per-event list of length n_events.

    Returns None if spec is empty (caller should fall back to constant
    --random_drop_rate). Raises ValueError on unknown registry name.
    """
    if not spec:
        return None
    if spec in DROP_SCHEDULE_REGISTRY:
        sched = list(DROP_SCHEDULE_REGISTRY[spec])
    elif spec.startswith("random_schedule:seed="):
        seed = int(spec.split("=", 1)[1])
        rng = np.random.default_rng(seed)
        sched = [float(rng.uniform(0.1, 0.9)) for _ in range(n_events)]
    elif "," in spec:
        sched = [float(x) for x in spec.split(",")]
    else:
        raise ValueError(
            f"unknown --drop_schedule '{spec}'. Use a registry name "
            f"({sorted(DROP_SCHEDULE_REGISTRY)}), a comma list, or "
            f"'random_schedule:seed=N'.")
    if len(sched) < n_events:
        sched = sched + [sched[-1]] * (n_events - len(sched))
    return sched[:n_events]


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
# OOD calibration sets for saliency (Task 3 fix)
# -----------------------------------------------------------------------------
# Default saliency_source='val' uses the SFT val split, which is the WRONG
# distribution when downstream eval is OOD (e.g. GSM8K reasoning, HellaSwag MCQ).
# These calib loaders provide eval-distribution-aligned saliency signals.
def build_gsm8k_calib(tok, max_len: int, log, n_calib: int = 256) -> SFTDataset:
    """GSM8K train split for saliency calibration (OOD math reasoning)."""
    from datasets import load_dataset
    log.info(f"[saliency_calib] loading gsm8k train split (n={n_calib})")
    ds = load_dataset("gsm8k", "main")
    train_raw = list(ds["train"])
    rng = random.Random(42)
    rng.shuffle(train_raw)
    samples = train_raw[:n_calib]

    def _fmt(ex):
        prompt = f"Question: {ex['question']}\nAnswer:"
        response = " " + ex["answer"]
        return _tokenize_pair(tok, prompt, response, max_len)
    ex_list = [_fmt(e) for e in samples]
    log.info(f"[saliency_calib] gsm8k tokenized: {len(ex_list)}")
    return SFTDataset(ex_list)


def build_hellaswag_calib(tok, max_len: int, log, n_calib: int = 256) -> SFTDataset:
    """HellaSwag val split for saliency calibration (OOD commonsense MCQ).

    Treated as completion: prompt = ctx_a + ctx_b, response = endings[label].
    """
    from datasets import load_dataset
    log.info(f"[saliency_calib] loading hellaswag val split (n={n_calib})")
    ds = load_dataset("hellaswag")
    val_raw = list(ds["validation"])
    rng = random.Random(42)
    rng.shuffle(val_raw)
    samples = val_raw[:n_calib]

    def _fmt(ex):
        ctx = (ex.get("ctx_a", "") or "") + " " + (ex.get("ctx_b", "") or "")
        ctx = ctx.strip()
        try:
            label = int(ex["label"])
        except (ValueError, TypeError):
            return None
        endings = ex.get("endings") or []
        if label < 0 or label >= len(endings):
            return None
        prompt = f"{ctx}"
        response = " " + endings[label]
        return _tokenize_pair(tok, prompt, response, max_len)
    ex_list = [r for r in (_fmt(e) for e in samples) if r is not None]
    log.info(f"[saliency_calib] hellaswag tokenized: {len(ex_list)}")
    return SFTDataset(ex_list)


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
                        keep_mask: dict[str, torch.Tensor], log,
                        keep_B_after_merge: bool = False) -> dict[str, Any]:
    """Merge kept components into base weight, then re-init for next ReLoRA segment.

    keep_B_after_merge=False (default, original ReLoRA/CoLA behaviour):
        all components -> kaiming(A), zero(B). Identical reset for kept+dropped;
        saliency only affects which delta gets folded into base weight.

    keep_B_after_merge=True (Task 2 fix; saliency-aware re-init):
        - kept components: keep B columns, set A rows to 0  (delta = B @ 0 = 0,
          no double-count after fold-in; B preserves saliency-selected direction
          for the next segment, A re-learns from zero).
        - dropped components: standard kaiming(A) + zero(B) (full re-init).
    """
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
        if keep_B_after_merge and mask.any():
            # Saliency-aware re-init: keep B[:, kept], zero corresponding A[kept, :].
            # Dropped columns (~mask): kaiming(A), zero(B) as usual.
            drop = ~mask
            # Reset DROPPED rows of A with kaiming, keep KEPT rows' B columns.
            if drop.any():
                A_tmp = torch.empty_like(h.A)
                nn.init.kaiming_uniform_(A_tmp, a=math.sqrt(5))
                h.A.data[drop, :] = A_tmp[drop, :]
                h.B.data[:, drop] = 0.0
            # Kept rows of A reset to 0 (so delta=B@A=0 right after merge);
            # B kept columns preserved.
            h.A.data[mask, :] = 0.0
        else:
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


def build_truncated_loader(loader, max_len: int, max_batches: int):
    """F1 fix for relora_train_gated OOM.

    Returns a list of dicts (compatible with `for batch in loader`) where every
    2-D tensor is truncated to `max_len` along the sequence axis. Materializes
    only the first `max_batches` batches to avoid memory blowup.
    """
    truncated = []
    for batch in loader:
        if len(truncated) >= max_batches:
            break
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2:
                new_batch[k] = v[:, :max_len].contiguous()
            else:
                new_batch[k] = v
        truncated.append(new_batch)
    return truncated


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="local path or HF id")
    p.add_argument("--model_key", required=True, help="short name for output dir")
    p.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    p.add_argument("--method", choices=METHOD_CHOICES, required=True)
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="HF attn_implementation. Use 'eager' for Gemma-3 (logit softcapping).")
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
                   help="(legacy) save peft adapter to out_root/adapter/ after training. "
                        "DEPRECATED for do_relora methods: the saved adapter has lora_B=0 "
                        "after final merge -> lm-eval reflects vanilla base. Use "
                        "--save_merged_final instead, which saves the full merged base.")
    p.add_argument("--save_merged_final", action="store_true",
                   help="(PI #5b Option 3) save the full base model with all merge deltas "
                        "folded in to out_root/merged_final/. This is the correct ckpt for "
                        "lm-eval on do_relora runs. ~16GB per cell for qwen3-8b.")
    p.add_argument("--ckpt_every", type=int, default=50,
                   help="save adapter checkpoint every N steps (0 = disabled)")
    p.add_argument("--saliency_max_seq_len", type=int, default=2048,
                   help="truncate batches to this length when computing saliency on long-seq train batches (F1 fix for relora_train_gated OOM); only triggers if < args.seq_len")
    p.add_argument("--keep_B_after_merge", action="store_true",
                   help="Task 2 method fix: saliency-aware re-init after merge. "
                        "Kept components keep B columns (A rows zeroed); dropped components "
                        "get standard kaiming(A)+zero(B). Auto-enabled by method "
                        "relora_diag_gated_S3pos_keepB.")
    p.add_argument("--saliency_calib_set",
                   choices=["none", "gsm8k_train", "hellaswag_val"],
                   default="none",
                   help="Task 3 fix: OOD calibration set for saliency. 'none' (default) "
                        "keeps the original behaviour (val/train SFT split per method). "
                        "'gsm8k_train' / 'hellaswag_val' override sal_loader with the "
                        "eval-distribution-aligned set so saliency ranks components under "
                        "the downstream OOD distribution.")
    p.add_argument("--saliency_calib_n", type=int, default=256,
                   help="number of calib samples (only when --saliency_calib_set != none)")
    p.add_argument("--random_drop_rate", type=float, default=0.5,
                   help="Bernoulli drop probability for method=relora_random_drop. "
                        "Default 0.5 preserves prior behaviour. Used by Exp-1 drop-rate sweep "
                        "{0.0, 0.1, 0.25, 0.5, 0.75, 0.9}. drop_rate=0.0 should reproduce relora_baseline.")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="Optimizer choice. 'muon' routes 2D LoRA matrices to Muon "
                        "(orthogonalized momentum via Newton-Schulz) and the rest to AdamW. "
                        "Used by Exp-2 Muon-decoupling experiment.")
    p.add_argument("--muon_lr", type=float, default=0.005,
                   help="Learning rate for Muon child (only used when --optimizer=muon). "
                        "Default 0.005; smoke confirmed Keller's default 0.02 diverges on "
                        "LoRA scaled (alpha/r) updates with cosine warmup peak.")
    p.add_argument("--muon_ns_steps", type=int, default=5,
                   help="Newton-Schulz iterations per Muon step (default 5).")
    p.add_argument("--saliency_estimator", choices=["v1", "v2"], default="v1",
                   help="v1: legacy first_order_saliency aggregated over batches "
                        "(sign-only). v2: per-sample IG (m points B->t*B) + "
                        "BH-FDR t-stat gating + Bernoulli random fallback "
                        "(PI 2026-05-26 v2 saliency revamp).")
    p.add_argument("--saliency_v2_m_ig", type=int, default=4,
                   help="IG interpolation points for v2 estimator (default 4).")
    p.add_argument("--saliency_v2_alpha", type=float, default=0.2,
                   help="BH-FDR significance level for v2 t-stat gating "
                        "(default 0.2 per PI feedback 2026-05-26: ~24k tests "
                        "across 6 events x 4032 components, alpha=0.1 too tight).")
    p.add_argument("--drop_schedule", default="",
                   help="Per-event drop_rate schedule. One of:\n"
                        "  - registry name: const_0p5, const_0p25, const_0p75,\n"
                        "      anneal_down, anneal_up, triangle_up_down, triangle_down_up,\n"
                        "      early_burst, late_burst, bookend_burst, extreme_alternate\n"
                        "  - comma list: '0.9,0.5,0.5,0.5,0.5,0.5'\n"
                        "  - 'random_schedule:seed=N' for random per-event in [0.1,0.9]\n"
                        "  - empty (default): use --random_drop_rate as constant")
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
    elif args.method == "relora_diag_gated_S3pos_keepB":
        # Same gating as S3pos, but on merge: kept components keep B, A reset to 0
        # (avoids double-count of B@A through delta after fold-in, while preserving
        #  saliency-selected B direction). Dropped components: standard kaiming(A)+zero(B).
        do_relora = True
        gate_sign = "S3pos_drops"
        args.keep_B_after_merge = True
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
    elif args.method == "cola":
        # COLA (Chain-of-LoRA, Xia et al. 2024 arxiv:2401.04151).
        # Functionally: merge-all + LoRA re-init (kaiming A, zero B) + fresh AdamW.
        # All of these are already implemented identically by `relora_baseline`.
        # Recommended schedule: K=4 stages of T_k steps (merge_every = total_steps/4).
        do_relora = True
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

    # Task 3: OOD saliency calibration loader (overrides val/train sal source).
    calib_loader = None
    if args.saliency_calib_set == "gsm8k_train":
        calib_ds = build_gsm8k_calib(tok, args.seq_len, log, n_calib=args.saliency_calib_n)
        calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate, drop_last=False)
    elif args.saliency_calib_set == "hellaswag_val":
        calib_ds = build_hellaswag_calib(tok, args.seq_len, log, n_calib=args.saliency_calib_n)
        calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate, drop_last=False)
    if calib_loader is not None:
        log.info(f"[saliency_calib] using OOD calib set '{args.saliency_calib_set}' "
                 f"(n={args.saliency_calib_n}) for saliency; overrides "
                 f"saliency_source={args.method} default.")

    # Model
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation, low_cpu_mem_usage=True,
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
    optim = build_optimizer(model, args, log)
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
        "saliency_max_seq_len": args.saliency_max_seq_len,
        "keep_B_after_merge": args.keep_B_after_merge,
        "saliency_calib_set": args.saliency_calib_set,
        "saliency_calib_n": args.saliency_calib_n,
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

    # Optimizer metadata (PI 2026-05-26 hard rule for Exp-2).
    opt_meta = {
        "optimizer": args.optimizer,
        "muon_lr": args.muon_lr if args.optimizer == "muon" else None,
        "muon_ns_steps": args.muon_ns_steps if args.optimizer == "muon" else None,
        "adamw_lr": args.lr,
        "weight_decay": args.weight_decay,
        "random_drop_rate": args.random_drop_rate if args.method == "relora_random_drop" else None,
        "betas_adamw": [0.9, 0.95],
    }
    with (out_root / "optimizer_metadata.json").open("w") as f:
        json.dump(opt_meta, f, indent=2)

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

    # Resolve --drop_schedule (PI 2026-05-26 v2, S2.5). None => constant --random_drop_rate.
    drop_schedule_list = parse_drop_schedule(args.drop_schedule, len(merge_steps))
    if drop_schedule_list is not None:
        log.info(f"drop_schedule '{args.drop_schedule}' -> per-event rates: {drop_schedule_list}")

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
                    log.info(f"=== MERGE EVENT {event_idx} at step {step} (method={args.method}, keep_B_after_merge={args.keep_B_after_merge}) ===")
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
                        # Bernoulli random drop. Per-event rate from --drop_schedule
                        # (PI 2026-05-26 v2 S2.5) when set; otherwise constant
                        # --random_drop_rate (Exp-1 sweep target).
                        if drop_schedule_list is not None:
                            rate_for_event = float(drop_schedule_list[event_idx - 1])
                            sched_name = args.drop_schedule or "constant"
                            n_events = len(drop_schedule_list)
                        else:
                            rate_for_event = float(args.random_drop_rate)
                            sched_name = "constant"
                            n_events = len(merge_steps)
                        log.info(f"[schedule={sched_name} event_idx={event_idx}/{n_events} "
                                 f"target_drop_rate={rate_for_event:.3f}]")
                        keep_masks, stats = build_keep_mask(
                            handles, "random", fo_val_signed={},
                            target_drop_rate=rate_for_event,
                            rng_seed=args.seed + event_idx,
                        )
                        stats["scheduled_drop_rate"] = rate_for_event
                        stats["schedule_name"] = sched_name
                        stats["event_idx"] = event_idx
                        stats["n_events"] = n_events
                        log.info(f"[schedule={sched_name} event_idx={event_idx}/{n_events} "
                                 f"realised_drop_rate={stats['drop_rate']:.4f} "
                                 f"target={rate_for_event:.4f} "
                                 f"diff={stats['drop_rate']-rate_for_event:+.4f}]")
                    else:
                        # gated: compute first-order saliency on val/train OR OOD calib batch.
                        # Task 3 fix: if --saliency_calib_set is set, override sal source.
                        if calib_loader is not None:
                            sal_loader = calib_loader
                            sal_src_label = f"calib:{args.saliency_calib_set}"
                        else:
                            sal_loader = diag_loader if saliency_source == "val" else train_loader
                            sal_src_label = saliency_source
                        # F2 OOM fix (S3pos large-model crash):
                        # Always truncate sal_loader to saliency_max_seq_len for both
                        # val and train sources. Previously only train was truncated;
                        # for large models (qwen35-4b/9b, qwen3-14b) the un-truncated
                        # val loader caused silent CUDA OOM (SIGKILL, no traceback)
                        # at MERGE EVENT 1 with method=relora_diag_gated_S3pos.
                        sal_loader = build_truncated_loader(
                            sal_loader,
                            max_len=min(args.saliency_max_seq_len, args.seq_len),
                            max_batches=args.diag_batches,
                        )
                        log.info(f"saliency loader (source={sal_src_label}) truncated to "
                                 f"seq_len={min(args.saliency_max_seq_len, args.seq_len)} "
                                 f"(diag_batches={args.diag_batches}) [F2 OOM fix]")
                        # Free optimizer state + cache before backward to give
                        # saliency activations more headroom on big models.
                        del optim, sched
                        gc.collect()
                        torch.cuda.empty_cache()
                        if args.saliency_estimator == "v2":
                            # PI 2026-05-26 v2: per-sample IG + BH-FDR t-stat gating.
                            from src.saliency_v2 import (
                                integrated_gradient_saliency_per_sample,
                                t_stat_decision,
                                fisher_signvote_score,
                            )
                            mem_pre = torch.cuda.memory_allocated() / (1024**3)
                            log.info(f"merge_event{event_idx} pre-saliency cuda_alloc={mem_pre:.2f}GB "
                                     f"[v2 estimator m_ig={args.saliency_v2_m_ig} alpha={args.saliency_v2_alpha}]")
                            n_samples_target = min(args.saliency_calib_n,
                                                   args.diag_batches * args.batch_size)
                            try:
                                per_sample = integrated_gradient_saliency_per_sample(
                                    model, handles, sal_loader, device,
                                    m=args.saliency_v2_m_ig,
                                    max_samples=max(n_samples_target, 16),
                                    signed=True,
                                )
                            except torch.cuda.OutOfMemoryError as e:
                                log.warning(f"v2 saliency OOM ({e}); retry with halved seq_len + samples")
                                torch.cuda.empty_cache()
                                base_sal_loader = (
                                    calib_loader if calib_loader is not None
                                    else (diag_loader if saliency_source == "val" else train_loader)
                                )
                                sal_loader = build_truncated_loader(
                                    base_sal_loader,
                                    max_len=max(args.saliency_max_seq_len // 2, 256),
                                    max_batches=max(args.diag_batches // 2, 1),
                                )
                                per_sample = integrated_gradient_saliency_per_sample(
                                    model, handles, sal_loader, device,
                                    m=args.saliency_v2_m_ig,
                                    max_samples=max(n_samples_target // 2, 8),
                                    signed=True,
                                )
                            keep_masks, v2_info = t_stat_decision(
                                per_sample, alpha=args.saliency_v2_alpha,
                                rng_seed=args.seed + event_idx,
                            )
                            fsv_scores = fisher_signvote_score(per_sample)
                            n_total = sum(h.r for h in handles)
                            n_kept = sum(int(m.sum().item()) for m in keep_masks.values())
                            flat_fsv = []
                            for L in fsv_scores:
                                flat_fsv.extend([float(x) for x in fsv_scores[L].tolist()])
                            qs = ([float(np.quantile(flat_fsv, q)) for q in (0.05,0.25,0.5,0.75,0.95)]
                                  if flat_fsv else [])
                            stats = {
                                "components_total": n_total,
                                "components_kept": n_kept,
                                "components_dropped": n_total - n_kept,
                                "drop_rate": 1.0 - n_kept / max(n_total, 1),
                                "score_quantiles": qs,
                                "per_layer_keep_counts": {L: int(m.sum().item())
                                                          for L, m in keep_masks.items()},
                                "saliency_estimator": "v2",
                                **v2_info,
                            }
                            # PI feedback #2 §1: per-event v2 breakdown.
                            n_random_keep_v2 = v2_info.get("n_random_keep", 0)
                            n_random_drop_v2 = v2_info.get("n_random", 0) - n_random_keep_v2
                            n_keep_sig_v2 = v2_info.get("n_keep_sig", 0)
                            n_drop_sig_v2 = v2_info.get("n_drop_sig", 0)
                            q05_v2 = qs[0] if qs else float("nan")
                            q50_v2 = qs[2] if qs else float("nan")
                            q95_v2 = qs[4] if qs else float("nan")
                            log.info(
                                f"[v2 estimator m_ig={args.saliency_v2_m_ig} "
                                f"alpha={args.saliency_v2_alpha}] "
                                f"merge_event={event_idx}\n"
                                f"  n_keep_sig={n_keep_sig_v2}  n_drop_sig={n_drop_sig_v2}  "
                                f"n_random_assigned_keep={n_random_keep_v2}  "
                                f"n_random_assigned_drop={n_random_drop_v2}\n"
                                f"  -> final keep={n_keep_sig_v2 + n_random_keep_v2}  "
                                f"final drop={n_drop_sig_v2 + n_random_drop_v2}  "
                                f"drop_rate={(n_drop_sig_v2 + n_random_drop_v2)/max(n_total,1):.4f}\n"
                                f"  fisher_signvote_score: q05={q05_v2:.3e}  "
                                f"q50={q50_v2:.3e}  q95={q95_v2:.3e}"
                            )
                            stats["n_random_drop"] = n_random_drop_v2
                            # PI feedback #2 §2: dump dropped (layer, idx) pairs
                            # so v1<->v2 IoU can be computed offline.
                            stats["dropped_component_ids"] = [
                                [L, int(i)] for L, mtensor in keep_masks.items()
                                for i, kept in enumerate(mtensor.tolist()) if not kept
                            ]
                            model.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                        else:
                            try:
                                mem_pre = torch.cuda.memory_allocated() / (1024**3)
                                log.info(f"merge_event{event_idx} pre-saliency cuda_alloc={mem_pre:.2f}GB")
                                fo_signed = first_order_saliency(
                                    model, handles, sal_loader, device,
                                    max_batches=args.diag_batches, signed=True,
                                )
                            except torch.cuda.OutOfMemoryError as e:
                                log.warning(f"saliency OOM ({e}); retry with diag_batches=1, max_len={args.saliency_max_seq_len // 2}")
                                torch.cuda.empty_cache()
                                base_sal_loader = (
                                    calib_loader if calib_loader is not None
                                    else (diag_loader if saliency_source == "val" else train_loader)
                                )
                                sal_loader = build_truncated_loader(
                                    base_sal_loader,
                                    max_len=max(args.saliency_max_seq_len // 2, 256),
                                    max_batches=1,
                                )
                                fo_signed = first_order_saliency(
                                    model, handles, sal_loader, device,
                                    max_batches=1, signed=True,
                                )
                            except Exception as e:
                                import traceback
                                log.error(f"saliency FAILED: {type(e).__name__}: {e}")
                                log.error(traceback.format_exc())
                                raise
                            finally:
                                model.zero_grad(set_to_none=True)
                                torch.cuda.empty_cache()
                            keep_masks, stats = build_keep_mask(handles, gate_sign, fo_signed)
                            stats["saliency_estimator"] = "v1"
                            # PI feedback #2 §2: per-event v1 breakdown for v1<->v2 IoU.
                            stats["dropped_component_ids"] = [
                                [L, int(i)] for L, mtensor in keep_masks.items()
                                for i, kept in enumerate(mtensor.tolist()) if not kept
                            ]
                            log.info(
                                f"[v1 estimator] merge_event={event_idx} "
                                f"n_dropped={stats['components_dropped']} "
                                f"drop_rate={stats['drop_rate']:.4f}"
                            )
                    merge_stats = merge_and_reset_lora(model, handles, keep_masks, log,
                                                       keep_B_after_merge=args.keep_B_after_merge)
                    # reset optimizer (Lialin protocol)
                    optim = build_optimizer(model, args, log)
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
                    # NOTE: do NOT update best ckpt from post-merge val_loss.
                    # Right after a ReLoRA merge, lora_B has just been zeroed
                    # (the trained delta is now baked into the frozen base
                    # weights). Saving model.save_pretrained() at this instant
                    # serializes a PEFT adapter with lora_B=0, which is
                    # numerically identical to the base model -> downstream
                    # lm-eval would reflect base model performance, not the
                    # learned adapter. Best ckpts come exclusively from the
                    # eval_every branch above (training-state evals).

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
        # NOTE: PI #5b — for do_relora methods this saves the pre-merge "best/"
        # ckpt (lora_B != 0 because saved between merges), but pre-merge ckpts
        # are method-blind (drop policy hasn't fired yet at best_step). Use
        # --save_merged_final for the post-all-merges base model which IS the
        # correct lm-eval target.
        _best_dir_final = ckpt_dir / "best"
        if do_relora and _best_dir_final.exists() and (_best_dir_final / "adapter_model.safetensors").exists():
            import shutil
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            shutil.copytree(str(_best_dir_final), str(adapter_dir))
            log.info(f"adapter saved (copied from best/ — METHOD-BLIND for do_relora; "
                     f"use --save_merged_final for method-aware eval) to {adapter_dir}")
        else:
            adapter_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            tok.save_pretrained(str(adapter_dir))
            log.info(f"adapter saved to {adapter_dir}")

    if args.save_merged_final and not aborted:
        # PI #5b Option 3: at end of training, the base_linear weights already
        # contain ALL merge deltas folded in (in-place mutation in
        # merge_and_reset_lora at L517). The PEFT wrapper still wraps the
        # base model with LoRA sub-modules; we need to extract the underlying
        # transformer and save it as a standalone HF model for offline lm-eval.
        #
        # NOTE: we cannot call peft.merge_and_unload() because PEFT's save
        # path triggers transformers.is_deepspeed_zero3_enabled() which on
        # this env tries to compile CUDA ops (CUDA_HOME unset) and crashes.
        # Instead, walk the model and replace each LoRA-wrapped Linear with
        # its underlying base_layer, then call save_pretrained on the
        # resulting plain transformer.
        merged_dir = out_root / "merged_final"
        merged_dir.mkdir(exist_ok=True)
        try:
            import torch.nn as _nn
            # Strip LoRA wrappers in-place: for each module that has a
            # base_layer attribute (LoraLayer subclass), we set the lora_A
            # and lora_B weights to zero/kaiming-equivalent (already done by
            # merge_and_reset_lora). The simplest correct path: find the
            # underlying transformer (model.base_model.model is the HF model
            # itself for PEFT-wrapped Causal LMs) and save THAT. The base
            # weights have all merges folded in; the PEFT wrapper just adds
            # zero-delta LoRA modules on top.
            if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                inner = model.base_model.model
            elif hasattr(model, "model"):
                inner = model.model
            else:
                inner = model
            # Replace all LoRA-wrapped submodules with their base_layer so
            # save_pretrained writes a clean transformer (no LoRA shards).
            n_replaced = 0
            for parent_name, parent_mod in list(inner.named_modules()):
                for child_name, child in list(parent_mod.named_children()):
                    if hasattr(child, "base_layer"):
                        setattr(parent_mod, child_name, child.base_layer)
                        n_replaced += 1
            log.info(f"replaced {n_replaced} LoRA wrappers with base_layer")
            # Now inner is a plain HF transformer with all merge deltas
            # folded in. Save via transformers' native save_pretrained.
            inner.save_pretrained(str(merged_dir), safe_serialization=True)
            tok.save_pretrained(str(merged_dir))
            log.info(f"merged_final saved to {merged_dir} "
                     f"(post-all-merges base; correct lm-eval target for do_relora)")
        except Exception as e:
            import traceback
            log.error(f"merged_final save FAILED: {e}\n{traceback.format_exc()}")
            with open(merged_dir / "SAVE_FAILED.flag", "w") as f:
                f.write(f"{type(e).__name__}: {e}\n")

    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
