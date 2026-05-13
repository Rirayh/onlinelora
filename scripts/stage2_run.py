"""Stage 2: ReLoRA failure reproduction + diagnostic-gated fix.

Implements four methods on a tiny LLaMA-style decoder-only LM (Path A, 11M):
  - full_rank          : standard pretraining (oracle ceiling, no LoRA)
  - relora_baseline    : vanilla ReLoRA (Lialin 2023) — merge ALL LoRA components
                          into base every K steps, reset LoRA+optimizer
  - relora_diag_gated  : our method — at each merge event, compute val saliency,
                          only merge components passing the gate; dropped components
                          are reinit + their B,A entries reset to 0 (fresh slot)
  - relora_diag_gated_fisher : same, but gate uses Fisher (S5_fisher_val)

Outputs under results/stage2/<size>/<run_name>/:
  config.yaml, train_loss.jsonl, val_loss.jsonl,
  effective_rank.jsonl, condition_number.jsonl,
  saliency_at_merge.jsonl, run.log,
  ckpt/{step}.pt  (only at merge events)

Phase A scope (per PI directive §0): 11M only, 4 jobs in parallel on GPU 0/1/3/4.
Wall clock target: ≤ 6h per job.
"""
from __future__ import annotations

import argparse
import json
import math
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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peft import LoraConfig, get_peft_model

from src.effective_rank import condition_number, effective_rank
from src.model import LoraHandle, count_lora_components, get_lora_BA_handles
from src.saliency import (
    first_order_saliency,
    fisher_saliency,
    saliency_dict_to_records,
)
from src.tiny_lm import SIZE_CONFIGS, build_tiny_lm, count_params
from src.utils import append_jsonl, dump_yaml, get_logger, set_seed, write_json

GATE_CHOICES = ["S3_fo_val_signed", "S5_fisher_val", "none"]
GATE_SIGN_CHOICES = ["S3pos_drops", "S3neg_drops"]
METHOD_CHOICES = [
    "full_rank",
    "relora_baseline",
    "relora_diag_gated",
    "relora_diag_gated_fisher",
]


# -----------------------------------------------------------------------------
# Data loader: pack wikitext-2 into fixed-length contiguous LM windows
# -----------------------------------------------------------------------------
class PackedLMDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        n_full = (token_ids.numel() - 1) // seq_len
        self.data = token_ids[: n_full * seq_len + 1]
        self.n = n_full

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        ids = self.data[start : start + self.seq_len].clone()
        labels = ids.clone()
        return {"input_ids": ids, "labels": labels}


def build_data(tokenizer, seq_len: int, log) -> tuple[PackedLMDataset, PackedLMDataset]:
    from datasets import load_dataset
    log.info("loading wikitext-2-raw-v1")
    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    txt_tr = "\n".join([t for t in ds_train["text"] if t.strip()])
    txt_va = "\n".join([t for t in ds_val["text"] if t.strip()])
    ids_tr = torch.tensor(tokenizer(txt_tr, add_special_tokens=False)["input_ids"], dtype=torch.long)
    ids_va = torch.tensor(tokenizer(txt_va, add_special_tokens=False)["input_ids"], dtype=torch.long)
    log.info(f"train tokens={ids_tr.numel():,}  val tokens={ids_va.numel():,}")
    return PackedLMDataset(ids_tr, seq_len), PackedLMDataset(ids_va, seq_len)


# -----------------------------------------------------------------------------
# LoRA wrap (target = query, value of each Attention block)
# -----------------------------------------------------------------------------
def wrap_lora(model, r: int, alpha: int) -> nn.Module:
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["query", "value"],
        bias="none",
        # task_type omitted: this is a custom LM, not seq_cls / clm wrapped via HF
    )
    return get_peft_model(model, cfg)


# -----------------------------------------------------------------------------
# ReLoRA merge + reset
# -----------------------------------------------------------------------------
@torch.no_grad()
def merge_and_reset_lora(
    peft_model: nn.Module,
    handles: list[LoraHandle],
    keep_mask: dict[str, torch.Tensor],
    log,
) -> dict[str, Any]:
    """Add scaling * B[:,i] A[i,:] to base layer weight for components where keep_mask is True.

    `keep_mask[layer_name]` is a bool tensor of shape (r,). Components with True are merged
    into base; ALL components (kept or dropped) have their B,A reset to fresh init so the
    LoRA adapter starts the next ReLoRA cycle from scratch (Lialin 2023 protocol).

    Returns merge_stats dict.
    """
    merged_total = 0
    kept_per_layer: dict[str, int] = {}
    for h in handles:
        mask = keep_mask[h.name].to(h.A.device)
        # Find the underlying base linear layer (the one wrapped by peft)
        # peft's LoraLayer holds `.base_layer` pointing at the original nn.Linear
        # walk up to find owner module of this A/B
        owner = _find_lora_owner(peft_model, h.name)
        base_linear = owner.base_layer
        # Update base weight in place: W += scaling * B_kept @ A_kept
        if mask.any():
            r_keep = int(mask.sum().item())
            B_kept = h.B[:, mask]                    # (out, r_keep)
            A_kept = h.A[mask, :]                    # (r_keep, in)
            delta = (B_kept @ A_kept) * h.scaling    # (out, in)
            base_linear.weight.data.add_(delta.to(base_linear.weight.dtype))
            merged_total += r_keep
            kept_per_layer[h.name] = r_keep
        else:
            kept_per_layer[h.name] = 0
        # reset ALL components: kaiming-style for A, zeros for B (peft default)
        nn.init.kaiming_uniform_(h.A, a=math.sqrt(5))
        nn.init.zeros_(h.B)
    return {"merged_total": merged_total, "kept_per_layer": kept_per_layer}


def _find_lora_owner(peft_model: nn.Module, handle_name: str):
    """Given handle.name = '<module path>.<adapter_key>' return the LoRA-wrapped module."""
    # handle_name is "base_model.model.blocks.0.attn.query.default"
    # strip the adapter key suffix
    parts = handle_name.rsplit(".", 1)[0]
    mod = peft_model
    for p in parts.split("."):
        if p == "":
            continue
        mod = getattr(mod, p)
    return mod


# -----------------------------------------------------------------------------
# Effective rank / condition number per attention block (averaged over q,v)
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_rank_stats(peft_model: nn.Module, log) -> dict[str, Any]:
    """For each LoRA-wrapped Linear, compute effective rank and condition number of the
    *current effective* weight = base + scaling * B @ A.

    Returns per-layer dict + means.
    """
    per_layer: dict[str, dict[str, float]] = {}
    handles = get_lora_BA_handles(peft_model)
    for h in handles:
        owner = _find_lora_owner(peft_model, h.name)
        base_W = owner.base_layer.weight.detach()       # (out, in)
        delta = (h.B.detach() @ h.A.detach()) * h.scaling
        W = base_W + delta.to(base_W.dtype)
        er = effective_rank(W)
        cn = condition_number(W)
        per_layer[h.name] = {"effective_rank": float(er), "condition_number": float(cn)}
    mean_er = float(np.mean([v["effective_rank"] for v in per_layer.values()]))
    mean_cn = float(np.mean([v["condition_number"] for v in per_layer.values()]))
    return {"per_layer": per_layer, "mean_effective_rank": mean_er, "mean_condition_number": mean_cn}


@torch.no_grad()
def compute_rank_stats_fullrank(model: nn.Module) -> dict[str, Any]:
    """For full_rank baseline: compute ER/CN on attention query+value weights."""
    per_layer: dict[str, dict[str, float]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.endswith(("query", "value")):
            W = mod.weight.detach()
            per_layer[name] = {
                "effective_rank": float(effective_rank(W)),
                "condition_number": float(condition_number(W)),
            }
    if not per_layer:
        return {"per_layer": {}, "mean_effective_rank": float("nan"),
                "mean_condition_number": float("nan")}
    mean_er = float(np.mean([v["effective_rank"] for v in per_layer.values()]))
    mean_cn = float(np.mean([v["condition_number"] for v in per_layer.values()]))
    return {"per_layer": per_layer, "mean_effective_rank": mean_er, "mean_condition_number": mean_cn}


# -----------------------------------------------------------------------------
# Validation evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_lm(model, loader, device, max_batches: int = 50) -> float:
    model.eval()
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        total += float(out.loss.item())
        n += 1
    return total / max(n, 1)


# -----------------------------------------------------------------------------
# Gate predicate
# -----------------------------------------------------------------------------
def build_keep_mask(
    handles: list[LoraHandle],
    gate_signal: str,
    gate_sign: str,
    fo_val_signed: Optional[dict[str, torch.Tensor]],
    fisher_val: Optional[dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Decide which (layer, comp) pairs are MERGED (True) vs DROPPED (False).

    Sign convention recap (per PI 05 §1.2 sign-check verdict):
      - operational default: drop_if_S3>0 = gate_sign='S3pos_drops'
      - keep = (s_i < 0); drop = (s_i >= 0)
      - the other arm ('S3neg_drops') flips that: keep = (s_i > 0); drop = (s_i <= 0)

    For Fisher (always >= 0), use per-layer median as threshold: above-median is
    "informative" -> KEEP+MERGE, below-median is "noise" -> DROP+RESET.
    """
    masks: dict[str, torch.Tensor] = {}
    stats: dict[str, Any] = {"per_layer_keep_counts": {}, "all_scores": []}
    total = 0
    kept_total = 0
    for h in handles:
        r = h.r
        if gate_signal == "none":
            mask = torch.ones(r, dtype=torch.bool)
        elif gate_signal == "S3_fo_val_signed":
            assert fo_val_signed is not None, "need fo_val_signed scores"
            s = fo_val_signed[h.name]                    # (r,)
            if gate_sign == "S3pos_drops":
                mask = s < 0.0                            # keep if s < 0
            elif gate_sign == "S3neg_drops":
                mask = s > 0.0                            # keep if s > 0
            else:
                raise ValueError(gate_sign)
            stats["all_scores"].extend([float(v) for v in s.tolist()])
        elif gate_signal == "S5_fisher_val":
            assert fisher_val is not None, "need fisher_val scores"
            s = fisher_val[h.name]                       # (r,) >= 0
            thr = float(s.median().item())
            mask = s > thr                                # keep above-median
            stats["all_scores"].extend([float(v) for v in s.tolist()])
        else:
            raise ValueError(gate_signal)
        masks[h.name] = mask
        kept = int(mask.sum().item())
        stats["per_layer_keep_counts"][h.name] = kept
        kept_total += kept
        total += r
    stats["components_total"] = total
    stats["components_kept"] = kept_total
    stats["components_dropped"] = total - kept_total
    stats["drop_rate"] = 1.0 - kept_total / max(total, 1)
    if stats["all_scores"]:
        sc = np.array(stats["all_scores"])
        stats["score_quantiles"] = [float(np.quantile(sc, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    else:
        stats["score_quantiles"] = []
    return masks, stats


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(SIZE_CONFIGS.keys()), required=True)
    p.add_argument("--method", choices=METHOD_CHOICES, required=True)
    p.add_argument("--gate_signal", choices=GATE_CHOICES, default="S3_fo_val_signed")
    p.add_argument("--gate_sign", choices=GATE_SIGN_CHOICES, default="S3pos_drops",
                   help="S3pos_drops: drop if S3_fo_val_signed > 0 (PI 06 default). "
                        "S3neg_drops: drop if S3_fo_val_signed < 0 (insurance-layer arm).")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--merge_every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--rank_stat_every", type=int, default=500)
    p.add_argument("--saliency_batches", type=int, default=16)
    p.add_argument("--fisher_max_samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", type=str, default=None,
                   help="Output root; default results/stage2/<size>/<run_name>")
    args = p.parse_args()

    # ----- resolve method -> effective gate -----
    if args.method == "relora_diag_gated_fisher":
        effective_gate = "S5_fisher_val"
    elif args.method in ("full_rank", "relora_baseline"):
        effective_gate = "none"
    else:  # relora_diag_gated
        effective_gate = args.gate_signal

    # ----- naming -----
    if args.method == "relora_diag_gated":
        run_name = f"{args.method}_{args.gate_sign}"
    else:
        run_name = args.method
    out_root = Path(args.out_root) if args.out_root else (
        ROOT / "results" / "stage2" / args.size / run_name
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ckpt").mkdir(exist_ok=True)

    log = get_logger(f"stage2.{args.size}.{run_name}", str(out_root / "run.log"))
    log.info(f"size={args.size} method={args.method} effective_gate={effective_gate} "
             f"gate_sign={args.gate_sign} out={out_root}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- tokenizer + data -----
    tok = AutoTokenizer.from_pretrained("roberta-base")
    train_ds, val_ds = build_data(tok, args.seq_len, log)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    diag_loader  = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    log.info(f"#train_windows={len(train_ds)}  #val_windows={len(val_ds)}")

    # ----- model -----
    base = build_tiny_lm(args.size, tok.vocab_size)
    n_params = count_params(base)
    log.info(f"tiny_lm {args.size}: total params={n_params/1e6:.2f}M (incl. embedding)")

    if args.method == "full_rank":
        model = base.to(device)
        handles: list[LoraHandle] = []
    else:
        model = wrap_lora(base, r=args.lora_r, alpha=args.lora_alpha).to(device)
        handles = get_lora_BA_handles(model)
        log.info(f"#LoRA layers={len(handles)} #components={count_lora_components(handles)}")

    # ----- optimizer + scheduler -----
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log.info(f"#trainable params={n_trainable/1e6:.2f}M")
    optim = AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps
    )

    # ----- persist config -----
    dump_yaml({
        "size": args.size, "method": args.method, "effective_gate": effective_gate,
        "gate_sign": args.gate_sign, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "seq_len": args.seq_len, "batch_size": args.batch_size,
        "total_steps": args.total_steps, "merge_every": args.merge_every,
        "lr": args.lr, "warmup_steps": args.warmup_steps, "seed": args.seed,
        "n_params_total": n_params, "n_trainable": n_trainable,
    }, str(out_root / "config.yaml"))

    # ----- logs (empty) -----
    train_loss_path  = out_root / "train_loss.jsonl"
    val_loss_path    = out_root / "val_loss.jsonl"
    er_path          = out_root / "effective_rank.jsonl"
    cn_path          = out_root / "condition_number.jsonl"
    merge_path       = out_root / "saliency_at_merge.jsonl"
    for f in [train_loss_path, val_loss_path, er_path, cn_path, merge_path]:
        if f.exists():
            f.unlink()

    # ----- baseline rank stats at step 0 -----
    if args.method == "full_rank":
        rs0 = compute_rank_stats_fullrank(model)
    else:
        rs0 = compute_rank_stats(model, log)
    append_jsonl(str(er_path), {"step": 0, "mean_effective_rank": rs0["mean_effective_rank"],
                                 "per_layer": {k: v["effective_rank"] for k, v in rs0["per_layer"].items()}})
    append_jsonl(str(cn_path), {"step": 0, "mean_condition_number": rs0["mean_condition_number"],
                                 "per_layer": {k: v["condition_number"] for k, v in rs0["per_layer"].items()}})
    log.info(f"[step 0] mean_effective_rank={rs0['mean_effective_rank']:.3f} "
             f"mean_condition_number={rs0['mean_condition_number']:.2e}")

    # ----- merge schedule -----
    merge_steps = list(range(args.merge_every, args.total_steps + 1, args.merge_every))
    if args.method == "full_rank":
        merge_steps = []   # full_rank never merges (no LoRA)
    log.info(f"merge events scheduled at steps: {merge_steps}")

    # ----- training loop -----
    model.train()
    step = 0
    running = 0.0
    log_every = max(50, args.total_steps // 200)
    t0 = time.time()
    KILL_THRESHOLD = 1.10   # PI red-line: if val_loss > baseline*1.10, abort

    val_loss_full_baseline: Optional[float] = None  # filled by full_rank reference if present

    val_loader_iter = None

    while step < args.total_steps:
        for batch in train_loader:
            if step >= args.total_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            running += float(out.loss.item())
            step += 1

            if step % log_every == 0:
                avg = running / log_every
                running = 0.0
                lr_now = sched.get_last_lr()[0]
                log.info(f"step={step}/{args.total_steps} train_loss={avg:.4f} lr={lr_now:.2e} "
                         f"elapsed={time.time()-t0:.0f}s")
                append_jsonl(str(train_loss_path), {"step": step, "train_loss": avg, "lr": lr_now})

            if step % args.eval_every == 0:
                vl = evaluate_lm(model, val_loader, device, max_batches=50)
                log.info(f"step={step} VAL_LOSS={vl:.4f}  ppl={math.exp(min(vl, 30)):.2f}")
                append_jsonl(str(val_loss_path), {"step": step, "val_loss": vl})
                model.train()

            if step % args.rank_stat_every == 0:
                if args.method == "full_rank":
                    rs = compute_rank_stats_fullrank(model)
                else:
                    rs = compute_rank_stats(model, log)
                append_jsonl(str(er_path), {"step": step, "mean_effective_rank": rs["mean_effective_rank"],
                                             "per_layer": {k: v["effective_rank"] for k, v in rs["per_layer"].items()}})
                append_jsonl(str(cn_path), {"step": step, "mean_condition_number": rs["mean_condition_number"],
                                             "per_layer": {k: v["condition_number"] for k, v in rs["per_layer"].items()}})
                log.info(f"step={step} mean_effective_rank={rs['mean_effective_rank']:.3f} "
                         f"mean_condition_number={rs['mean_condition_number']:.2e}")

            # ----- ReLoRA merge event -----
            if step in merge_steps and args.method != "full_rank":
                event_idx = merge_steps.index(step) + 1
                log.info(f"=== MERGE EVENT {event_idx} at step {step} (gate={effective_gate}) ===")

                # --- compute val saliency at the merge boundary ---
                fo_val_signed = None
                fisher_val = None
                if effective_gate == "S3_fo_val_signed":
                    fo_val_signed = first_order_saliency(
                        model, handles, diag_loader, device,
                        max_batches=args.saliency_batches, signed=True,
                    )
                elif effective_gate == "S5_fisher_val":
                    fisher_val = fisher_saliency(
                        model, handles, diag_loader, device,
                        max_samples=args.fisher_max_samples,
                    )
                # effective_gate == "none" -> keep_mask all True

                keep_masks, stats = build_keep_mask(
                    handles, effective_gate, args.gate_sign, fo_val_signed, fisher_val
                )
                merge_stats = merge_and_reset_lora(model, handles, keep_masks, log)
                # reset optimizer state (Lialin protocol)
                optim = AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
                # rebuild scheduler keeping current step alignment
                remaining = args.total_steps - step
                sched = get_cosine_schedule_with_warmup(
                    optim, num_warmup_steps=min(args.warmup_steps, remaining // 4),
                    num_training_steps=remaining,
                )

                rec = {
                    "step": step,
                    "merge_event": event_idx,
                    "gate_signal": effective_gate,
                    "gate_sign": args.gate_sign,
                    "components_total": stats["components_total"],
                    "components_kept": stats["components_kept"],
                    "components_dropped": stats["components_dropped"],
                    "drop_rate": stats["drop_rate"],
                    "score_quantiles": stats["score_quantiles"],
                    "per_layer_keep_counts": stats["per_layer_keep_counts"],
                    "merged_total": merge_stats["merged_total"],
                }
                append_jsonl(str(merge_path), rec)
                log.info(f"merge stats: total={stats['components_total']} kept={stats['components_kept']} "
                         f"dropped={stats['components_dropped']} drop_rate={stats['drop_rate']:.3f}")

                # post-merge rank stats
                rs_post = compute_rank_stats(model, log)
                append_jsonl(str(er_path), {"step": step, "mean_effective_rank": rs_post["mean_effective_rank"],
                                             "per_layer": {k: v["effective_rank"] for k, v in rs_post["per_layer"].items()},
                                             "post_merge": True})
                append_jsonl(str(cn_path), {"step": step, "mean_condition_number": rs_post["mean_condition_number"],
                                             "per_layer": {k: v["condition_number"] for k, v in rs_post["per_layer"].items()},
                                             "post_merge": True})

                # post-merge val loss
                vl_post = evaluate_lm(model, val_loader, device, max_batches=50)
                append_jsonl(str(val_loss_path), {"step": step, "val_loss": vl_post, "post_merge": True})
                log.info(f"step={step} POST-MERGE VAL_LOSS={vl_post:.4f}")

                # PI red-line: kill job if val_loss > 1.10x of init val_loss
                if val_loss_full_baseline is None:
                    val_loss_full_baseline = vl_post
                else:
                    if vl_post > val_loss_full_baseline * KILL_THRESHOLD:
                        log.warning(f"KILL_THRESHOLD HIT: post-merge val_loss={vl_post:.4f} > "
                                    f"{val_loss_full_baseline:.4f}*{KILL_THRESHOLD}. Aborting per PI red-line.")
                        with open(out_root / "ABORTED.flag", "w") as f:
                            f.write(f"step={step} val_loss={vl_post:.4f} baseline={val_loss_full_baseline:.4f}\n")
                        write_json(str(out_root / "summary.json"), {
                            "size": args.size, "method": args.method,
                            "aborted": True, "abort_step": step,
                            "abort_val_loss": vl_post, "abort_baseline": val_loss_full_baseline,
                            "elapsed_sec": time.time() - t0,
                        })
                        return 2

                model.train()

    elapsed = time.time() - t0
    log.info(f"training done in {elapsed:.1f}s")

    # ----- final val eval -----
    vl_final = evaluate_lm(model, val_loader, device, max_batches=200)
    log.info(f"FINAL VAL_LOSS={vl_final:.4f}  ppl={math.exp(min(vl_final, 30)):.2f}")
    append_jsonl(str(val_loss_path), {"step": args.total_steps, "val_loss": vl_final, "final": True})

    if args.method != "full_rank":
        rs_final = compute_rank_stats(model, log)
    else:
        rs_final = compute_rank_stats_fullrank(model)
    write_json(str(out_root / "summary.json"), {
        "size": args.size, "method": args.method, "run_name": run_name,
        "effective_gate": effective_gate, "gate_sign": args.gate_sign,
        "total_steps": args.total_steps, "merge_every": args.merge_every,
        "final_val_loss": vl_final,
        "final_mean_effective_rank": rs_final["mean_effective_rank"],
        "final_mean_condition_number": rs_final["mean_condition_number"],
        "elapsed_sec": elapsed,
        "n_params_total": n_params, "n_trainable": n_trainable,
        "aborted": False,
    })
    log.info("summary.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
