#!/usr/bin/env python
"""Stage 1: predictive validity of val saliency vs train saliency.

Per-task pipeline (handover §3):
  1. Build LoRA model, load 3-way split.
  2. Train for total_steps, snapshot at save_steps.
  3. At each checkpoint:
       - eval baseline test loss on test_holdout
       - compute 5 saliency variants (S1..S5) + S3 signed
       - run oracle ablation (zero each component, measure delta_test)
       - write components.jsonl with all per-(layer,comp) records
       - compute Spearman correlations + AUC for harmful detection
  4. Save per-step JSON + final train/test loss curves.

Outputs under cfg.output_dir / <step>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import evaluate, oracle_ablation
from src.data import load_glue_three_split, make_loaders, tokenize_splits
from src.model import build_lora_model, count_lora_components, get_lora_BA_handles
from src.saliency import (
    fisher_saliency,
    first_order_saliency,
    magnitude_saliency,
    merge_records_by_key,
    saliency_dict_to_records,
)
from src.utils import (
    append_jsonl,
    dump_yaml,
    get_logger,
    load_yaml,
    set_seed,
    write_json,
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; ties handled via average ranks."""
    from scipy.stats import spearmanr
    if a.size < 2:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho) if rho == rho else 0.0  # guard NaN


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC for binary labels; tie-aware."""
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def run_diagnostics_at_checkpoint(
    model, handles, device, train_diag_loader, diag_loader, test_loader,
    cfg, step, log, out_root: Path,
    k_fold: int = 1,
) -> dict[str, Any]:
    """Compute baseline + 5 saliency variants + oracle ablation; write artifacts.

    If k_fold > 1, additionally compute S5_fisher_val_kfold by averaging Fisher
    over K disjoint folds of the diagnostic loader (PI 05 §3.3 directive).
    """
    step_dir = out_root / str(step)
    step_dir.mkdir(parents=True, exist_ok=True)

    # 1. baseline test loss
    base_loss, base_acc = evaluate(model, test_loader, device)
    log.info(f"[step {step}] baseline test loss={base_loss:.4f} acc={base_acc:.4f}")

    # 2. saliency
    mag = magnitude_saliency(handles)
    fo_train = first_order_saliency(model, handles, train_diag_loader, device,
                                    max_batches=cfg["saliency_batches"], signed=False)
    fo_val = first_order_saliency(model, handles, diag_loader, device,
                                  max_batches=cfg["saliency_batches"], signed=False)
    fo_val_signed = first_order_saliency(model, handles, diag_loader, device,
                                         max_batches=cfg["saliency_batches"], signed=True)
    fisher_tr = fisher_saliency(model, handles, train_diag_loader, device,
                                max_samples=cfg["fisher_max_samples"])
    fisher_vl = fisher_saliency(model, handles, diag_loader, device,
                                max_samples=cfg["fisher_max_samples"])

    # 2b. K-fold Fisher val (variance-reduction; PI 05 §3.3)
    fisher_vl_kfold: dict[str, torch.Tensor] | None = None
    if k_fold > 1:
        from collections import defaultdict
        from torch.utils.data import DataLoader, Subset
        diag_dataset = diag_loader.dataset
        n_diag = len(diag_dataset)
        fold_size = n_diag // k_fold
        fold_accum: dict[str, list] = defaultdict(list)
        for fold in range(k_fold):
            fold_idx = list(range(fold * fold_size,
                                  n_diag if fold == k_fold - 1 else (fold + 1) * fold_size))
            sub = Subset(diag_dataset, fold_idx)
            fold_loader = DataLoader(sub, batch_size=diag_loader.batch_size, shuffle=False)
            s5_fold = fisher_saliency(model, handles, fold_loader, device,
                                      max_samples=cfg["fisher_max_samples"])
            for layer, vec in s5_fold.items():
                fold_accum[layer].append(vec)
        fisher_vl_kfold = {layer: torch.stack(vs).mean(dim=0) for layer, vs in fold_accum.items()}
        log.info(f"[step {step}] computed {k_fold}-fold Fisher val saliency")

    # 3. oracle ablation
    log.info(f"[step {step}] starting oracle ablation over {sum(h.r for h in handles)} components")
    t_abl = time.time()
    abl_records = oracle_ablation(
        model, handles, test_loader, device, baseline_loss=base_loss,
        max_test_examples=cfg.get("oracle_test_max"),
    )
    log.info(f"[step {step}] oracle ablation done in {time.time()-t_abl:.1f}s")

    # 4. assemble per-component table
    r_mag = saliency_dict_to_records(mag, "S1_mag")
    r_fot = saliency_dict_to_records(fo_train, "S2_fo_tr")
    r_fov = saliency_dict_to_records(fo_val, "S3_fo_val")
    r_fovs = saliency_dict_to_records(fo_val_signed, "S3_fo_val_signed")
    r_ftr = saliency_dict_to_records(fisher_tr, "S4_fisher_tr")
    r_fva = saliency_dict_to_records(fisher_vl, "S5_fisher_val")
    record_lists = [r_mag, r_fot, r_fov, r_fovs, r_ftr, r_fva]
    if fisher_vl_kfold is not None:
        r_fkf = saliency_dict_to_records(fisher_vl_kfold, "S5_fisher_val_kfold")
        record_lists.append(r_fkf)
    record_lists.append(abl_records)
    merged = merge_records_by_key(*record_lists, key_fields=("layer", "comp"))
    # add harmful flag
    for rec in merged:
        rec["harmful_flag"] = bool(rec["delta_test"] < 0)
        rec["step"] = step

    comp_path = step_dir / "components.jsonl"
    if comp_path.exists():
        comp_path.unlink()
    for rec in merged:
        append_jsonl(str(comp_path), rec)

    # 5. correlations and AUC
    delta = np.array([r["delta_test"] for r in merged])
    labels = np.array([1 if r["harmful_flag"] else 0 for r in merged])
    corr = {}
    for name in ["S1_mag", "S2_fo_tr", "S3_fo_val", "S4_fisher_tr", "S5_fisher_val"]:
        s = np.array([r[name] for r in merged])
        corr[name + "_rho_vs_delta"] = _spearman(s, delta)
        corr[name + "_rho_vs_abs_delta"] = _spearman(s, np.abs(delta))
    if fisher_vl_kfold is not None and "S5_fisher_val_kfold" in merged[0]:
        s = np.array([r["S5_fisher_val_kfold"] for r in merged])
        corr["S5_fisher_val_kfold_rho_vs_delta"] = _spearman(s, delta)
        corr["S5_fisher_val_kfold_rho_vs_abs_delta"] = _spearman(s, np.abs(delta))
    write_json(str(step_dir / "correlations.json"), corr)

    # AUC: use NEGATIVE signed val saliency to score "harmful"
    s_signed = np.array([r["S3_fo_val_signed"] for r in merged])
    auc_signed = _auc(-s_signed, labels)
    auc_dict = {
        "S3_fo_val_signed_neg_auc_harmful": auc_signed,
        "n_harmful": int(labels.sum()),
        "n_total": int(labels.size),
        "harmful_rate": float(labels.mean()),
        "baseline_test_loss": float(base_loss),
        "baseline_test_acc": float(base_acc),
    }
    write_json(str(step_dir / "auc_signed.json"), auc_dict)

    log.info(f"[step {step}] rho(S5_fisher_val vs delta)= {corr['S5_fisher_val_rho_vs_delta']:.3f}, "
             f"rho(S4_fisher_tr)= {corr['S4_fisher_tr_rho_vs_delta']:.3f}, "
             f"rho(S3_fo_val)= {corr['S3_fo_val_rho_vs_delta']:.3f}, "
             f"rho(S2_fo_tr)= {corr['S2_fo_tr_rho_vs_delta']:.3f}, "
             f"AUC_signed(harmful)= {auc_signed:.3f}, harmful%= {labels.mean():.2%}")
    return {"step": step, **corr, **auc_dict}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None,
                        help="Override output dir; default uses cfg.output_dir under ROOT.")
    parser.add_argument("--out_dir", default=None,
                        help="Alias for --out (PI 05 §3.4 compatibility).")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Override total training steps (for quick tests).")
    parser.add_argument("--k_fold", type=int, default=1,
                        help="K-fold cross-validation for S5_fisher_val. Each fold "
                             "averages fisher saliency over its hold-out chunk. "
                             "K=1 reproduces Stage 1 behavior (PI 05 §3.3).")
    parser.add_argument("--fisher_max_samples", type=int, default=None,
                        help="Override cfg.fisher_max_samples (e.g. 512 for Path B).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.fisher_max_samples is not None:
        cfg["fisher_max_samples"] = args.fisher_max_samples
    out_override = args.out or args.out_dir
    out_root = Path(out_override) if out_override else (ROOT / cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"stage1.{cfg['task']}", str(out_root / "run.log"))
    if args.k_fold > 1:
        log.info(f"K-fold Fisher rerun: k_fold={args.k_fold} fisher_max_samples={cfg['fisher_max_samples']}")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"task={cfg['task']} device={device} out={out_root}")

    model, tok = build_lora_model(
        model_name=cfg["model_name"],
        num_labels=cfg["num_labels"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
    )
    model.to(device)
    handles = get_lora_BA_handles(model)
    n_comp = count_lora_components(handles)
    log.info(f"#LoRA layers={len(handles)}, total rank-1 components={n_comp}")

    splits = load_glue_three_split(cfg["task"], cfg["diagnostic_ratio"], cfg["seed"])
    tok_splits = tokenize_splits(splits, cfg["task"], tok, max_len=cfg["max_seq_len"])
    train_loader, diag_loader, test_loader = make_loaders(
        tok_splits, batch_size=cfg["batch_size"], eval_batch_size=cfg["eval_batch_size"]
    )
    # A train-side loader matching batch size for fair saliency comparison
    from torch.utils.data import DataLoader
    train_diag_loader = DataLoader(
        tok_splits.train_main, batch_size=cfg["eval_batch_size"], shuffle=False,
    )
    log.info(f"#train_main={len(tok_splits.train_main)}, #diag={len(tok_splits.diagnostic)}, "
             f"#test_holdout={len(tok_splits.test_holdout)}")

    total_steps = args.max_train_steps if args.max_train_steps else cfg["total_steps"]
    save_steps = sorted(s for s in cfg["save_steps"] if s <= total_steps)
    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"],
    )
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=cfg["optim"]["warmup_steps"], num_training_steps=total_steps,
    )

    dump_yaml(cfg, str(out_root / "config.yaml"))
    train_loss_path = out_root / "train_loss.jsonl"
    test_loss_path = out_root / "test_loss.jsonl"
    if train_loss_path.exists(): train_loss_path.unlink()
    if test_loss_path.exists(): test_loss_path.unlink()
    summary_records: list[dict[str, Any]] = []

    log.info(f"Training {total_steps} steps; saving diagnostics at {save_steps}")
    model.train()
    step = 0
    running = 0.0
    log_every = max(50, total_steps // 50)
    t0 = time.time()
    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            running += float(out.loss.item())
            step += 1
            if step % log_every == 0:
                avg = running / log_every
                running = 0.0
                log.info(f"step={step}/{total_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")
                append_jsonl(str(train_loss_path), {"step": step, "train_loss": avg})

            if step in save_steps:
                # snapshot adapter state
                ckpt_dir = out_root / str(step)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                            if "lora_" in k},
                           str(ckpt_dir / "lora_state.pt"))
                # diagnostics
                model.eval()
                rec = run_diagnostics_at_checkpoint(
                    model, handles, device, train_diag_loader, diag_loader, test_loader,
                    cfg, step, log, out_root,
                    k_fold=args.k_fold,
                )
                append_jsonl(str(test_loss_path), {"step": step,
                                                    "test_loss": rec["baseline_test_loss"],
                                                    "test_acc": rec["baseline_test_acc"]})
                summary_records.append(rec)
                model.train()

    elapsed = time.time() - t0
    log.info(f"training+diagnostics done in {elapsed:.1f}s")

    # Save aggregated summary for this task
    write_json(str(out_root / "summary.json"), {
        "task": cfg["task"],
        "elapsed_sec": elapsed,
        "checkpoints": summary_records,
        "n_components": n_comp,
        "n_lora_layers": len(handles),
    })
    log.info(f"summary.json -> {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
