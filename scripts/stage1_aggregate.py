#!/usr/bin/env python
"""Stage 1 aggregation + decision (handover §3.8).

Reads `results/stage1/<task>/summary.json` for 3 tasks AND
re-derives AUCs directly from `<task>/<step>/components.jsonl` so we can:
  - compute AUC for harmful detection in BOTH sign directions and take max
    (the handover wants "informativeness", which is sign-symmetric)
  - compute AUC for unsigned val-Fisher (S5) and val-magnitude (S3)
  - cross-check Spearman correlations

Outputs:
  - results/stage1/summary/correlation_matrix.csv
  - results/stage1/summary/correlation_aggregate.json
  - results/stage1/summary/decision.json
  - results/stage1/summary/per_pair_table.csv   (one row per (task, step))

Decision rule (handover §3.8):
  GO iff ALL of:
    1. mean(delta_rho_fisher) >= 0.10 AND mean(delta_rho_fo) >= 0.05
    2. positive on >=10 of 15 (task,step) pairs
    3. SYMMETRIC AUC for S3_fo_val_signed (harmful detection) >= 0.65 on >=1 task at LATEST checkpoint
  STOP if mean(delta_rho_fisher) < 0 AND val worse on > 8/15 pairs, OR all sym-AUCs < 0.55.
  AMBIGUOUS otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import write_json

TASKS = ["sst2", "mrpc", "rte"]
SAL_NAMES = ["S1_mag", "S2_fo_tr", "S3_fo_val", "S4_fisher_tr", "S5_fisher_val"]


def _read_summary(stage1_root: Path, task: str) -> dict[str, Any] | None:
    p = stage1_root / task / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _read_components(stage1_root: Path, task: str, step: int) -> list[dict] | None:
    p = stage1_root / task / str(step) / "components.jsonl"
    if not p.exists():
        return None
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _safe_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC; returns NaN when labels are single-class."""
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def _symmetric_auc(score: np.ndarray, labels: np.ndarray) -> float:
    """Return max(AUC(+score), AUC(-score)). Sign-direction-agnostic informativeness."""
    a = _safe_auc(score, labels)
    if a != a:  # NaN
        return float("nan")
    return max(a, 1.0 - a)


def _aucs_from_components(recs: list[dict]) -> dict[str, float]:
    """Compute AUC variants from a per-checkpoint components list."""
    if not recs:
        return {}
    labels = np.array([1 if r["harmful_flag"] else 0 for r in recs])
    s_signed = np.array([r["S3_fo_val_signed"] for r in recs])
    s_unsigned = np.array([r["S3_fo_val"] for r in recs])
    s5_unsigned = np.array([r["S5_fisher_val"] for r in recs])
    s4_unsigned = np.array([r["S4_fisher_tr"] for r in recs])
    return {
        "auc_s3_signed_sym": _symmetric_auc(s_signed, labels),
        "auc_s3_signed_neg": _safe_auc(-s_signed, labels),
        "auc_s3_signed_pos": _safe_auc(+s_signed, labels),
        "auc_s3_unsigned": _safe_auc(s_unsigned, labels),
        "auc_s5_unsigned": _safe_auc(s5_unsigned, labels),
        "auc_s4_unsigned": _safe_auc(s4_unsigned, labels),
        "n_harmful": int(labels.sum()),
        "n_total": int(labels.size),
        "harmful_rate": float(labels.mean()),
    }


def _bootstrap_mean_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05,
                       rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots[i] = sample.mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_root", default=str(ROOT / "results/stage1"))
    parser.add_argument("--out_dir", default=str(ROOT / "results/stage1/summary"))
    parser.add_argument("--use_abs_delta", action="store_true",
                        help="Use rho vs |delta_test| instead of rho vs signed delta_test.")
    args = parser.parse_args()

    stage1_root = Path(args.stage1_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for t in TASKS:
        s = _read_summary(stage1_root, t)
        if s is None:
            print(f"[warn] missing summary for {t}; skipping")
            continue
        summaries[t] = s
    if not summaries:
        print("ERROR: no summaries found"); return 2

    rho_key_suffix = "_rho_vs_abs_delta" if args.use_abs_delta else "_rho_vs_delta"

    # 1. Per-(task, step, saliency) correlation matrix
    rows: list[dict[str, Any]] = []
    for task, s in summaries.items():
        for ckpt in s["checkpoints"]:
            step = int(ckpt["step"])
            for name in SAL_NAMES:
                rho = ckpt.get(name + rho_key_suffix)
                rows.append({
                    "task": task, "step": step, "saliency": name,
                    "rho": rho,
                    "auc_harmful": (ckpt.get("S3_fo_val_signed_neg_auc_harmful")
                                    if name == "S3_fo_val" else None),
                    "n_harmful": ckpt.get("n_harmful"),
                    "harmful_rate": ckpt.get("harmful_rate"),
                    "baseline_test_loss": ckpt.get("baseline_test_loss"),
                    "baseline_test_acc": ckpt.get("baseline_test_acc"),
                })

    # Save CSV (simple manual writer to avoid pandas dependency reordering)
    cm_path = out_dir / "correlation_matrix.csv"
    with open(cm_path, "w") as f:
        cols = ["task", "step", "saliency", "rho", "auc_harmful",
                "n_harmful", "harmful_rate", "baseline_test_loss", "baseline_test_acc"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(("" if r[c] is None else str(r[c])) for c in cols) + "\n")

    # 2. Per-pair (task, step) delta_rho table + RECOMPUTED AUCs
    # NOTE handover §3.6: "Higher in absolute value means better predictor".
    # So delta_rho is computed on |rho| (informativeness, sign-agnostic).
    pairs: list[dict[str, Any]] = []
    for task, s in summaries.items():
        for ckpt in s["checkpoints"]:
            step = int(ckpt["step"])
            rho_s5 = ckpt.get("S5_fisher_val" + rho_key_suffix)
            rho_s4 = ckpt.get("S4_fisher_tr" + rho_key_suffix)
            rho_s3 = ckpt.get("S3_fo_val" + rho_key_suffix)
            rho_s2 = ckpt.get("S2_fo_tr" + rho_key_suffix)
            def _absnone(x):
                return abs(x) if x is not None else None
            arho_s5, arho_s4 = _absnone(rho_s5), _absnone(rho_s4)
            arho_s3, arho_s2 = _absnone(rho_s3), _absnone(rho_s2)
            delta_rho_fisher = (arho_s5 - arho_s4) if (arho_s5 is not None and arho_s4 is not None) else None
            delta_rho_fo = (arho_s3 - arho_s2) if (arho_s3 is not None and arho_s2 is not None) else None
            # AUCs from components.jsonl (overrides the in-summary AUC which is sign-fixed)
            recs = _read_components(stage1_root, task, step)
            aucs = _aucs_from_components(recs) if recs else {}
            pairs.append({
                "task": task, "step": step,
                "rho_S5_fisher_val": rho_s5,
                "rho_S4_fisher_tr": rho_s4,
                "rho_S3_fo_val": rho_s3,
                "rho_S2_fo_tr": rho_s2,
                "abs_rho_S5_fisher_val": arho_s5,
                "abs_rho_S4_fisher_tr": arho_s4,
                "abs_rho_S3_fo_val": arho_s3,
                "abs_rho_S2_fo_tr": arho_s2,
                "delta_rho_fisher": delta_rho_fisher,
                "delta_rho_fo": delta_rho_fo,
                "auc_s3_signed_sym": aucs.get("auc_s3_signed_sym"),
                "auc_s3_signed_pos": aucs.get("auc_s3_signed_pos"),
                "auc_s3_unsigned": aucs.get("auc_s3_unsigned"),
                "auc_s5_unsigned": aucs.get("auc_s5_unsigned"),
                "auc_s4_unsigned": aucs.get("auc_s4_unsigned"),
                "harmful_rate": aucs.get("harmful_rate", ckpt.get("harmful_rate")),
            })

    pp_path = out_dir / "per_pair_table.csv"
    with open(pp_path, "w") as f:
        cols = ["task", "step", "rho_S5_fisher_val", "rho_S4_fisher_tr",
                "rho_S3_fo_val", "rho_S2_fo_tr",
                "abs_rho_S5_fisher_val", "abs_rho_S4_fisher_tr",
                "abs_rho_S3_fo_val", "abs_rho_S2_fo_tr",
                "delta_rho_fisher", "delta_rho_fo",
                "auc_s3_signed_sym", "auc_s3_signed_pos", "auc_s3_unsigned",
                "auc_s5_unsigned", "auc_s4_unsigned",
                "harmful_rate"]
        f.write(",".join(cols) + "\n")
        for r in pairs:
            f.write(",".join(("" if r[c] is None else f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")

    # 3. Aggregates across all pairs
    dr_fi = [p["delta_rho_fisher"] for p in pairs if p["delta_rho_fisher"] is not None]
    dr_fo = [p["delta_rho_fo"] for p in pairs if p["delta_rho_fo"] is not None]
    n_pos_fi = sum(1 for v in dr_fi if v > 0)
    n_pos_fo = sum(1 for v in dr_fo if v > 0)
    n_total = max(len(dr_fi), len(dr_fo))

    rng = np.random.default_rng(42)
    mean_fi, lo_fi, hi_fi = _bootstrap_mean_ci(dr_fi, rng=rng)
    mean_fo, lo_fo, hi_fo = _bootstrap_mean_ci(dr_fo, rng=rng)

    # Per-task latest-checkpoint AUC (use symmetric)
    auc_sym_per_task: dict[str, float | None] = {}
    auc_unsigned_per_task: dict[str, float | None] = {}
    latest_step: dict[str, int | None] = {}
    for task, s in summaries.items():
        ckpts = s["checkpoints"]
        if not ckpts:
            auc_sym_per_task[task] = None; auc_unsigned_per_task[task] = None
            latest_step[task] = None; continue
        last = max(ckpts, key=lambda c: int(c["step"]))
        step = int(last["step"])
        recs = _read_components(stage1_root, task, step)
        if recs is None:
            auc_sym_per_task[task] = None; auc_unsigned_per_task[task] = None
            latest_step[task] = step; continue
        aucs = _aucs_from_components(recs)
        auc_sym_per_task[task] = aucs["auc_s3_signed_sym"]
        auc_unsigned_per_task[task] = aucs["auc_s3_unsigned"]
        latest_step[task] = step

    aggregate = {
        "delta_rho_fisher": {"mean": mean_fi, "ci95_lo": lo_fi, "ci95_hi": hi_fi,
                              "n_pos": n_pos_fi, "n_total": len(dr_fi)},
        "delta_rho_fo":     {"mean": mean_fo, "ci95_lo": lo_fo, "ci95_hi": hi_fo,
                              "n_pos": n_pos_fo, "n_total": len(dr_fo)},
        "auc_s3_signed_sym_per_task_at_latest_ckpt": auc_sym_per_task,
        "auc_s3_unsigned_per_task_at_latest_ckpt": auc_unsigned_per_task,
        "latest_ckpt_per_task": latest_step,
        "n_tasks": len(summaries),
        "rho_key": rho_key_suffix.lstrip("_"),
    }
    write_json(str(out_dir / "correlation_aggregate.json"), aggregate)

    # 4. Decision rule (handover §3.8) — use SYMMETRIC AUC since sign convention
    # in the original handover may have been backwards relative to empirical direction.
    aucs_sym = [v for v in auc_sym_per_task.values() if v is not None and v == v]
    any_sym_auc_ge_065 = any(v >= 0.65 for v in aucs_sym)
    all_sym_aucs_below_055 = (len(aucs_sym) > 0) and all(v < 0.55 for v in aucs_sym)

    cond1 = (mean_fi >= 0.10) and (mean_fo >= 0.05)
    cond2 = (n_pos_fi >= 10) or (n_pos_fo >= 10)
    cond3 = any_sym_auc_ge_065
    go = bool(cond1 and cond2 and cond3)

    stop_cond_a = (mean_fi < 0) and (len(dr_fi) - n_pos_fi > 8)
    stop_cond_b = all_sym_aucs_below_055
    stop = bool(stop_cond_a or stop_cond_b)

    if go and stop:
        verdict, label = "AMBIGUOUS_conflicting", "ambiguous"
    elif go:
        verdict, label = "GO_Stage_2", "go"
    elif stop:
        verdict, label = "STOP", "stop"
    else:
        verdict, label = "AMBIGUOUS_neither", "ambiguous"

    rationale = []
    rationale.append(f"mean(delta_rho_fisher)={mean_fi:.3f} (CI95 [{lo_fi:.3f},{hi_fi:.3f}]); threshold>=0.10 -> {'OK' if mean_fi >= 0.10 else 'FAIL'}")
    rationale.append(f"mean(delta_rho_fo)={mean_fo:.3f} (CI95 [{lo_fo:.3f},{hi_fo:.3f}]); threshold>=0.05 -> {'OK' if mean_fo >= 0.05 else 'FAIL'}")
    rationale.append(f"sign test fisher: {n_pos_fi}/{len(dr_fi)} positive (need >=10) -> {'OK' if n_pos_fi >= 10 else 'FAIL'}")
    rationale.append(f"sign test fo: {n_pos_fo}/{len(dr_fo)} positive -> {'OK' if n_pos_fo >= 10 else 'FAIL'}")
    rationale.append(f"sym AUC per task @ latest: {auc_sym_per_task}; any>=0.65 -> {'OK' if any_sym_auc_ge_065 else 'FAIL'}")
    rationale.append(f"all sym AUCs<0.55 -> {'YES (STOP signal)' if all_sym_aucs_below_055 else 'no'}")

    decision = {
        "go": go,
        "label": label,
        "verdict": verdict,
        "delta_rho_fisher_mean": mean_fi,
        "delta_rho_fisher_ci95": [lo_fi, hi_fi],
        "delta_rho_fo_mean": mean_fo,
        "delta_rho_fo_ci95": [lo_fo, hi_fo],
        "n_positive_fisher": n_pos_fi,
        "n_positive_fo": n_pos_fo,
        "n_pairs": len(dr_fi),
        "auc_s3_signed_sym_per_task_at_latest": auc_sym_per_task,
        "auc_s3_unsigned_per_task_at_latest": auc_unsigned_per_task,
        "rationale": rationale,
    }
    write_json(str(out_dir / "decision.json"), decision)

    print("\n========== Stage 1 decision ==========")
    print(f"verdict = {verdict}")
    for line in rationale:
        print("  " + line)
    print(f"\nWrote: {cm_path}\n       {pp_path}\n       {out_dir/'correlation_aggregate.json'}\n       {out_dir/'decision.json'}")
    return 0 if go else (3 if stop else 4)


if __name__ == "__main__":
    sys.exit(main())
