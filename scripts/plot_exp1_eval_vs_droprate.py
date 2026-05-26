"""Exp-1 vllm eval visualization + S3 route classifier (PI feedback #2 §4 + #3 §2).

Reads `lm_eval/.../results_*.json` for each Exp-1 cell and produces:

  analysis/results_v3/exp1_eval_vs_droprate.png       (2x2 grid plot)
  analysis/results_v3/exp1_eval_vs_droprate.json      (numeric summary)
  analysis/results_v3/exp1_eval_route.json            (S3 route classification)
  analysis/COMM_GPU5_2026-05-26_<HHMM>_exp1_eval_summary.md
                                                       (interpretation memo)

Decision tree (gsm8k_flex primary, cross-metric sanity required):
  monotonic↑ : strictly increasing AND dr=0.9 best by ≥1.0pp over dr=0  -> Branch A
  monotonic↓ : strictly decreasing AND dr=0 best by ≥1.0pp over dr=0.9  -> Branch C
  U-shape    : peak at dr ∈ {0.25, 0.5} AND peak ≥ both endpoints by ≥1.0pp  -> Branch B
  flat       : (max - min) < 1.0pp                                         -> Branch D
  ambiguous  : none of the above                                           -> Branch E

If gsm8k_flex shape disagrees with the other 3 metrics -> demote to Branch E.

Usage:
    python scripts/plot_exp1_eval_vs_droprate.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "results" / "exp_drop_rate" / "qwen3-8b" / "tulu3-sft"
DROP_RATES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
DR_LABELS = ["dr0", "dr0.1", "dr0.25", "dr0.5", "dr0.75", "dr0.9"]

METRICS = [
    ("gsm8k_strict", "GSM8K (strict-match)"),
    ("gsm8k_flex", "GSM8K (flexible-extract)"),
    ("hellaswag", "HellaSwag (acc_norm)"),
    ("arc_challenge", "ARC-Challenge (acc_norm)"),
]


def find_results_json(label_dir: Path) -> Path | None:
    eval_root = label_dir / "seed42" / "lm_eval"
    if not eval_root.exists():
        return None
    cands = list(eval_root.rglob("results_*.json"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def extract_metric(results: dict, metric_key: str) -> float | None:
    """Pull metric from lm_eval results dict; handle multiple key variants."""
    res = results.get("results", {})
    # gsm8k specifically: 'gsm8k' top-level; sub keys 'exact_match,strict-match' / 'exact_match,flexible-extract'
    if metric_key == "gsm8k_strict":
        for k, v in res.items():
            if "gsm8k" in k.lower():
                for sub in v:
                    if sub.startswith("exact_match,strict"):
                        return float(v[sub])
        return None
    if metric_key == "gsm8k_flex":
        for k, v in res.items():
            if "gsm8k" in k.lower():
                for sub in v:
                    if sub.startswith("exact_match,flexible"):
                        return float(v[sub])
        return None
    if metric_key == "hellaswag":
        v = res.get("hellaswag", {})
        for sub in v:
            if sub.startswith("acc_norm,"):
                return float(v[sub])
        if "acc_norm" in v:
            return float(v["acc_norm"])
        return None
    if metric_key == "arc_challenge":
        v = res.get("arc_challenge", {})
        for sub in v:
            if sub.startswith("acc_norm,"):
                return float(v[sub])
        if "acc_norm" in v:
            return float(v["acc_norm"])
        return None
    return None


def classify_shape(values: list[float],
                   drop_rates: list[float],
                   thresh_pp: float = 1.0) -> dict:
    """Apply PI #3 §2 decision tree on a list of metric values.

    All values in [0,1] -> we compare in percentage points (pp).
    """
    arr = np.array(values, dtype=float)
    if np.any(np.isnan(arr)):
        return {"shape": "incomplete", "reason": "missing values"}
    pp = arr * 100.0
    diffs = np.diff(pp)
    spread = float(pp.max() - pp.min())
    peak_idx = int(np.argmax(pp))
    peak_dr = drop_rates[peak_idx]
    end_diff = float(pp[-1] - pp[0])  # dr=0.9 - dr=0
    info = {
        "values_pct": [round(v, 3) for v in pp.tolist()],
        "drop_rates": drop_rates,
        "spread_pp": round(spread, 3),
        "peak_dr": peak_dr,
        "peak_pct": round(float(pp[peak_idx]), 3),
        "endpoint_gap_pp": round(end_diff, 3),
    }
    if spread < thresh_pp:
        info["shape"] = "flat"
        info["reason"] = f"spread {spread:.2f}pp < {thresh_pp}pp"
        return info
    monotonic_up = bool(np.all(diffs >= -1e-6)) and end_diff >= thresh_pp
    monotonic_down = bool(np.all(diffs <= 1e-6)) and end_diff <= -thresh_pp
    if monotonic_up:
        info["shape"] = "monotonic_up"
        info["reason"] = f"strictly nondecreasing, dr=0.9-dr=0 = {end_diff:+.2f}pp"
        return info
    if monotonic_down:
        info["shape"] = "monotonic_down"
        info["reason"] = f"strictly nonincreasing, dr=0.9-dr=0 = {end_diff:+.2f}pp"
        return info
    if peak_dr in (0.25, 0.5):
        gap0 = float(pp[peak_idx] - pp[0])
        gap_end = float(pp[peak_idx] - pp[-1])
        if gap0 >= thresh_pp and gap_end >= thresh_pp:
            info["shape"] = "U_shape"
            info["reason"] = (f"peak at dr={peak_dr}, "
                              f"gaps to endpoints = {gap0:+.2f} / {gap_end:+.2f}pp")
            return info
    info["shape"] = "ambiguous"
    info["reason"] = (f"peak at dr={peak_dr} (gap {pp[peak_idx]-pp[0]:.2f}pp), "
                      f"spread {spread:.2f}pp, end_diff {end_diff:+.2f}pp")
    return info


def route_decision(metrics: dict) -> dict:
    """Apply gsm8k_flex primary + cross-metric sanity per PI #3 §2."""
    primary = metrics.get("gsm8k_flex", {})
    primary_shape = primary.get("shape")
    other_shapes = [metrics[k]["shape"]
                    for k in ("gsm8k_strict", "hellaswag", "arc_challenge")
                    if k in metrics]
    agreement = sum(1 for s in other_shapes if s == primary_shape)
    sanity_pass = agreement >= 2
    branch_map = {
        "monotonic_up":   "A_monotonic_up_schedule_sweep",
        "U_shape":        "B_U_shape_schedule_sweep",
        "monotonic_down": "C_monotonic_down_BLOCKER",
        "flat":           "D_FLAT_BLOCKER",
        "ambiguous":      "E_ambiguous_tiebreak",
        "incomplete":     "X_incomplete",
    }
    route_branch = branch_map.get(primary_shape, "X_incomplete")
    if not sanity_pass and primary_shape in ("monotonic_up", "U_shape"):
        route_branch = "E_ambiguous_tiebreak"
        sanity_note = (f"primary={primary_shape} but cross-metric agreement only "
                       f"{agreement}/3 -> demoted to Branch E per PI #3 §2 sanity")
    else:
        sanity_note = f"cross-metric agreement {agreement}/3 with primary"
    return {
        "primary_metric": "gsm8k_flex",
        "primary_shape": primary_shape,
        "cross_metric_agreement": f"{agreement}/3",
        "sanity_pass": sanity_pass,
        "route": route_branch,
        "sanity_note": sanity_note,
    }


def load_all() -> dict:
    out = {"cells": {}, "missing": []}
    for dr_label in DR_LABELS:
        cell_dir = EXP_ROOT / dr_label
        path = find_results_json(cell_dir)
        if path is None:
            out["missing"].append(dr_label)
            continue
        with path.open() as f:
            results = json.load(f)
        cell_metrics = {mk: extract_metric(results, mk) for mk, _ in METRICS}
        out["cells"][dr_label] = {
            "results_json": str(path.relative_to(ROOT)),
            "metrics": cell_metrics,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_png",
                    default="analysis/results_v3/exp1_eval_vs_droprate.png")
    ap.add_argument("--out_json",
                    default="analysis/results_v3/exp1_eval_vs_droprate.json")
    ap.add_argument("--out_route",
                    default="analysis/results_v3/exp1_eval_route.json")
    ap.add_argument("--allow_partial", action="store_true",
                    help="Proceed even with missing cells (route may be incomplete).")
    args = ap.parse_args()

    data = load_all()
    if data["missing"] and not args.allow_partial:
        print(f"[ERR] Missing eval results for: {data['missing']}")
        print("      Use --allow_partial or wait for vllm eval to finish.")
        return 1

    by_metric = {}
    metric_classifications = {}
    for mk, _ in METRICS:
        vals = []
        for dr_label in DR_LABELS:
            v = data["cells"].get(dr_label, {}).get("metrics", {}).get(mk)
            vals.append(float(v) if v is not None else float("nan"))
        by_metric[mk] = vals
        metric_classifications[mk] = classify_shape(vals, DROP_RATES)

    route = route_decision(metric_classifications)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for (mk, title), ax in zip(METRICS, axes.flat):
        vals = np.array(by_metric[mk]) * 100
        ax.plot(DROP_RATES, vals, "o-", lw=2, color="steelblue")
        for x, y in zip(DROP_RATES, vals):
            if not np.isnan(y):
                ax.text(x, y, f" {y:.2f}", fontsize=8, va="center")
        cls = metric_classifications[mk]
        ax.set_title(f"{title}\nshape={cls['shape']}  spread={cls.get('spread_pp','?')}pp"
                     f"  peak@dr={cls.get('peak_dr','?')}",
                     fontsize=10)
        ax.set_xlabel("drop_rate")
        ax.set_ylabel("score (%)")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Exp-1 vllm eval vs drop_rate (qwen3-8b/tulu3-sft, total_steps=3000)\n"
        f"S3 route = {route['route']}  ({route['sanity_note']})",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = ROOT / args.out_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"WROTE {out_png}")

    summary = {
        "drop_rates": DROP_RATES,
        "by_metric_pct": {k: [round(v * 100, 3) if not np.isnan(v) else None
                              for v in by_metric[k]]
                          for k in by_metric},
        "shape_classifications": metric_classifications,
        "route": route,
        "missing_cells": data["missing"],
        "cells_meta": data["cells"],
    }
    out_json = ROOT / args.out_json
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"WROTE {out_json}")

    out_route = ROOT / args.out_route
    out_route.write_text(json.dumps(route, indent=2))
    print(f"WROTE {out_route}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    memo_path = (ROOT / "analysis" /
                 f"COMM_GPU5_{ts}_exp1_eval_summary.md")
    memo_lines = [
        f"# Exp-1 vllm eval summary — {ts} UTC",
        "",
        "Auto-generated by `scripts/plot_exp1_eval_vs_droprate.py`.",
        "",
        f"**Route decision: `S3_ROUTE={route['route']}`**",
        "",
        f"- Primary metric: gsm8k_flex (shape={route['primary_shape']})",
        f"- Cross-metric agreement: {route['cross_metric_agreement']} "
        f"({'PASS' if route['sanity_pass'] else 'FAIL -> demoted to E'})",
        f"- Sanity note: {route['sanity_note']}",
        "",
        "## Metric values (%)",
        "",
        "| dr | gsm8k_strict | gsm8k_flex | hellaswag | arc_challenge |",
        "|---:|---:|---:|---:|---:|",
    ]
    for i, dr in enumerate(DROP_RATES):
        cells = []
        for mk, _ in METRICS:
            v = by_metric[mk][i]
            cells.append(f"{v*100:.2f}" if not np.isnan(v) else "—")
        memo_lines.append(f"| {dr} | " + " | ".join(cells) + " |")
    memo_lines += [
        "",
        "## Per-metric shape classification",
        "",
    ]
    for mk, _ in METRICS:
        cls = metric_classifications[mk]
        memo_lines.append(f"- **{mk}**: shape=`{cls['shape']}` — {cls.get('reason','')}")
    memo_lines += [
        "",
        "## Next action",
        "",
        f"Per PI feedback #3 §2 decision tree:",
    ]
    if route["route"].startswith(("A_", "B_")):
        memo_lines.append(
            f"- AUTO-LAUNCH 12-cell schedule × selection sweep "
            f"(see `scripts/exp_schedule_pilot_orchestrator.py`)."
        )
        memo_lines.append(
            f"- Commit body string: `S3_ROUTE={route['route']}`"
        )
    elif route["route"].startswith("E_"):
        memo_lines.append(
            "- AUTO-LAUNCH 4-cell tie-break sweep "
            "(dr ∈ {0.05, 0.15, 0.2, 0.3}, qwen3-8b/tulu3, 3000 steps)."
        )
        memo_lines.append(
            "- Push notice memo before launching so PI can override."
        )
    elif route["route"].startswith("C_"):
        memo_lines.append(
            "- DO NOT auto-launch. Push BLOCKER memo and wait for PI."
        )
    elif route["route"].startswith("D_"):
        memo_lines.append(
            "- DO NOT auto-launch. EMERGENCY ping with full triage data."
        )
    memo_path.write_text("\n".join(memo_lines))
    print(f"WROTE {memo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
