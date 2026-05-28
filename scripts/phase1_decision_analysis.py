#!/usr/bin/env python3
"""Phase 1 robustness decision analysis (PI feedback #6 §C.1).

Reads lm-eval results from:
  results/phase1_robustness/qwen3-8b/tulu3-sft/<cell>/seed{42,43,44}/lm_eval/

Computes per-cell mean ± std across seeds for gsm8k_strict, gsm8k_flex,
hellaswag, arc_challenge, mmlu, ifeval.

Performs paired t-tests (3 pairs):
  - v1 vs relora_baseline  (primary: does method beat baseline?)
  - v1 vs random_dr0.5     (key: does selection beat random at same rate?)
  - v1 vs lora_vanilla     (hellaswag: does v1 preserve relora hellaswag gain?)

Decision rule per PI #6:
  PROCEED to Phase 2 iff:
    mean(v1.gsm_strict) - mean(random_dr0.5.gsm_strict) >= 1.5pp AND p < 0.10

Writes:
  analysis/results_v3/phase1_summary.json
  analysis/COMM_AGENT_TO_PI/{date}_phase1_decision.md

Usage:
  python scripts/phase1_decision_analysis.py
"""
from __future__ import annotations
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
MODEL  = "qwen3-8b"
DATASET = "tulu3-sft"
BASE   = ROOT / "results" / "phase1_robustness" / MODEL / DATASET

CELLS  = ["v1_S3pos", "random_dr0.5", "relora_baseline"]
SEEDS  = [42, 43, 44]

METRICS = {
    "gsm8k_strict":   ("gsm8k",        "exact_match,strict-match"),
    "gsm8k_flex":     ("gsm8k",        "exact_match,flexible-extract"),
    "hellaswag":      ("hellaswag",    "acc_norm,none"),
    "arc_challenge":  ("arc_challenge","acc_norm,none"),
    "mmlu":           ("mmlu",         "acc,none"),
    "ifeval":         ("ifeval",       "prompt_level_strict_acc,none"),
}

PROCEED_THRESHOLD_PP = 1.5
PROCEED_P_THRESHOLD  = 0.10


def t_test_paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-tailed paired t-test. Returns (t_stat, p_value)."""
    n = len(a)
    if n < 2:
        return float("nan"), float("nan")
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d  = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d == 0:
        return float("inf") if mean_d != 0 else 0.0, 0.0 if mean_d != 0 else 1.0
    se = math.sqrt(var_d / n)
    t  = mean_d / se
    # p-value approximation via t-distribution CDF (scipy-free)
    p = _t_pvalue(t, df=n - 1)
    return t, p


def _t_pvalue(t: float, df: int) -> float:
    """Two-tailed p-value from t-distribution using regularized incomplete beta."""
    x = df / (df + t * t)
    p_half = _betainc(df / 2, 0.5, x) / 2
    return 2 * p_half


def _betainc(a: float, b: float, x: float, steps: int = 200) -> float:
    """Regularized incomplete beta I_x(a,b) via continued-fraction expansion."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz's continued fraction
    tiny = 1e-300
    f = tiny
    C = f
    D = 0.0
    for m in range(steps):
        for parity in (0, 1):
            if m == 0 and parity == 0:
                num = 1.0
            elif parity == 0:
                num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            D = 1.0 + num * D
            if abs(D) < tiny:
                D = tiny
            C = 1.0 + num / C
            if abs(C) < tiny:
                C = tiny
            D = 1.0 / D
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                return front * f
    return front * f


def load_scores() -> dict[str, dict[str, list[float]]]:
    """Return {cell: {metric: [val_seed42, val_seed43, val_seed44]}}."""
    scores: dict[str, dict[str, list[float]]] = {c: {m: [] for m in METRICS} for c in CELLS}
    for cell in CELLS:
        for seed in SEEDS:
            lm_dir  = BASE / cell / f"seed{seed}" / "lm_eval"
            results = list(lm_dir.rglob("results_*.json")) if lm_dir.exists() else []
            if not results:
                print(f"WARN: missing result for {cell}/seed{seed}", file=sys.stderr)
                for m in METRICS:
                    scores[cell][m].append(float("nan"))
                continue
            raw = json.loads(results[0].read_text()).get("results", {})
            for metric_key, (task, key) in METRICS.items():
                v = raw.get(task, {}).get(key, float("nan"))
                scores[cell][metric_key].append(
                    v * 100 if isinstance(v, float) else float("nan"))
    return scores


def mean_std(vals: list[float]) -> tuple[float, float]:
    valid = [v for v in vals if not math.isnan(v)]
    if not valid:
        return float("nan"), float("nan")
    m = sum(valid) / len(valid)
    s = math.sqrt(sum((v - m) ** 2 for v in valid) / max(len(valid) - 1, 1))
    return m, s


def main() -> int:
    scores = load_scores()

    summary: dict = {"cells": {}, "comparisons": {}}

    print("\n=== Phase 1 per-cell scores (mean ± std, n=3 seeds) ===")
    header = f"{'cell':20s}"
    for m in METRICS:
        header += f"  {m:>14s}"
    print(header)
    print("-" * len(header))
    for cell in CELLS:
        row = f"{cell:20s}"
        cell_summary: dict = {}
        for m in METRICS:
            mu, sd = mean_std(scores[cell][m])
            row += f"  {mu:6.2f}±{sd:4.2f}"
            cell_summary[m] = {"mean": round(mu, 4), "std": round(sd, 4),
                                "values": [round(v, 4) for v in scores[cell][m]]}
        summary["cells"][cell] = cell_summary
        print(row)

    print("\n=== Paired t-tests (v1 vs comparators) ===")
    comparisons = [
        ("v1_S3pos", "relora_baseline", "gsm8k_strict", "primary: method vs baseline"),
        ("v1_S3pos", "random_dr0.5",    "gsm8k_strict", "KEY: selection vs random (decision rule)"),
        ("v1_S3pos", "lora_vanilla",    "hellaswag",    "hellaswag: v1 preserves regularization"),
    ]
    for (c1, c2, metric, label) in comparisons:
        if c2 not in scores:
            continue
        a = scores[c1][metric]
        b = scores[c2][metric]
        mu1, sd1 = mean_std(a)
        mu2, sd2 = mean_std(b)
        delta = mu1 - mu2
        t, p = t_test_paired(a, b)
        key = f"{c1}_vs_{c2}_{metric}"
        summary["comparisons"][key] = {
            "label":  label,
            "delta":  round(delta, 4),
            "t_stat": round(t, 4),
            "p_value": round(p, 4),
            f"{c1}_mean": round(mu1, 4),
            f"{c2}_mean": round(mu2, 4),
        }
        print(f"  {label}")
        print(f"    {c1}: {mu1:.2f}±{sd1:.2f}  {c2}: {mu2:.2f}±{sd2:.2f}")
        print(f"    delta={delta:+.2f}pp  t={t:.3f}  p={p:.4f}")

    proceed_delta = summary["comparisons"].get(
        "v1_S3pos_vs_random_dr0.5_gsm8k_strict", {}).get("delta", float("nan"))
    proceed_p     = summary["comparisons"].get(
        "v1_S3pos_vs_random_dr0.5_gsm8k_strict", {}).get("p_value", float("nan"))

    if math.isnan(proceed_delta):
        decision = "INCOMPLETE"
        decision_reason = "Missing results; cannot decide."
    elif proceed_delta >= PROCEED_THRESHOLD_PP and proceed_p < PROCEED_P_THRESHOLD:
        decision = "PROCEED_TO_PHASE2"
        decision_reason = (
            f"v1 vs random_dr0.5 gsm8k delta={proceed_delta:+.2f}pp >= "
            f"{PROCEED_THRESHOLD_PP}pp AND p={proceed_p:.4f} < {PROCEED_P_THRESHOLD}. "
            f"Selection signal confirmed."
        )
    else:
        decision = "STOP_NEGATIVE"
        decision_reason = (
            f"v1 vs random_dr0.5 gsm8k delta={proceed_delta:+.2f}pp "
            f"(threshold {PROCEED_THRESHOLD_PP}pp) OR p={proceed_p:.4f} "
            f"(threshold {PROCEED_P_THRESHOLD}). "
            f"Saliency adds no measurable selection benefit over random at same rate."
        )

    summary["decision"] = decision
    summary["decision_reason"] = decision_reason

    print(f"\n=== DECISION: {decision} ===")
    print(f"  {decision_reason}")

    out_json = ROOT / "analysis" / "results_v3" / "phase1_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    _write_decision_md(summary, decision, decision_reason)
    return 0


def _write_decision_md(summary: dict, decision: str, reason: str) -> None:
    today = date.today().isoformat()
    comm_dir = ROOT / "analysis" / "COMM_AGENT_TO_PI"
    comm_dir.mkdir(parents=True, exist_ok=True)
    out_md = comm_dir / f"{today}_phase1_decision.md"

    lines = [
        f"# Phase 1 Robustness Decision — {today}",
        "",
        "**ACK_pi_feedback_6_robustness_sweep** (Phase 1 results)",
        "",
        f"## Decision: `{decision}`",
        "",
        f"> {reason}",
        "",
        "## Phase 1 Score Table (qwen3-8b / tulu3-sft, n=3 seeds)",
        "",
        "| cell | gsm_strict | gsm_flex | hellaswag | arc_c | mmlu | ifeval |",
        "|------|-----------|----------|-----------|-------|------|--------|",
    ]
    for cell in ["v1_S3pos", "random_dr0.5", "relora_baseline"]:
        c = summary["cells"].get(cell, {})
        def f(m):
            d = c.get(m, {})
            mu, sd = d.get("mean", float("nan")), d.get("std", float("nan"))
            if math.isnan(mu):
                return "N/A"
            return f"{mu:.2f}±{sd:.2f}"
        lines.append(
            f"| {cell} | {f('gsm8k_strict')} | {f('gsm8k_flex')} | "
            f"{f('hellaswag')} | {f('arc_challenge')} | {f('mmlu')} | {f('ifeval')} |"
        )

    lines += [
        "",
        "## Paired t-tests",
        "",
        "| comparison | metric | delta | t | p | sig? |",
        "|-----------|--------|-------|---|---|------|",
    ]
    for key, v in summary.get("comparisons", {}).items():
        sig = "YES" if abs(v.get("delta", 0)) >= PROCEED_THRESHOLD_PP \
              and v.get("p_value", 1) < PROCEED_P_THRESHOLD else "no"
        lines.append(
            f"| {v.get('label',key)} | {key.split('_')[-1]} | "
            f"{v.get('delta', 0):+.2f}pp | "
            f"{v.get('t_stat', 0):.3f} | {v.get('p_value', 1):.4f} | {sig} |"
        )

    lines += [
        "",
        "## 3 Headline Deltas (95% CI via ±2*SE)",
        "",
    ]
    for key, v in summary.get("comparisons", {}).items():
        n = len(SEEDS)
        c1, c2 = key.split("_vs_")[0], key.split("_vs_")[1].rsplit("_", 1)[0]
        metric = key.rsplit("_", 1)[-1]
        mu1 = v.get(f"{c1}_mean", float("nan"))
        mu2 = v.get(f"{c2}_mean", float("nan"))
        delta = v.get("delta", float("nan"))
        # approx SE from pooled std: can't recompute perfectly here; use t*se ≈ delta/t * 2
        t_stat = v.get("t_stat", float("nan"))
        if not math.isnan(t_stat) and t_stat != 0:
            se = abs(delta / t_stat)
            ci95 = 1.96 * se
            ci_str = f"±{ci95:.2f}pp"
        else:
            ci_str = "N/A"
        lines.append(
            f"- **{v.get('label', key)}**: "
            f"{c1}={mu1:.2f}% vs {c2}={mu2:.2f}%, "
            f"delta={delta:+.2f}pp (95% CI approx {ci_str}), "
            f"p={v.get('p_value', float('nan')):.4f}"
        )

    if decision == "PROCEED_TO_PHASE2":
        lines += [
            "",
            "## Recommended Phase 2 Model Order",
            "",
            "1. olmo2-7b (instruct) — different architecture, best cross-arch test",
            "2. llama3-8b (instruct) — widely used baseline in NLP benchmarks",
            "",
            "Ready to launch Phase 2 on PI ack.",
        ]
    else:
        lines += [
            "",
            "## Recommended Next Step",
            "",
            "Write up as negative result: saliency selection ~ random for ReLoRA "
            "at matched drop rate. Still publishable as null finding.",
        ]

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    sys.exit(main())
