#!/usr/bin/env python3
"""Phase 1.5 schedule ablation decision analysis (PI feedback #7).

Reads lm-eval results from:
  results/phase1p5_schedule_ablation/qwen3-8b/tulu3-sft/<cell>/seed42/lm_eval/
  results/s2_pi5b_v3/qwen3-8b/tulu3-sft/random_dr0.5/seed42/lm_eval/  (reused)
  results/s2_pi5b_v3/qwen3-8b/tulu3-sft/v1_S3pos/seed42/lm_eval/      (reused)

Computes:
  delta_v1_vs_best_random_schedule = v1_gsm8k - max(random_*_gsm8k)

Decision rule (PI #7):
  >= 2.0pp  -> v1 saliency adds value beyond schedule
  [0, 2.0)  -> saliency contributes but weakly
  < 0       -> story flips: schedule drives recovery, not saliency

Writes:
  analysis/results_v3/phase1p5_summary.json
  analysis/COMM_AGENT_TO_PI/{date}_phase1p5_decision.md

Usage:
  python scripts/phase1p5_decision_analysis.py
"""
from __future__ import annotations
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
MODEL   = "qwen3-8b"
DATASET = "tulu3-sft"

PHASE1P5_BASE = ROOT / "results" / "phase1p5_schedule_ablation" / MODEL / DATASET
V3_BASE       = ROOT / "results" / "s2_pi5b_v3" / MODEL / DATASET

NEW_CELLS = [
    "random_anneal_up",
    "random_anneal_down",
    "random_triangle_up_down",
    "random_triangle_down_up",
]
REUSED = {
    "random_const_0p5": V3_BASE / "random_dr0.5" / "seed42",
    "v1_S3pos":         V3_BASE / "v1_S3pos"     / "seed42",
}

METRICS = {
    "gsm8k_strict":   ("gsm8k",        "exact_match,strict-match"),
    "gsm8k_flex":     ("gsm8k",        "exact_match,flexible-extract"),
    "hellaswag":      ("hellaswag",    "acc_norm,none"),
    "arc_challenge":  ("arc_challenge","acc_norm,none"),
    "mmlu":           ("mmlu",         "acc,none"),
    "ifeval":         ("ifeval",       "prompt_level_strict_acc,none"),
}

DECISION_STRONG  = 2.0
DECISION_WEAK    = 0.0


def _load_result(seed_dir: Path) -> dict[str, float]:
    lm_dir  = seed_dir / "lm_eval"
    results = list(lm_dir.rglob("results_*.json")) if lm_dir.exists() else []
    if not results:
        return {m: float("nan") for m in METRICS}
    raw = json.loads(results[0].read_text()).get("results", {})
    out: dict[str, float] = {}
    for metric_key, (task, key) in METRICS.items():
        v = raw.get(task, {}).get(key, float("nan"))
        out[metric_key] = v * 100 if isinstance(v, float) else float("nan")
    return out


def main() -> int:
    all_cells: dict[str, dict[str, float]] = {}

    for cell in NEW_CELLS:
        seed_dir = PHASE1P5_BASE / cell / "seed42"
        all_cells[cell] = _load_result(seed_dir)

    for label, seed_dir in REUSED.items():
        all_cells[label] = _load_result(seed_dir)

    print("\n=== Phase 1.5 schedule ablation scores (seed 42) ===")
    col_w = 14
    header = f"{'cell':30s}" + "".join(f"{m:>{col_w}s}" for m in METRICS)
    print(header)
    print("-" * len(header))

    cell_order = NEW_CELLS + list(REUSED.keys())
    for cell in cell_order:
        scores = all_cells[cell]
        row = f"{cell:30s}"
        for m in METRICS:
            v = scores[m]
            row += f"{'N/A':>{col_w}s}" if math.isnan(v) else f"{v:>{col_w}.2f}"
        print(row)

    v1_gsm = all_cells["v1_S3pos"].get("gsm8k_strict", float("nan"))
    random_cells = [c for c in all_cells if c.startswith("random_")]
    best_random_gsm = max(
        (all_cells[c].get("gsm8k_strict", float("nan")) for c in random_cells),
        default=float("nan")
    )
    best_random_cell = max(
        random_cells,
        key=lambda c: all_cells[c].get("gsm8k_strict", float("-inf"))
    )

    delta = v1_gsm - best_random_gsm if not (math.isnan(v1_gsm) or math.isnan(best_random_gsm)) else float("nan")

    print(f"\nv1_gsm8k_strict      = {v1_gsm:.2f}%")
    print(f"best_random schedule = {best_random_cell}: {best_random_gsm:.2f}%")
    print(f"delta_v1_vs_best     = {delta:+.2f}pp")

    if math.isnan(delta):
        verdict   = "INCOMPLETE"
        interp    = "Missing results; cannot decide."
    elif delta >= DECISION_STRONG:
        verdict   = "SALIENCY_ADDS_VALUE"
        interp    = (f"v1 leads best random schedule by {delta:+.2f}pp >= {DECISION_STRONG}pp. "
                     f"Saliency selection is genuinely informative beyond schedule shape. "
                     f"Paper story as in #6 stands.")
    elif delta >= DECISION_WEAK:
        verdict   = "SALIENCY_WEAKLY_ADDS_VALUE"
        interp    = (f"v1 leads best random schedule by {delta:+.2f}pp (< {DECISION_STRONG}pp). "
                     f"Saliency provides a modest improvement over schedule-matched random. "
                     f"Soften paper claim accordingly.")
    else:
        verdict   = "SCHEDULE_DOMINATES_STORY_FLIP"
        interp    = (f"Best random schedule ({best_random_cell}: {best_random_gsm:.2f}%) beats v1 "
                     f"({v1_gsm:.2f}%) by {-delta:.2f}pp. "
                     f"Drop-rate scheduling drives gsm8k recovery; saliency selection has no "
                     f"measurable benefit. Reposition paper to scheduling contribution.")

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  {interp}")

    summary: dict = {
        "scores": {c: {m: round(v, 4) for m, v in all_cells[c].items()}
                   for c in cell_order},
        "v1_gsm8k_strict": round(v1_gsm, 4),
        "best_random_cell": best_random_cell,
        "best_random_gsm8k_strict": round(best_random_gsm, 4),
        "delta_v1_vs_best_random": round(delta, 4),
        "verdict": verdict,
        "interpretation": interp,
        "ranking": sorted(
            [(c, round(all_cells[c].get("gsm8k_strict", float("nan")), 2))
             for c in cell_order],
            key=lambda x: x[1], reverse=True
        ),
    }

    out_json = ROOT / "analysis" / "results_v3" / "phase1p5_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    _write_decision_md(summary)
    return 0


def _write_decision_md(summary: dict) -> None:
    today = date.today().isoformat()
    comm_dir = ROOT / "analysis" / "COMM_AGENT_TO_PI"
    comm_dir.mkdir(parents=True, exist_ok=True)
    out_md = comm_dir / f"{today}_phase1p5_decision.md"

    verdict = summary["verdict"]
    delta   = summary["delta_v1_vs_best_random"]
    best_c  = summary["best_random_cell"]
    best_v  = summary["best_random_gsm8k_strict"]
    v1_v    = summary["v1_gsm8k_strict"]

    lines = [
        f"# Phase 1.5 Schedule Ablation Decision — {today}",
        "",
        "**ACK_pi_feedback_7_phase1p5_schedule_ablation** (Phase 1.5 results)",
        "",
        f"## Verdict: `{verdict}`",
        "",
        f"> {summary['interpretation']}",
        "",
        "## gsm8k_strict Ranking (seed 42, qwen3-8b/tulu3-sft)",
        "",
        "| rank | cell | gsm8k_strict | gsm_flex | hellaswag | arc_c | mmlu | ifeval |",
        "|------|------|-------------|----------|-----------|-------|------|--------|",
    ]

    scores = summary["scores"]

    def f(cell, m):
        v = scores.get(cell, {}).get(m, float("nan"))
        return "N/A" if math.isnan(v) else f"{v:.2f}"

    ranking = summary["ranking"]
    for rank, (cell, gsm) in enumerate(ranking, 1):
        marker = " **(v1)**" if cell == "v1_S3pos" else \
                 " **(best random)**" if cell == best_c and cell != "v1_S3pos" else ""
        lines.append(
            f"| {rank} | {cell}{marker} | {f(cell,'gsm8k_strict')} | "
            f"{f(cell,'gsm8k_flex')} | {f(cell,'hellaswag')} | "
            f"{f(cell,'arc_challenge')} | {f(cell,'mmlu')} | {f(cell,'ifeval')} |"
        )

    lines += [
        "",
        "## Key Numbers",
        "",
        f"- v1_S3pos gsm8k_strict: **{v1_v:.2f}%**",
        f"- best random schedule ({best_c}): **{best_v:.2f}%**",
        f"- delta_v1_vs_best_random: **{delta:+.2f}pp**",
        "",
        "## Interpretation (per PI #7 decision rule)",
        "",
    ]

    if verdict == "SALIENCY_ADDS_VALUE":
        lines += [
            f"delta={delta:+.2f}pp >= 2.0pp threshold.",
            "v1 saliency selection is genuinely informative beyond drop-rate schedule shape.",
            "Paper story from #6 (saliency closes 53% of gsm8k gap) stands with this ablation",
            "as supporting evidence.",
            "",
            "Recommendation: continue with Phase 2 (olmo2-7b + llama3-8b) as planned.",
        ]
    elif verdict == "SALIENCY_WEAKLY_ADDS_VALUE":
        lines += [
            f"0 <= delta={delta:+.2f}pp < 2.0pp.",
            "Saliency contributes modestly over schedule-matched random.",
            "Soften paper claim to: 'saliency provides a marginal improvement over",
            "  schedule-matched random; drop-rate scheduling is the primary driver.'",
            "",
            "Recommendation: proceed with Phase 2 but report weakened saliency claim.",
        ]
    else:
        lines += [
            f"delta={delta:+.2f}pp < 0: a random schedule beats v1.",
            "Drop-rate scheduling drives gsm8k recovery; saliency selection has no",
            "measurable benefit on this benchmark.",
            "",
            "Recommendation: reposition paper to 'drop-rate scheduling for ReLoRA'.",
            "  Primary contribution = anneal_up schedule outperforms flat random_drop.",
            "  Saliency = open question for future work.",
        ]

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    sys.exit(main())
