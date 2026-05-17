#!/usr/bin/env python
"""F5: active rank vs cumulative rank tracking.

Reads:
- `effective_rank.jsonl`: per eval step, layer -> active effective rank (live LoRA basis)
- `cumulative_rank.jsonl`: per merge event, cumulative_merged_total / cumulative_dropped_total

Plots, per (model, dataset, method):
   x = step
   y left axis: mean active effective rank (across layers, normalized to LoRA r)
   y right axis: cumulative merged components count (running sum of kept across all merges)

The hypothesis is: S3pos should have HIGH cumulative_merged (lots of stuff
absorbed into base) but the active LoRA rank should stay near full r between
merges, while baseline drops nothing -> cumulative_merged grows linearly with
step (= total params * n_merges).

Usage:
    python scripts/analyze_active_vs_cumulative_rank.py \\
        --root results/stage3_v2 \\
        --out  results/stage3_v2/summary/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def mean_eff_rank_per_step(eff_rows: list[dict], lora_rank: int = 16) -> tuple[list[int], list[float]]:
    """Returns (steps, mean_eff_rank_normalized)."""
    steps = []; vals = []
    for row in eff_rows:
        per_layer = row.get("per_layer_effective_rank") or row.get("effective_ranks")
        if not per_layer:
            # alternative key naming
            for k in ("ranks", "rank_per_layer"):
                if k in row:
                    per_layer = row[k]
                    break
        if not per_layer:
            continue
        if isinstance(per_layer, dict):
            ranks = list(per_layer.values())
        else:
            ranks = list(per_layer)
        ranks = [float(r) for r in ranks if r is not None]
        if not ranks:
            continue
        steps.append(int(row.get("step", -1)))
        vals.append(np.mean(ranks) / lora_rank)
    return steps, vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/stage3_v2"))
    ap.add_argument("--out",  type=Path, default=Path("results/stage3_v2/summary/figures"))
    ap.add_argument("--lora_rank", type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    runs = sorted(args.root.glob("*/*/*/seed*"))
    by_md: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for rd in runs:
        rel = rd.relative_to(args.root)
        if len(rel.parts) < 4:
            continue
        model, dataset, method, _seed = rel.parts[:4]
        eff = read_jsonl(rd / "effective_rank.jsonl")
        cum = read_jsonl(rd / "cumulative_rank.jsonl")
        if not eff and not cum:
            continue
        by_md[(model, dataset)][method] = {
            "eff": eff,
            "cum": cum,
        }

    n_done = 0
    for (model, dataset), methods_map in by_md.items():
        if not methods_map:
            continue
        fig, ax_left = plt.subplots(figsize=(10, 5))
        ax_right = ax_left.twinx()
        cmap = plt.cm.tab10
        colors = {m: cmap(i) for i, m in enumerate(sorted(methods_map))}

        for method, data in sorted(methods_map.items()):
            eff_steps, eff_vals = mean_eff_rank_per_step(data["eff"], args.lora_rank)
            if eff_steps:
                ax_left.plot(eff_steps, eff_vals, "-", color=colors[method],
                             label=f"{method} (eff rank /r)", alpha=0.8)
            cum = data["cum"]
            if cum:
                cs = [int(r["step"]) for r in cum]
                cm = [int(r.get("cumulative_merged_total", 0)) for r in cum]
                ax_right.plot(cs, cm, "--", color=colors[method], alpha=0.6,
                              label=f"{method} (cum merged)")

        ax_left.set_xlabel("step")
        ax_left.set_ylabel("mean effective rank / LoRA r (solid)")
        ax_right.set_ylabel("cumulative merged components (dashed)")
        ax_left.set_title(f"{model} / {dataset}: active rank vs cumulative merged")
        ax_left.legend(loc="upper left", fontsize=7)
        ax_right.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        out_path = args.out / f"{model}_{dataset}_active_vs_cumulative_rank.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  -> {out_path}")
        n_done += 1

    print(f"\nrendered {n_done} figures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
