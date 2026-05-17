#!/usr/bin/env python
"""F5: drop-rate heatmap per (method, layer, merge_event).

For each run, reads `dropped_components.jsonl`, builds a 2D grid:
   x = merge_event (1..N)
   y = layer index (0..L-1) aggregated across qkvo/gate_up_down per layer
   z = sum of (rank - kept) / rank = drop rate per layer at that merge event

Saves one PNG per method showing how drop pattern evolves across merge events.

Usage:
    python scripts/analyze_drop_heatmap.py \\
        --root results/stage3_v2 \\
        --out  results/stage3_v2/summary/figures
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def parse_layer_idx(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def heatmap_for_run(run_dir: Path, lora_rank: int = 16) -> np.ndarray | None:
    """Returns 2D array shape (n_merge, n_layers) of drop rate per layer."""
    fp = run_dir / "dropped_components.jsonl"
    if not fp.exists():
        return None
    rows = []
    with fp.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return None
    n_merge = len(rows)
    layers_seen: dict[int, list[int]] = defaultdict(list)
    n_modules_per_layer: dict[int, int] = defaultdict(int)

    # First pass to determine layer set
    layer_set: set[int] = set()
    for row in rows:
        for name in row.get("per_layer_keep_counts", {}).keys():
            li = parse_layer_idx(name)
            if li is not None:
                layer_set.add(li)
    if not layer_set:
        return None
    n_layers = max(layer_set) + 1

    grid = np.full((n_merge, n_layers), np.nan, dtype=np.float64)

    for mi, row in enumerate(rows):
        per_layer_drop_total = defaultdict(int)
        per_layer_max_total = defaultdict(int)
        for name, kept in row.get("per_layer_keep_counts", {}).items():
            li = parse_layer_idx(name)
            if li is None:
                continue
            per_layer_drop_total[li] += (lora_rank - kept)
            per_layer_max_total[li] += lora_rank
        for li in layer_set:
            if per_layer_max_total[li] > 0:
                grid[mi, li] = per_layer_drop_total[li] / per_layer_max_total[li]
    return grid


def render_heatmap(grid: np.ndarray, title: str, out_path: Path, lora_rank: int):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(grid.T, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0.0, vmax=1.0)
    ax.set_xlabel("merge event #")
    ax.set_ylabel("layer index")
    ax.set_title(f"{title}\nfraction of LoRA components dropped per layer (rank={lora_rank})")
    fig.colorbar(im, ax=ax, label="drop rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/stage3_v2"))
    ap.add_argument("--out",  type=Path, default=Path("results/stage3_v2/summary/figures"))
    ap.add_argument("--lora_rank", type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = sorted(args.root.glob("*/*/*/seed*/dropped_components.jsonl"))
    print(f"found {len(runs)} runs with dropped_components.jsonl")
    n_done = 0
    for fp in runs:
        run_dir = fp.parent
        rel = run_dir.relative_to(args.root)
        title = "/".join(rel.parts[:3])
        grid = heatmap_for_run(run_dir, lora_rank=args.lora_rank)
        if grid is None:
            print(f"  SKIP {rel} (no data)")
            continue
        # one png per (model, dataset, method)
        out_name = "_".join(rel.parts[:3]).replace("/", "_") + "_drop_heatmap.png"
        out_path = args.out / out_name
        render_heatmap(grid, title, out_path, args.lora_rank)
        print(f"  -> {out_path} (shape {grid.shape})")
        n_done += 1

    print(f"\nrendered {n_done} heatmaps to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
