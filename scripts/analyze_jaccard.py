#!/usr/bin/env python
"""F5: pairwise Jaccard similarity of dropped-component sets across methods.

For each (model, dataset) pair, finds all ReLoRA-style methods (those that have
`dropped_components.jsonl`) and computes, at each merge event, the Jaccard
similarity J(A,B) = |drop_A ∩ drop_B| / |drop_A ∪ drop_B| of the dropped LoRA
components between every pair of methods.

The "dropped set" is implicit from per_layer_keep_counts: for each layer X with
rank R, if kept count is K, it dropped (R - K) components but the IDENTITY of
which components is NOT in the JSON (only counts). So we approximate by
treating per-layer drop-COUNT vector as a feature, and use cosine on those
vectors as the similarity metric, AND a "drop pattern Jaccard" defined on
per-layer binary mask "this layer had any drop".

Output: one heatmap PNG per (model, dataset) at each merge event, plus a CSV
summary.

Usage:
    python scripts/analyze_jaccard.py \\
        --root results/stage3_v2 \\
        --out  results/stage3_v2/summary/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYER_RE = re.compile(r"layers\.(\d+)\.([^.]+)\.([qkvo]_proj|gate_proj|up_proj|down_proj)")


def load_drop_events(run_dir: Path, lora_rank: int = 16) -> list[dict]:
    """Returns list of merge events. Each event is dict
       {merge_event:int, step:int, drop_count_per_module: {fullname: int_dropped}}
    """
    fp = run_dir / "dropped_components.jsonl"
    if not fp.exists():
        return []
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
    out = []
    for row in rows:
        drop = {}
        for name, kept in row.get("per_layer_keep_counts", {}).items():
            drop[name] = lora_rank - int(kept)
        out.append({
            "merge_event": int(row.get("merge_event", -1)),
            "step": int(row.get("step", -1)),
            "drop_count_per_module": drop,
        })
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard_binary(a: np.ndarray, b: np.ndarray, thresh: int = 1) -> float:
    """Jaccard on binary masks: module x got at least `thresh` drops."""
    A = (a >= thresh).astype(np.int64)
    B = (b >= thresh).astype(np.int64)
    inter = int((A & B).sum())
    union = int((A | B).sum())
    return inter / union if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/stage3_v2"))
    ap.add_argument("--out",  type=Path, default=Path("results/stage3_v2/summary/figures"))
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--csv",  type=Path, default=Path("results/stage3_v2/summary/jaccard.csv"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    runs = sorted(args.root.glob("*/*/*/seed*/dropped_components.jsonl"))
    by_md: dict[tuple, dict[str, list[dict]]] = defaultdict(dict)
    for fp in runs:
        rd = fp.parent
        rel = rd.relative_to(args.root)
        if len(rel.parts) < 4:
            continue
        model, dataset, method, _seed = rel.parts[:4]
        events = load_drop_events(rd, args.lora_rank)
        if events:
            by_md[(model, dataset)][method] = events

    csv_rows = []
    for (model, dataset), methods_map in by_md.items():
        method_names = sorted(methods_map.keys())
        if len(method_names) < 2:
            continue
        # Determine common module ordering across all methods at first event
        module_order = None
        for m in method_names:
            mods = set(methods_map[m][0]["drop_count_per_module"].keys())
            if module_order is None:
                module_order = mods
            else:
                module_order = module_order & mods
        if not module_order:
            continue
        module_order = sorted(module_order)

        # number of merge events to render (use min across methods)
        n_events = min(len(methods_map[m]) for m in method_names)
        for ei in range(n_events):
            vecs = {}
            for m in method_names:
                ev = methods_map[m][ei]
                v = np.array([ev["drop_count_per_module"].get(k, 0) for k in module_order],
                             dtype=np.int64)
                vecs[m] = v
            # build pairwise jaccard + cosine matrices
            n = len(method_names)
            jac = np.zeros((n, n))
            cos = np.zeros((n, n))
            for i, mi in enumerate(method_names):
                for j, mj in enumerate(method_names):
                    jac[i, j] = jaccard_binary(vecs[mi], vecs[mj], thresh=1)
                    cos[i, j] = cosine_sim(vecs[mi].astype(np.float64),
                                           vecs[mj].astype(np.float64))
                    csv_rows.append({
                        "model": model, "dataset": dataset,
                        "merge_event": ei + 1,
                        "method_a": mi, "method_b": mj,
                        "jaccard_binary": f"{jac[i,j]:.4f}",
                        "cosine":         f"{cos[i,j]:.4f}",
                    })

            # render dual-panel figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for ax, mat, title in zip(axes, [jac, cos],
                                      ["Jaccard (binary, thresh=1)", "Cosine (drop-count vec)"]):
                im = ax.imshow(mat, cmap="magma", vmin=0, vmax=1)
                ax.set_xticks(range(n)); ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(n)); ax.set_yticklabels(method_names, fontsize=7)
                ax.set_title(title, fontsize=10)
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                                color="w" if mat[i,j] < 0.5 else "k", fontsize=6)
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"{model} / {dataset}  --  merge event #{ei+1}", fontsize=11)
            fig.tight_layout()
            out_name = f"{model}_{dataset}_jaccard_event{ei+1:02d}.png"
            fig.savefig(args.out / out_name, dpi=120)
            plt.close(fig)
            print(f"  -> {args.out / out_name}")

    if csv_rows:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"\nwrote {len(csv_rows)} rows to {args.csv}")
    print(f"\nrendered figures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
