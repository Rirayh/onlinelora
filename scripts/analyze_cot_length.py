"""
F5: analyze_cot_length.py

Parse lm-eval `samples_*.jsonl` (produced when running lm-eval with --log_samples)
to compute mean / median / p95 generation length per (model, dataset, method, task).

Skips silently any cell without samples (lm-eval was run without --log_samples).

Output:
  results/stage3_v2/summary/cot_length.csv  (one row per cell+task)
  results/stage3_v2/summary/figures/{model}_{dataset}_cot_length_{task}.png
    bar chart of mean tokens per method
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "stage3_v2"
SUMMARY = RESULTS / "summary"
FIGS = SUMMARY / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

EXCLUDE_DIRS = {"summary", "figures"}


def open_samples(p: Path):
    if p.suffix == ".gz":
        return gzip.open(p, "rt")
    return p.open("r")


def gen_tokens(rec: dict) -> int | None:
    """Best-effort estimation of generation length in characters or tokens."""
    # lm-eval-harness samples_*.jsonl typically has:
    #   "resps": [["..."]], "filtered_resps": ["..."], "target": "...", "doc": ...
    resps = rec.get("filtered_resps") or rec.get("resps")
    if not resps:
        return None
    if isinstance(resps, list):
        # could be [[str]] or [str]
        first = resps[0]
        if isinstance(first, list) and first:
            text = first[0]
        elif isinstance(first, str):
            text = first
        else:
            return None
        # whitespace-token count is fast and consistent enough
        return len(text.split())
    return None


def find_task_from_path(p: Path) -> str | None:
    # files like: samples_gsm8k_2026-...jsonl[.gz]
    m = re.match(r"samples_([A-Za-z0-9_]+)_\d{4}", p.name)
    return m.group(1) if m else None


def discover_cells():
    out = []
    for model_dir in sorted(RESULTS.iterdir()):
        if not model_dir.is_dir() or model_dir.name in EXCLUDE_DIRS:
            continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for method_dir in sorted(dataset_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                for seed_dir in sorted(method_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                        continue
                    out.append({
                        "model": model_dir.name,
                        "dataset": dataset_dir.name,
                        "method": method_dir.name,
                        "seed": seed_dir.name.replace("seed", ""),
                        "lm_eval_dir": seed_dir / "lm_eval",
                    })
    return out


def gather_lengths(cell: dict) -> dict:
    """Return {task: [n_tokens, ...]}."""
    out: dict[str, list[int]] = defaultdict(list)
    if not cell["lm_eval_dir"].exists():
        return out
    for samples in cell["lm_eval_dir"].rglob("samples_*.jsonl*"):
        task = find_task_from_path(samples)
        if not task:
            continue
        try:
            with open_samples(samples) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    n = gen_tokens(rec)
                    if n is not None:
                        out[task].append(n)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=Path, default=SUMMARY / "cot_length.csv")
    args = ap.parse_args()

    cells = discover_cells()
    rows = []
    by_md_task: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)

    for c in cells:
        per_task = gather_lengths(c)
        for task, lens in per_task.items():
            arr = np.asarray(lens, dtype=float)
            row = {
                "model": c["model"], "dataset": c["dataset"],
                "method": c["method"], "seed": c["seed"], "task": task,
                "n_samples": int(arr.size),
                "mean_tokens": float(arr.mean()) if arr.size else None,
                "median_tokens": float(np.median(arr)) if arr.size else None,
                "p95_tokens": float(np.percentile(arr, 95)) if arr.size else None,
            }
            rows.append(row)
            by_md_task[(c["model"], c["dataset"], task)][c["method"]] = row

    if not rows:
        print("no samples_*.jsonl found anywhere; lm-eval was probably run without --log_samples")
    else:
        import csv
        cols = ["model", "dataset", "method", "seed", "task",
                "n_samples", "mean_tokens", "median_tokens", "p95_tokens"]
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"wrote {args.out_csv} ({len(rows)} rows)")

    # plot per (model, dataset, task)
    for (model, dataset, task), per_method in sorted(by_md_task.items()):
        names = sorted(per_method.keys())
        means = [per_method[m]["mean_tokens"] or 0 for m in names]
        if not means:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(names) + 2), 4))
        ax.bar(range(len(names)), means)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("mean generated tokens")
        ax.set_title(f"{model} / {dataset} / {task}")
        fig.tight_layout()
        out = FIGS / f"{model}_{dataset}_cot_length_{task}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
