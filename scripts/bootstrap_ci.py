#!/usr/bin/env python
"""F4: bootstrap 95% CI on every (model, dataset, method, task) cell.

Reads `samples_<task>_<timestamp>.jsonl` files produced by lm-eval with
`--log_samples`. Each line is a single doc; the metric is encoded in a key on
that doc (e.g. `exact_match`, `acc`, `acc_norm`, `pass@1`, ...).

Aggregates over `results/stage3_v2/<model_key>/<dataset_key>/<method>/seed*/lm_eval/<task>/**/samples_*.jsonl`
and emits a CSV at `results/stage3_v2/summary/bootstrap_ci.csv`.

Usage:
    python scripts/bootstrap_ci.py \\
        --root results/stage3_v2 \\
        --out  results/stage3_v2/summary/bootstrap_ci.csv \\
        [--n_boot 1000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Per-task metric keys to extract from each sample row. lm-eval encodes the
# scalar value of the metric directly on the doc dict.
TASK_METRICS: dict[str, list[str]] = {
    "gsm8k":          ["exact_match", "exact_match,strict-match", "exact_match,flexible-extract"],
    "mmlu":           ["acc"],
    "mmlu_pro":       ["acc"],
    "bbh":            ["acc"],
    "hendrycks_math": ["exact_match"],
    "math_hendrycks": ["exact_match"],   # alias
    "humaneval":      ["pass@1"],
    "ifeval":         ["prompt_level_strict_acc", "inst_level_strict_acc"],
    "truthfulqa_mc1": ["acc"],
    "hellaswag":      ["acc_norm", "acc"],
    "arc_challenge":  ["acc_norm", "acc"],
}

SAMPLES_RE = re.compile(r"^samples_(?P<task>[A-Za-z0-9_\-]+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$")


def find_sample_files(root: Path) -> list[Path]:
    """Return all log_samples jsonls under root."""
    out = []
    for p in root.rglob("samples_*.jsonl"):
        if SAMPLES_RE.match(p.name):
            out.append(p)
    return out


def parse_path(p: Path, root: Path) -> dict[str, str] | None:
    """Extract (model_key, dataset, method, seed, task) from path."""
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    # Expected layout under results/stage3_v2/:
    #   <model_key>/<dataset>/<method>/seed42/lm_eval/<task>/.../samples_*.jsonl
    # Some legacy paths may have lm_eval directly:
    #   <model_key>/<dataset>/<method>/seed42/lm_eval/.../samples_*.jsonl
    if len(parts) < 5:
        return None
    model_key, dataset, method, seed = parts[:4]
    if not seed.startswith("seed"):
        return None
    m = SAMPLES_RE.match(p.name)
    task = m.group("task") if m else "unknown"
    return {
        "model_key": model_key,
        "dataset":   dataset,
        "method":    method,
        "seed":      seed,
        "task":      task,
    }


def extract_metric(rows: Iterable[dict], task: str) -> tuple[list[float], str | None]:
    """Pick the first available metric key for the task and return (values, metric_name)."""
    keys = TASK_METRICS.get(task, [])
    sample_rows = list(rows)
    if not sample_rows:
        return [], None
    for k in keys:
        if k in sample_rows[0]:
            vals = []
            for r in sample_rows:
                v = r.get(k)
                if v is None:
                    continue
                if isinstance(v, bool):
                    vals.append(float(v))
                elif isinstance(v, (int, float)):
                    vals.append(float(v))
            return vals, k
    # Fallback: scan all keys for any scalar that looks metric-like.
    candidates = [k for k, v in sample_rows[0].items()
                  if isinstance(v, (int, float, bool))
                  and k not in {"doc_id"}]
    if candidates:
        k = candidates[0]
        vals = [float(r[k]) for r in sample_rows if k in r and isinstance(r[k], (int, float, bool))]
        return vals, k
    return [], None


def bootstrap_ci(values: list[float], n_boot: int, seed: int,
                 q_lo: float = 2.5, q_hi: float = 97.5) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    return (
        float(arr.mean()),
        float(np.percentile(boots, q_lo)),
        float(np.percentile(boots, q_hi)),
        int(n),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("results/stage3_v2"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/stage3_v2/summary/bootstrap_ci.csv"))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 1

    files = find_sample_files(root)
    print(f"found {len(files)} samples_*.jsonl files under {root}")
    if not files:
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []

    for f in sorted(files):
        meta = parse_path(f, root)
        if meta is None:
            print(f"  SKIP unparseable path: {f}")
            continue
        try:
            with f.open() as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        except (OSError, json.JSONDecodeError) as e:
            print(f"  SKIP read error {f}: {e}")
            continue
        vals, metric_name = extract_metric(rows, meta["task"])
        if not vals:
            print(f"  SKIP no metric ({meta['task']}): {f}")
            continue
        mean, lo, hi, n = bootstrap_ci(vals, args.n_boot, args.seed)
        rec = {
            **meta,
            "metric":      metric_name,
            "mean":        f"{mean:.6f}",
            "ci_lo_2.5":   f"{lo:.6f}",
            "ci_hi_97.5":  f"{hi:.6f}",
            "ci_width":    f"{hi - lo:.6f}",
            "n_samples":   n,
            "samples_file": str(f.relative_to(root)),
        }
        rows_out.append(rec)
        print(f"  {meta['model_key']}/{meta['dataset']}/{meta['method']}/{meta['task']:<14} "
              f"mean={mean:.4f} ci=[{lo:.4f},{hi:.4f}] n={n}")

    if not rows_out:
        print("ERROR: no rows aggregated", file=sys.stderr)
        return 1

    cols = ["model_key", "dataset", "method", "seed", "task", "metric",
            "mean", "ci_lo_2.5", "ci_hi_97.5", "ci_width", "n_samples", "samples_file"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nwrote {len(rows_out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
