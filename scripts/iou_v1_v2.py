"""v1<->v2 dropped-component IoU analysis (PI feedback #3 §3).

Reads `merge_events.jsonl` from two runs and computes per-event,
per-layer-type IoU on the dropped-component sets.

Output: TSV with columns
    event_idx | layer_type | n_components | n_v1_drop | n_v2_drop | iou | jaccard_dist

Usage:
    python scripts/iou_v1_v2.py \
        --v1 results/s2_v1_recheck/qwen3-8b/tulu3-sft/relora_diag_gated_S3pos_v1_recheck/seed42/merge_events.jsonl \
        --v2 results/s2/qwen3-8b/tulu3-sft/v2_full/seed42/merge_events.jsonl \
        --out analysis/results_v3/v1_v2_iou.tsv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


LAYER_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj",
               "up_proj", "gate_proj", "down_proj"]


def load_events(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def layer_type_of(layer_name: str) -> str:
    """Extract proj type from a layer name like 'model.layers.0.self_attn.q_proj'."""
    for t in LAYER_TYPES:
        if t in layer_name:
            return t
    return "other"


def build_drop_set(event: dict) -> set[tuple[str, int]]:
    ids = event.get("dropped_component_ids", [])
    return {(L, int(i)) for L, i in ids}


def iou(a: set, b: set) -> tuple[float, float]:
    if not a and not b:
        return float("nan"), float("nan")
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return float("nan"), float("nan")
    j = inter / union
    return j, 1.0 - j


def per_layer_breakdown(drop_set: set[tuple[str, int]]) -> dict[str, set]:
    out = defaultdict(set)
    for L, i in drop_set:
        out[layer_type_of(L)].add((L, i))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    v1_events = load_events(Path(args.v1))
    v2_events = load_events(Path(args.v2))

    n = min(len(v1_events), len(v2_events))
    rows = []
    rows.append(["event_idx", "layer_type", "n_components",
                 "n_v1_drop", "n_v2_drop", "iou", "jaccard_dist"])

    for i in range(n):
        e1, e2 = v1_events[i], v2_events[i]
        d1 = build_drop_set(e1)
        d2 = build_drop_set(e2)
        if not d1 or not d2:
            print(f"[WARN] event {i}: missing dropped_component_ids "
                  f"(v1={len(d1)}, v2={len(d2)}), skipping")
            continue

        # ALL row
        n_total = len({L for L, _ in d1} | {L for L, _ in d2})  # unique layers
        # n_components for ALL = total component count = sum of ranks; we
        # approximate from union of dropped, which is a lower bound. The
        # caller can also pass --n_total for exact value if needed.
        j, jd = iou(d1, d2)
        rows.append([i, "ALL", "—", len(d1), len(d2),
                     f"{j:.4f}", f"{jd:.4f}"])

        # Per-layer-type
        b1 = per_layer_breakdown(d1)
        b2 = per_layer_breakdown(d2)
        for lt in LAYER_TYPES:
            s1 = b1.get(lt, set())
            s2 = b2.get(lt, set())
            if not s1 and not s2:
                continue
            j, jd = iou(s1, s2)
            rows.append([i, lt, "—", len(s1), len(s2),
                         f"{j:.4f}", f"{jd:.4f}"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"WROTE {out_path} ({len(rows)-1} data rows)")

    print("\nIoU interpretation per PI #3 §3:")
    print("  0.45-0.55 random-equivalent (strong claim: IG creates real signal)")
    print("  0.30-0.45 weak overlap (moderate claim)")
    print("  0.55-0.70 partial overlap (sharpens v1)")
    print("  > 0.70    high overlap (weak claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
