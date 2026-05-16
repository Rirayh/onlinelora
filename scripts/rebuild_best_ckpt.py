#!/usr/bin/env python
"""Rebuild `checkpoints/best/` using ONLY non-post-merge val_loss entries.

Background (silent bug fix):
  Stage 3 v2 originally updated `best_val_loss` on every val eval, including
  the *post-merge* eval right after a ReLoRA merge step. At that instant,
  `lora_B` has just been zeroed (the trained delta is now in the frozen base
  weights), so the persisted PEFT adapter is numerically the identity. Any
  downstream lm-eval would only reflect the base model.

This script scans every `results/stage3_v2/<model>/<ds>/<method>/seed*/`,
picks the lowest `val_loss` among rows that DO NOT carry `post_merge: True`
(and DO NOT carry `final: True` either, because final adapter equals
checkpoints/final and may also be post-merge), then copies the matching
`checkpoints/step_<best_step>/` -> `checkpoints/best/`.

If no such row exists (e.g. ReLoRA method that only logs post-merge evals at
exactly merge_every == eval_every), the run is flagged as `BEST_AMBIGUOUS`.

Usage:
    python scripts/rebuild_best_ckpt.py --root results/stage3_v2 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def pick_pre_merge_best(val_loss_path: Path) -> tuple[int, float] | None:
    rows = []
    with val_loss_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    pre = [r for r in rows
           if not r.get("post_merge") and not r.get("final")
           and "val_loss" in r and "step" in r]
    if not pre:
        return None
    best = min(pre, key=lambda r: r["val_loss"])
    return int(best["step"]), float(best["val_loss"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/stage3_v2"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    seed_dirs = sorted(root.glob("*/*/*/seed*"))
    print(f"scanning {len(seed_dirs)} run dirs under {root}\n")

    fixed = []
    same = []
    ambiguous = []
    missing_step = []

    for d in seed_dirs:
        vl_path = d / "val_loss.jsonl"
        ckpt_root = d / "checkpoints"
        best_dir = ckpt_root / "best"
        if not vl_path.exists() or not ckpt_root.exists():
            continue

        pick = pick_pre_merge_best(vl_path)
        if pick is None:
            ambiguous.append(str(d.relative_to(root)))
            continue
        step, val = pick

        src = ckpt_root / f"step_{step:06d}"
        if not src.exists():
            missing_step.append((str(d.relative_to(root)), step))
            continue

        # Read existing meta if any
        meta_old = {}
        old_meta_p = best_dir / "meta.json"
        if old_meta_p.exists():
            try:
                meta_old = json.loads(old_meta_p.read_text())
            except json.JSONDecodeError:
                pass

        old_step = meta_old.get("step")
        old_was_post_merge = bool(meta_old.get("post_merge"))

        if old_step == step and not old_was_post_merge:
            same.append(str(d.relative_to(root)))
            continue

        rel = d.relative_to(root)
        print(f"{rel}: best old=(step={old_step},post_merge={old_was_post_merge}) "
              f"-> new=(step={step}, val={val:.4f})")
        if args.dry_run:
            fixed.append(str(rel))
            continue

        # Replace best/ atomically: write to .new, swap.
        new_dir = ckpt_root / "best.new"
        if new_dir.exists():
            shutil.rmtree(new_dir)
        shutil.copytree(src, new_dir)
        meta_new = {
            "step": step,
            "val_loss": val,
            "post_merge": False,
            "rebuilt_from": f"step_{step:06d}",
            "previous_meta": meta_old,
        }
        (new_dir / "meta.json").write_text(json.dumps(meta_new, indent=2))
        if best_dir.exists():
            shutil.rmtree(best_dir)
        new_dir.rename(best_dir)
        fixed.append(str(rel))

    print()
    print(f"fixed     : {len(fixed)}")
    print(f"unchanged : {len(same)}")
    print(f"ambiguous : {len(ambiguous)}")
    if ambiguous:
        for a in ambiguous: print(f"  - {a}")
    print(f"missing step ckpt: {len(missing_step)}")
    if missing_step:
        for d, s in missing_step:
            print(f"  - {d}: step_{s:06d} not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
