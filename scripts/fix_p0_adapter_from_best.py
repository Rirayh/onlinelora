#!/usr/bin/env python3
"""
P0 retroactive fix: copy seed42/ckpt/best/ -> seed42/adapter/ for all
merge-based methods (relora_*, S3pos, S3neg, random_drop, train_gated, cola)
whose currently-saved adapter has lora_B == 0.

Triggered by stage3_run.py bug: end-of-training save_pretrained captures the
post-final-merge state where lora_B has just been zeroed -> downstream
lm-eval is numerically identical to base model.

Usage:
    python scripts/fix_p0_adapter_from_best.py            # dry run (default)
    python scripts/fix_p0_adapter_from_best.py --apply    # actually copy
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import safetensors.torch as st

MERGE_METHODS = {
    "relora_baseline",
    "relora_diag_gated_S3pos",
    "relora_diag_gated_S3neg",
    "relora_random_drop",
    "relora_train_gated",
    "cola",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "stage3_v2"


def adapter_b_is_zero(adapter_dir: Path) -> bool | None:
    """Return True if all lora_B params are zero, False if any nonzero, None if unknown."""
    f = adapter_dir / "adapter_model.safetensors"
    if not f.exists():
        return None
    try:
        sd = st.load_file(str(f))
    except Exception as e:
        print(f"  [WARN] cannot load {f}: {e}")
        return None
    b_keys = [k for k in sd if "lora_B" in k]
    if not b_keys:
        return None
    for k in b_keys:
        if float(sd[k].abs().max().item()) > 0.0:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually copy best/->adapter/ (default: dry run)")
    args = ap.parse_args()

    if not RESULTS_ROOT.exists():
        print(f"ERR: {RESULTS_ROOT} not found", file=sys.stderr)
        sys.exit(1)

    candidates: list[tuple[Path, Path, Path]] = []  # (cell_root, adapter, best)
    for model_dir in sorted(RESULTS_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_") or model_dir.name == "summary":
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for method_dir in sorted(ds_dir.iterdir()):
                if not method_dir.is_dir() or method_dir.name not in MERGE_METHODS:
                    continue
                for seed_dir in sorted(method_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    if seed_dir.name.endswith("_smoke"):
                        continue
                    adapter_dir = seed_dir / "adapter"
                    best_dir = seed_dir / "checkpoints" / "best"
                    if not adapter_dir.exists():
                        continue
                    candidates.append((seed_dir, adapter_dir, best_dir))

    print(f"Scanned {len(candidates)} merge-method cells with adapter/ saved")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print("-" * 80)

    fixed = 0
    skipped_ok = 0
    no_best = 0
    for seed_dir, adapter_dir, best_dir in candidates:
        rel = seed_dir.relative_to(RESULTS_ROOT)
        is_zero = adapter_b_is_zero(adapter_dir)
        if is_zero is False:
            print(f"[OK]      {rel}  (lora_B != 0, leave alone)")
            skipped_ok += 1
            continue
        if is_zero is None:
            print(f"[?]       {rel}  (no safetensors / unreadable)")
            continue
        # is_zero == True -> need fix
        if not best_dir.exists() or not (best_dir / "adapter_model.safetensors").exists():
            print(f"[NO-BEST] {rel}  (lora_B=0 but no best/ ckpt -> needs RETRAIN)")
            no_best += 1
            continue
        best_zero = adapter_b_is_zero(best_dir)
        if best_zero is True:
            print(f"[BEST=0]  {rel}  (best/ also has B=0 -> needs RETRAIN)")
            no_best += 1
            continue
        if args.apply:
            shutil.rmtree(adapter_dir)
            shutil.copytree(str(best_dir), str(adapter_dir))
            print(f"[FIXED]   {rel}  best/ -> adapter/")
        else:
            print(f"[WOULD]   {rel}  best/ -> adapter/")
        fixed += 1

    print("-" * 80)
    print(f"Summary: fixed={fixed}  ok_already={skipped_ok}  needs_retrain={no_best}")


if __name__ == "__main__":
    main()
