#!/usr/bin/env python3
"""find_p0_tainted_evals.py — list cells whose lm_eval/ output is tainted by P0 bug.

P0 bug (commit b7d07dc fix at 2026-05-19 15:45 UTC):
  Merge-method adapters had lora_B=0 saved at end-of-training because the
  final merge zeroed B and `model.save_pretrained()` was called right after.
  Any lm_eval result computed against such an adapter equals the BASE MODEL
  score, not the trained adapter.

Fix b7d07dc: copy checkpoints/best/ -> adapter/ at end-of-training; the retro
cleanup script `scripts/fix_p0_adapter_from_best.py` was run that day to
patch all 14 contaminated cells' adapter/ dirs.

A cell is TAINTED iff:
  - method is a merge method (relora_*, cola), AND
  - it has at least one results_*.json under lm_eval*/ with mtime BEFORE
    2026-05-19 15:45 UTC (the fix landed at that point — note adapter was
    fixed retroactively but old eval JSONs were never rerun)
  - AND there is no post-fix vLLM-on-merged result (tagged lm_eval_v3) that
    would supersede it.

This script outputs a JSON manifest plus markdown table.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "stage3_v2"

# fix landed at this UTC timestamp; lm_eval JSONs older than this against a
# merge-method adapter MAY have been computed against a B=0 adapter
FIX_DT = datetime(2026, 5, 19, 15, 45, 0, tzinfo=timezone.utc)
FIX_TS = FIX_DT.timestamp()

# methods whose adapter is built via merge-and-reset (B was zeroed at final merge)
MERGE_METHODS = {
    "relora_baseline", "relora_random_drop", "relora_train_gated",
    "relora_diag_gated_S3pos", "relora_diag_gated_S3neg",
    "relora_diag_gated_S3pos_keepB",  # future variant
    "cola",
}
# methods whose adapter is intact (no merges)
SAFE_METHODS = {"lora_vanilla", "dora", "adalora"}

PI_TARGET_MODELS = {"mistral-7b", "qwen25-7b", "qwen3-8b"}


def find_eval_jsons(seed_dir: Path) -> list[Path]:
    """Return all results_*.json under any lm_eval*/ subtree of seed_dir."""
    out: list[Path] = []
    for sub in seed_dir.iterdir() if seed_dir.is_dir() else []:
        if sub.is_dir() and sub.name.startswith("lm_eval"):
            for f in sub.rglob("results_*.json"):
                out.append(f)
    return out


def file_mtime(p: Path) -> float:
    return p.stat().st_mtime


def parse_results_dt_from_filename(p: Path) -> float | None:
    """Filenames look like results_2026-05-15T16-31-29.579778.json — parse to ts."""
    name = p.name
    if not name.startswith("results_") or not name.endswith(".json"):
        return None
    s = name[len("results_"):-len(".json")]
    # "2026-05-15T16-31-29.579778"
    try:
        date_part, time_part = s.split("T", 1)
        h, m, rest = time_part.split("-", 2)
        if "." in rest:
            sec, _us = rest.split(".", 1)
        else:
            sec = rest
        dt = datetime.strptime(f"{date_part} {h}:{m}:{sec}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def best_ts(p: Path) -> float:
    """Use filename timestamp if parseable else mtime."""
    ts = parse_results_dt_from_filename(p)
    return ts if ts is not None else file_mtime(p)


def cell_status(model: str, dataset: str, method: str, seed_dir: Path) -> dict:
    info = {
        "model": model, "dataset": dataset, "method": method,
        "seed_dir": str(seed_dir.relative_to(ROOT)),
        "is_merge_method": method in MERGE_METHODS,
        "has_summary": (seed_dir / "summary.json").exists(),
        "has_adapter": (seed_dir / "adapter" / "adapter_model.safetensors").exists(),
        "has_best_ckpt": (seed_dir / "checkpoints" / "best" / "adapter_model.safetensors").exists(),
        "evals": [],
        "is_tainted": False,
        "needs_v3_reeval": False,
        "needs_vllm_v3": False,
    }
    for f in find_eval_jsons(seed_dir):
        ts = best_ts(f)
        rel = str(f.relative_to(ROOT))
        is_pre = ts < FIX_TS
        is_v3 = "/lm_eval_v3/" in rel
        # peek backend (vllm vs hf) by reading model field
        backend = "?"
        try:
            d = json.loads(f.read_text())
            backend = d.get("config", {}).get("model", "?")
        except Exception:
            pass
        info["evals"].append({
            "path": rel,
            "ts": ts,
            "pre_fix": is_pre,
            "is_v3": is_v3,
            "backend": backend,
        })
    has_post_fix_vllm_v3 = any(
        e["is_v3"] and not e["pre_fix"] and e["backend"] == "vllm" for e in info["evals"]
    )
    has_pre_fix = any(e["pre_fix"] for e in info["evals"])
    info["is_tainted"] = info["is_merge_method"] and has_pre_fix
    info["needs_v3_reeval"] = (
        info["is_merge_method"]
        and (info["has_adapter"] or info["has_best_ckpt"])
        and not any(e["is_v3"] and not e["pre_fix"] for e in info["evals"])
    )
    # PI's stricter requirement: all merge-method cells must have a vLLM-on-merged v3.
    # Even if an HF-backend v3 exists (lm_eval_v3 with backend='hf'), it counts as
    # 'needs vllm v3' because PI mandates vLLM-on-merged for the post-P0 evaluation.
    info["needs_vllm_v3"] = (
        info["is_merge_method"]
        and (info["has_adapter"] or info["has_best_ckpt"])
        and not has_post_fix_vllm_v3
    )
    return info


def main() -> int:
    rows: list[dict] = []
    for model_dir in sorted(RESULTS.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir() or ds_dir.name.startswith("_"):
                continue
            dataset = ds_dir.name
            for method_dir in sorted(ds_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                seed = method_dir / "seed42"
                if not seed.is_dir():
                    continue
                rows.append(cell_status(model, dataset, method, seed))

    pi_target = [r for r in rows if r["model"] in PI_TARGET_MODELS]
    needs = [r for r in rows if r["needs_v3_reeval"]]
    needs_pi = [r for r in pi_target if r["needs_v3_reeval"]]
    needs_vllm = [r for r in rows if r["needs_vllm_v3"]]
    needs_vllm_pi = [r for r in pi_target if r["needs_vllm_v3"]]
    tainted_only = [r for r in rows if r["is_tainted"]]

    out = {
        "fix_commit": "b7d07dc",
        "fix_dt_utc": FIX_DT.isoformat(),
        "merge_methods": sorted(MERGE_METHODS),
        "n_rows": len(rows),
        "n_tainted": len(tainted_only),
        "n_needs_v3_reeval": len(needs),
        "n_needs_v3_reeval_pi_targets": len(needs_pi),
        "n_needs_vllm_v3": len(needs_vllm),
        "n_needs_vllm_v3_pi_targets": len(needs_vllm_pi),
        "rows": rows,
    }
    out_path = ROOT / "analysis" / "p0_tainted_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"=== Manifest written: {out_path.relative_to(ROOT)} ===")
    print(f"Total cells scanned: {len(rows)}")
    print(f"Tainted (merge-method + pre-fix eval exists): {len(tainted_only)}")
    print(f"Needs v3 re-eval (no post-fix v3 of ANY backend yet): {len(needs)}")
    print(f"  ... within PI's target models {sorted(PI_TARGET_MODELS)}: {len(needs_pi)}")
    print(f"Needs vLLM-on-merged v3 (PI strict requirement): {len(needs_vllm)}")
    print(f"  ... within PI's target models {sorted(PI_TARGET_MODELS)}: {len(needs_vllm_pi)}")
    print()
    print("=== Cells needing vLLM-on-merged v3 re-eval (PI target models) ===")
    for r in needs_vllm_pi:
        adp = "adapter" if r["has_adapter"] else ("best" if r["has_best_ckpt"] else "NONE")
        existing = [e for e in r["evals"] if e["is_v3"] and not e["pre_fix"]]
        prev = f" (has hf-v3)" if any(e["backend"] == "hf" for e in existing) else ""
        print(f"  {r['model']:14s} {r['dataset']:18s} {r['method']:30s}  src={adp}{prev}")
    print()
    print("=== Cells needing vLLM-on-merged v3 re-eval (other models) ===")
    for r in needs_vllm:
        if r["model"] in PI_TARGET_MODELS:
            continue
        adp = "adapter" if r["has_adapter"] else ("best" if r["has_best_ckpt"] else "NONE")
        existing = [e for e in r["evals"] if e["is_v3"] and not e["pre_fix"]]
        prev = f" (has hf-v3)" if any(e["backend"] == "hf" for e in existing) else ""
        print(f"  {r['model']:14s} {r['dataset']:18s} {r['method']:30s}  src={adp}{prev}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
