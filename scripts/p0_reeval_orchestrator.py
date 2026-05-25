#!/usr/bin/env python3
"""P0 re-eval orchestrator (Task 1).

Reads analysis/p0_tainted_manifest.json, picks all rows with
`needs_vllm_v3=True`, then for each cell:

  1. If lm_eval/ exists with pre-fix mtime, rename to lm_eval_PRE_P0_FIX_TAINTED/.
  2. Merge adapter (best/) into seed_dir/merged/ if not already (CPU, RRenv).
  3. Run lm_eval on the merged checkpoint:
       - DENSE models (env=espo)  -> vLLM (gpu_memory_utilization=0.85,
                                           max_model_len=4096, batch=auto)
       - HYBRID Qwen3.5 models    -> HF backend with peft= (no merge needed for
                                     HF; merge step skipped, pass adapter directly)

Free GPUs are detected by `nvidia-smi`. Runs up to N_PARALLEL evals.

Designed to be safe to re-run: it skips cells that already have a v3 vllm
backend output post-fix.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "p0_reeval"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "orchestrator.log"

PY_ESPO  = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"
PY_RRENV = "/mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python"

HYBRID_MODELS = {"qwen35-0p8b", "qwen35-2b", "qwen35-4b", "qwen35-9b", "qwen35-27b"}

MODEL_PATH = {
    "qwen3-1p7b":   "/mnt/cpfs/junlongke/onlinelora/models/qwen3-1p7b",
    "qwen3-4b":     "/mnt/cpfs/junlongke/onlinelora/models/qwen3-4b",
    "qwen3-14b":    "/mnt/cpfs/junlongke/onlinelora/models/qwen3-14b",
    "qwen35-0p8b":  "/mnt/cpfs/junlongke/onlinelora/models/qwen35-0p8b",
    "qwen35-2b":    "/mnt/cpfs/junlongke/onlinelora/models/qwen35-2b",
    "qwen35-4b":    "/mnt/cpfs/junlongke/onlinelora/models/qwen35-4b",
    "qwen35-9b":    "/mnt/cpfs/junlongke/onlinelora/models/qwen35-9b",
    "llama3-8b":    "/mnt/cpfs/junlongke/onlinelora/models/Llama-3.1-8B-Instruct",
    "olmo2-7b":     "/mnt/cpfs/junlongke/onlinelora/models/OLMo-2-7B",
    "r1-distill-7b":"/mnt/cpfs/junlongke/onlinelora/models/R1-Distill-Qwen-7B",
    "gemma3-12b":   "/mnt/cpfs/junlongke/onlinelora/models/gemma-3-12b-it",
    # PI targets (live in public_model, not local mirror).
    "mistral-7b":   "/mnt/cpfs/public_data/public_model/Mistral/Mistral-7B-v0.3",
    "qwen25-7b":    "/mnt/cpfs/public_data/public_model/Qwen2.5/Qwen2.5-7B-Instruct",
    "qwen3-8b":     "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B",
}


def _resolve_base_path(model: str, adapter_dir: Path) -> str:
    """Prefer adapter_config.json:base_model_name_or_path; fall back to MODEL_PATH."""
    cfg = adapter_dir / "adapter_config.json"
    if cfg.exists():
        try:
            data = json.load(open(cfg))
            p = data.get("base_model_name_or_path")
            if p and Path(p).exists():
                return p
        except Exception:
            pass
    return MODEL_PATH.get(model, "")

HF_BS_BY_SIZE = {"qwen35-0p8b": 16, "qwen35-2b": 8, "qwen35-4b": 4,
                 "qwen35-9b": 2, "qwen35-27b": 1}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with DAEMON_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def is_hybrid(model: str) -> bool:
    return model in HYBRID_MODELS


def gpu_state() -> list[dict]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    rs = []
    for line in out.strip().splitlines():
        idx, used = [s.strip() for s in line.split(",")]
        rs.append({"idx": int(idx), "used_mb": int(used)})
    return rs


def free_gpus(threshold_mb: int = 1500) -> list[int]:
    return [g["idx"] for g in gpu_state() if g["used_mb"] < threshold_mb]


def rename_old_lmeval(lm_eval_dir: Path) -> bool:
    if not lm_eval_dir.exists():
        return False
    target = lm_eval_dir.parent / "lm_eval_PRE_P0_FIX_TAINTED"
    if target.exists():
        # already renamed (idempotent)
        return False
    log(f"  rename {lm_eval_dir} -> {target.name}")
    lm_eval_dir.rename(target)
    return True


def has_post_fix_vllm_eval(lm_eval_dir: Path) -> bool:
    """Check if lm_eval/ contains any vllm-backend results.json."""
    if not lm_eval_dir.exists():
        return False
    for p in lm_eval_dir.rglob("results_*.json"):
        try:
            data = json.load(open(p))
            mc = data.get("model_configuration", {}) or data.get("config", {})
            backend = mc.get("model") or mc.get("model_type") or ""
            if "vllm" in str(backend).lower():
                return True
        except Exception:
            continue
    return False


def merge_adapter(model_path: str, adapter_dir: Path, merged_dir: Path,
                  log_path: Path) -> bool:
    """Run merge_adapter.py (CPU, RRenv) to bake adapter into base weights."""
    sentinel = merged_dir / ".merge.done"
    if sentinel.exists() and (merged_dir / "config.json").exists():
        return True
    cmd = [
        PY_RRENV, str(ROOT / "scripts" / "merge_adapter.py"),
        "--base", model_path,
        "--adapter", str(adapter_dir),
        "--out", str(merged_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUDA_HOME"] = "/usr/local/cuda-12"
    log(f"  MERGE -> {merged_dir}")
    with log_path.open("w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                            cwd=str(ROOT), timeout=900).returncode
    if rc != 0:
        log(f"  MERGE FAIL rc={rc} (see {log_path})")
        return False
    return True


def launch_eval_dense(gpu: int, model: str, model_path: str,
                      merged_dir: Path, out_dir: Path, log_path: Path) -> int:
    """vLLM on merged checkpoint (Qwen3 dense + non-Qwen3.5 models)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        PY_RRENV, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args",
        f"pretrained={merged_dir},dtype=bfloat16,gpu_memory_utilization=0.85,"
        f"max_model_len=4096,trust_remote_code=True",
        "--tasks", "gsm8k,hellaswag,arc_challenge",
        "--num_fewshot", "5",
        "--batch_size", "auto",
        "--log_samples",
        "--output_path", str(out_dir),
    ]
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    return proc.pid


def launch_eval_hybrid(gpu: int, model: str, model_path: str,
                       adapter_dir: Path, out_dir: Path, log_path: Path) -> int:
    """HF backend for Qwen3.5 hybrid models (vLLM doesn't support them)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    bs = HF_BS_BY_SIZE.get(model, 4)
    cmd = [
        PY_RRENV, "-m", "lm_eval",
        "--model", "hf",
        "--model_args",
        f"pretrained={model_path},peft={adapter_dir},dtype=bfloat16,"
        f"attn_implementation=sdpa,trust_remote_code=True",
        "--tasks", "gsm8k,hellaswag,arc_challenge",
        "--num_fewshot", "5",
        "--batch_size", str(bs),
        "--log_samples",
        "--output_path", str(out_dir),
    ]
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    return proc.pid


def build_jobs(manifest: dict) -> list[dict]:
    jobs = []
    for r in manifest["rows"]:
        if not r.get("needs_vllm_v3"):
            continue
        model = r["model"]
        if model not in MODEL_PATH:
            log(f"SKIP (no model_path): {model}/{r['dataset']}/{r['method']}")
            continue
        seed_dir = ROOT / r["seed_dir"]
        adapter_src = seed_dir / "checkpoints" / "best"
        if not adapter_src.exists():
            adapter_src = seed_dir / "adapter"
        if not adapter_src.exists():
            log(f"SKIP (no adapter): {model}/{r['dataset']}/{r['method']}")
            continue
        base_path = _resolve_base_path(model, adapter_src)
        if not base_path or not Path(base_path).exists():
            log(f"SKIP (base path missing): {model} -> {base_path}")
            continue
        name = f"{model}__{r['dataset']}__{r['method']}"
        jobs.append({
            "name": name,
            "model": model,
            "model_path": base_path,
            "dataset": r["dataset"],
            "method": r["method"],
            "seed_dir": seed_dir,
            "adapter": adapter_src,
            "merged_dir": seed_dir / "merged",
            "out_dir": seed_dir / "lm_eval",
            "is_hybrid": is_hybrid(model),
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "analysis" / "p0_tainted_manifest.json"))
    ap.add_argument("--max_parallel", type=int, default=8,
                    help="Max concurrent eval jobs (default 8 = 8 GPUs).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--gpus", default="",
                    help="Restrict to these GPU indices (comma-sep). Empty = all free.")
    ap.add_argument("--poll", type=int, default=20)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    jobs = build_jobs(manifest)
    log(f"P0 re-eval orchestrator: {len(jobs)} cells queued (manifest n_needs_vllm_v3="
        f"{manifest.get('n_needs_vllm_v3','?')})")

    if args.dry_run:
        for j in jobs[:10]:
            log(f"  {j['name']} hybrid={j['is_hybrid']} adapter={j['adapter']}")
        log(f"  ... ({len(jobs)} total)")
        return 0

    # Step 1: rename all old lm_eval/ dirs (PRE_P0_FIX_TAINTED) up front.
    for j in jobs:
        rename_old_lmeval(j["out_dir"])

    # Step 2: skip cells that already have a post-fix vllm eval (idempotent).
    pending = []
    for j in jobs:
        if has_post_fix_vllm_eval(j["out_dir"]):
            log(f"SKIP (already has v3 vllm): {j['name']}")
            continue
        pending.append(j)
    log(f"After idempotency check: {len(pending)} jobs remain")

    allowed_gpus = None
    if args.gpus:
        allowed_gpus = {int(x) for x in args.gpus.split(",") if x.strip()}

    running: dict[int, dict] = {}   # gpu -> job
    pids: dict[int, int] = {}       # gpu -> pid
    queue = list(pending)
    done = 0
    failed = 0

    while queue or running:
        # Poll: clean up finished jobs.
        for gpu, pid in list(pids.items()):
            try:
                rc = subprocess.run(["kill", "-0", str(pid)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL).returncode
                if rc != 0:
                    j = running.pop(gpu, None)
                    pids.pop(gpu, None)
                    if j:
                        # check if eval dir got results
                        ok = any(p.suffix == ".json" for p in
                                 (j["out_dir"]).rglob("results_*.json"))
                        log(f"DONE gpu={gpu} {j['name']} ok={ok}")
                        if ok:
                            done += 1
                        else:
                            failed += 1
            except Exception:
                pass

        # Launch new jobs onto idle GPUs.
        idle = free_gpus()
        if allowed_gpus:
            idle = [g for g in idle if g in allowed_gpus]
        idle = [g for g in idle if g not in running]

        while queue and idle and len(running) < args.max_parallel:
            gpu = idle.pop(0)
            j = queue.pop(0)
            log_eval = LOG_DIR / f"{j['name']}.eval.log"
            log_merge = LOG_DIR / f"{j['name']}.merge.log"
            if not j["is_hybrid"]:
                # Need to merge first (CPU; while GPU is reserved for this job).
                # Reserve gpu by adding to running dict before merge so we don't
                # double-allocate.
                running[gpu] = j
                pids[gpu] = -1
                log(f"PREP gpu={gpu} {j['name']} (merge then vllm)")
                ok = merge_adapter(j["model_path"], j["adapter"],
                                   j["merged_dir"], log_merge)
                if not ok:
                    log(f"FAIL merge: {j['name']}")
                    running.pop(gpu, None); pids.pop(gpu, None)
                    failed += 1
                    continue
                pid = launch_eval_dense(gpu, j["model"], j["model_path"],
                                        j["merged_dir"], j["out_dir"], log_eval)
                pids[gpu] = pid
                log(f"LAUNCH eval(vllm) gpu={gpu} {j['name']} pid={pid}")
            else:
                running[gpu] = j
                pid = launch_eval_hybrid(gpu, j["model"], j["model_path"],
                                         j["adapter"], j["out_dir"], log_eval)
                pids[gpu] = pid
                log(f"LAUNCH eval(hf-hybrid) gpu={gpu} {j['name']} pid={pid}")

        # Stop signal.
        if (ROOT / "logs" / "p0_reeval" / ".STOP").exists():
            log("STOP signal received; waiting for running jobs to finish...")
            break

        time.sleep(args.poll)

    log(f"orchestrator finished: done={done} failed={failed} remaining={len(queue)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
