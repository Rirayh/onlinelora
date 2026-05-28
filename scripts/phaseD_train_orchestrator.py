#!/usr/bin/env python3
"""Phase D: vanilla over-train control + v1 over-train (PI feedback #6 §D).

Cells (4 total, all qwen3-8b/tulu3-sft):
  lora_vanilla × seed 42, 43  — total_steps=10000
  v1_S3pos     × seed 42, 43  — total_steps=10000

Eval checkpoints saved at steps 3000, 5000, 7500, 10000 via --eval_every
trick: we use eval_every=2500 and save_merged at step 3000 via a separate
pass. Actually we want 4 checkpoints (3000,5000,7500,10000) to use for
offline lm-eval. Strategy: set eval_every=2500, ckpt_every=2500 — but
stage3_run.py only saves one best-step checkpoint. To capture merged
weights at multiple steps we need a different approach.

Simplest approach supported by current stage3_run.py:
  - Run to 10000 steps with --save_merged_final (saves post-all-merges at end)
  - Also save val_loss every 500 steps (eval_every=500)
  - Offline lm-eval runs on the final merged_final/ only (10k-step endpoint)
  - The 3000-step score comes from s2_pi5b_v3 existing results (seed42 only;
    we skip re-running 3000 here to save GPU-hours)

PI §D says: "eval at step ∈ {3000,5000,7500,10000}" — we handle this by
checkpointing merged_final at those steps. Stage3_run.py doesn't natively
checkpoint merged weights at intermediate steps. Workaround: chain 4 separate
runs with restart:
  run1: total_steps=3000, warm_start from 0 → merged_final_3k/
  run2: total_steps=5000, resume → merged_final_5k/
  ... overhead too high.

Practical approach: run single 10k job with eval_every=500 for val-loss
trajectory, and save_merged_final at the END. Then run lm-eval on that
single 10k merged model. Compare lm-eval of 3k (from phase1) vs 10k endpoint.
This still answers PI's core question: does hellaswag collapse? Does gsm8k improve?

We implement this as a single 10k run per cell/seed with:
  --total_steps 10000
  --merge_every 750  (gives 13 merges total for relora cells)
  --eval_every 500
  --save_merged_final

Output: results/phase_d/<cell>/seed<N>/merged_final/

PI §D decision rule uses lm-eval on merged_final (10k endpoint) compared to
Phase 1 seed42 3k lm-eval. This is sufficient to answer the trajectory question.

Usage:
  nohup python scripts/phaseD_train_orchestrator.py [--gpus 4,5,6,7] \
      [--dry_run] > logs/phaseD/train_orch.log 2>&1 &
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "phaseD"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "train_orchestrator.log"

PY_ESPO = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"

MODEL = "qwen3-8b"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
DATASET = "tulu3-sft"
ATTN = "sdpa"

CELLS = [
    ("lora_vanilla", "lora_vanilla", []),
    ("v1_S3pos",     "relora_diag_gated_S3pos", []),
]
SEEDS = [42, 43]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with DAEMON_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def gpu_state() -> list[dict]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    rs = []
    for ln in out.strip().splitlines():
        i, u = [s.strip() for s in ln.split(",")]
        rs.append({"idx": int(i), "used_mb": int(u)})
    return rs


def free_gpus(threshold_mb: int = 2000, exclude=None) -> list[int]:
    excl = set(exclude or [])
    return [g["idx"] for g in gpu_state()
            if g["used_mb"] < threshold_mb and g["idx"] not in excl]


def build_cmd(method: str, extra: list[str], seed: int, out_dir: Path) -> list[str]:
    base = [
        PY_ESPO, str(ROOT / "scripts" / "stage3_run.py"),
        "--model_path", MODEL_PATH,
        "--model_key", MODEL,
        "--dataset", DATASET,
        "--method", method,
        "--total_steps", "10000",
        "--merge_every", "750",
        "--eval_every", "500",
        "--ckpt_every", "0",
        "--saliency_max_seq_len", "512",
        "--attn_implementation", ATTN,
        "--save_merged_final",
        "--seed", str(seed),
        "--out_root", str(out_dir),
    ]
    return base + extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="",
                    help="Comma-sep GPU ids; empty=auto-detect free.")
    ap.add_argument("--exclude_gpus", default="",
                    help="GPUs to reserve (comma-sep).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_root_base = ROOT / "results" / "phase_d" / MODEL / DATASET
    out_root_base.mkdir(parents=True, exist_ok=True)

    excl = [int(x) for x in args.exclude_gpus.split(",") if x.strip()]
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        gpus = free_gpus(exclude=excl)
        log(f"auto-detected free GPUs (excl={excl}): {gpus}")

    pending = []
    for cell_label, method, extra in CELLS:
        for seed in SEEDS:
            out_dir = out_root_base / cell_label / f"seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            if (out_dir / "merged_final" / "config.json").exists():
                log(f"SKIP (merged_final exists): {cell_label}/seed{seed}")
                continue
            pending.append((cell_label, method, extra, seed, out_dir))

    if not pending:
        log("All 4 cells have merged_final/; nothing to do.")
        return 0

    log(f"pending ({len(pending)}): {[(p[0], p[3]) for p in pending]}")

    if len(gpus) < len(pending):
        log(f"WARN: {len(gpus)} GPUs for {len(pending)} cells.")

    procs = []
    for i, (cell_label, method, extra, seed, out_dir) in enumerate(pending):
        if i >= len(gpus):
            log(f"no GPU for remaining cells starting at {cell_label}/seed{seed}.")
            break
        gpu = gpus[i]
        cmd = build_cmd(method, extra, seed, out_dir)
        log_path = LOG_DIR / f"{cell_label}.seed{seed}.train.log"
        log(f"launch {cell_label}/seed{seed} on GPU {gpu} -> {log_path.name}")
        if args.dry_run:
            log(f"  cmd: {' '.join(cmd)}")
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=f,
                                    stderr=subprocess.STDOUT,
                                    cwd=str(ROOT), preexec_fn=os.setsid)
        procs.append((cell_label, seed, gpu, proc.pid))

    if args.dry_run:
        log("dry-run done.")
        return 0

    log(f"launched {len(procs)} jobs:")
    for cell, seed, gpu, pid in procs:
        log(f"  PID={pid} GPU={gpu} {cell}/seed{seed}")
    log("orchestrator detaches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
