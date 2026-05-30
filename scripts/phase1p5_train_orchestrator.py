#!/usr/bin/env python3
"""Phase 1.5 schedule ablation training orchestrator (PI feedback #7/#8).

qwen3-8b/tulu3-sft × 4 new cells × seed 42 = 4 cells
  - random_anneal_up       relora_random_drop --drop_schedule anneal_up
  - random_anneal_down     relora_random_drop --drop_schedule anneal_down
  - random_triangle_up_down relora_random_drop --drop_schedule triangle_up_down
  - random_triangle_down_up relora_random_drop --drop_schedule triangle_down_up

random_const_0p5 (flat schedule) = s2_pi5b_v3 random_dr0.5/seed42 (reused).
v1_S3pos comparison = s2_pi5b_v3 v1_S3pos/seed42 (reused).

Config: total_steps=3000, merge_every=750, --save_merged_final.
Output: results/phase1p5_schedule_ablation/qwen3-8b/tulu3-sft/<cell>/seed<seed>/

This script polls for free GPUs and launches when 4 are available.
Safe to run while Phase 1 is active — waits until GPUs free up.

Usage:
  nohup python scripts/phase1p5_train_orchestrator.py \
      [--exclude_gpus ""] [--cells random_anneal_down] [--seeds 43 44] \
      [--dry_run] [--nowait] \
      > logs/phase1p5/train_orch.log 2>&1 &
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "phase1p5"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "train_orchestrator.log"

PY_ESPO    = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"
MODEL      = "qwen3-8b"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
DATASET    = "tulu3-sft"
ATTN       = "sdpa"
SEED       = 42

CELLS = [
    ("random_anneal_up",        "relora_random_drop", ["--drop_schedule", "anneal_up"]),
    ("random_anneal_down",      "relora_random_drop", ["--drop_schedule", "anneal_down"]),
    ("random_triangle_up_down", "relora_random_drop", ["--drop_schedule", "triangle_up_down"]),
    ("random_triangle_down_up", "relora_random_drop", ["--drop_schedule", "triangle_down_up"]),
]

POLL_INTERVAL_S = 120
FREE_THRESHOLD_MB = 2000


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


def free_gpus(exclude: set[int]) -> list[int]:
    return [g["idx"] for g in gpu_state()
            if g["used_mb"] < FREE_THRESHOLD_MB and g["idx"] not in exclude]


def build_cmd(method: str, extra: list[str], out_dir: Path, seed: int) -> list[str]:
    base = [
        PY_ESPO, str(ROOT / "scripts" / "stage3_run.py"),
        "--model_path", MODEL_PATH,
        "--model_key", MODEL,
        "--dataset", DATASET,
        "--method", method,
        "--total_steps", "3000",
        "--merge_every", "750",
        "--eval_every", "250",
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
    ap.add_argument("--exclude_gpus", default="",
                    help="GPU indices to never use (comma-sep).")
    ap.add_argument("--cells", nargs="*", choices=[c[0] for c in CELLS],
                    help="Optional subset of Phase 1.5 cells to launch.")
    ap.add_argument("--seeds", nargs="*", type=int,
                    help="Optional seeds to launch. Defaults to seed42.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--nowait", action="store_true",
                    help="Fail immediately if <4 GPUs free instead of polling.")
    args = ap.parse_args()

    excl = {int(x) for x in args.exclude_gpus.split(",") if x.strip()}
    cell_filter = set(args.cells or [c[0] for c in CELLS])
    seeds = args.seeds or [SEED]

    out_root_base = ROOT / "results" / "phase1p5_schedule_ablation" / MODEL / DATASET
    out_root_base.mkdir(parents=True, exist_ok=True)

    pending = []
    for seed in seeds:
        for cell_label, method, extra in CELLS:
            if cell_label not in cell_filter:
                continue
            out_dir = out_root_base / cell_label / f"seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            if (out_dir / "merged_final" / "config.json").exists():
                log(f"SKIP (merged_final exists): {cell_label}/seed{seed}")
                continue
            pending.append((cell_label, seed, method, extra, out_dir))

    if not pending:
        log("Selected cells/seeds have merged_final/; nothing to do.")
        return 0

    log(f"pending ({len(pending)}): {[f'{p[0]}/seed{p[1]}' for p in pending]}")
    n_needed = len(pending)

    if args.dry_run:
        gpus = free_gpus(excl) or list(range(n_needed))
        for i, (cell_label, seed, method, extra, out_dir) in enumerate(pending):
            gpu = gpus[i] if i < len(gpus) else f"?{i}"
            cmd = build_cmd(method, extra, out_dir, seed)
            log(f"  DRY {cell_label}/seed{seed} gpu={gpu} cmd={' '.join(cmd)}")
        log("dry-run done.")
        return 0

    log(f"waiting for {n_needed} free GPUs (excl={sorted(excl)}) ...")
    while True:
        avail = free_gpus(excl)
        if len(avail) >= n_needed:
            log(f"found {len(avail)} free GPUs: {avail}")
            break
        if args.nowait:
            log(f"ABORT: only {len(avail)} free GPUs, need {n_needed}.")
            return 1
        log(f"  {len(avail)} free (need {n_needed}); retry in {POLL_INTERVAL_S}s ...")
        time.sleep(POLL_INTERVAL_S)

    gpus = avail[:n_needed]
    procs = []
    for i, (cell_label, seed, method, extra, out_dir) in enumerate(pending):
        gpu = gpus[i]
        cmd = build_cmd(method, extra, out_dir, seed)
        log_path = LOG_DIR / f"{cell_label}.seed{seed}.train.log"
        log(f"launch {cell_label}/seed{seed} on GPU {gpu} -> {log_path.name}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=f,
                                    stderr=subprocess.STDOUT,
                                    cwd=str(ROOT), preexec_fn=os.setsid)
        procs.append((cell_label, seed, gpu, proc.pid))

    log(f"launched {len(procs)} jobs:")
    for cell, seed, gpu, pid in procs:
        log(f"  PID={pid} GPU={gpu} {cell}/seed{seed}")
    log("orchestrator detaches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
