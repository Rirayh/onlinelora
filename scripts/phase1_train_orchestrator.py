#!/usr/bin/env python3
"""Phase 1 robustness-sweep training orchestrator (PI feedback #6 §C.1).

qwen3-8b/tulu3-sft × 3 cells × 3 seeds = 9 cells
  - v1_S3pos        (relora_diag_gated_S3pos)
  - random_dr0.5    (relora_random_drop dr=0.5)
  - relora_baseline (relora_baseline)
seeds: 42, 43, 44

Config: identical to s2_pi5b_v3 (total_steps=3000, merge_every=750,
        eval_every=250, --save_merged_final).

Outputs: results/phase1_robustness/qwen3-8b/tulu3-sft/<cell>/seed<N>/

Usage:
  nohup python scripts/phase1_train_orchestrator.py [--gpus 0,1,2,3,4,5,6,7] \
      [--exclude_gpus ""] [--dry_run] > logs/phase1/train_orch.log 2>&1 &
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "phase1"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "train_orchestrator.log"

PY_ESPO = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"

MODEL = "qwen3-8b"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
DATASET = "tulu3-sft"
ATTN = "sdpa"

CELLS = [
    ("v1_S3pos",        "relora_diag_gated_S3pos", []),
    ("random_dr0.5",    "relora_random_drop",       ["--random_drop_rate", "0.5"]),
    ("relora_baseline", "relora_baseline",           []),
]
SEEDS = [42, 43, 44]


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


POLL_INTERVAL_S = 120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="",
                    help="Comma-sep GPU ids; empty=auto-detect free.")
    ap.add_argument("--exclude_gpus", default="",
                    help="GPUs to reserve (comma-sep).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="Poll until enough GPUs are free before launching.")
    args = ap.parse_args()

    out_root_base = ROOT / "results" / "phase1_robustness" / MODEL / DATASET
    out_root_base.mkdir(parents=True, exist_ok=True)

    excl = set(int(x) for x in args.exclude_gpus.split(",") if x.strip())

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
        log("All cells have merged_final/; nothing to do.")
        return 0

    log(f"pending ({len(pending)}): {[(p[0], p[3]) for p in pending]}")

    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        n_needed = len(pending)
        if args.wait:
            log(f"waiting for {n_needed} free GPU(s) (excl={sorted(excl)}) ...")
            while True:
                avail = free_gpus(exclude=list(excl))
                if avail:
                    log(f"found {len(avail)} free GPUs: {avail}")
                    break
                log(f"  0 free; retry in {POLL_INTERVAL_S}s ...")
                time.sleep(POLL_INTERVAL_S)
            gpus = avail
        else:
            gpus = free_gpus(exclude=list(excl))
            log(f"auto-detected free GPUs (excl={sorted(excl)}): {gpus}")

    log(f"pending ({len(pending)}): {[(p[0], p[3]) for p in pending]}")

    if len(gpus) < len(pending):
        log(f"WARN: {len(gpus)} GPUs for {len(pending)} cells; "
            f"first {len(gpus)} will launch, re-run for rest.")

    procs = []
    for i, (cell_label, method, extra, seed, out_dir) in enumerate(pending):
        if i >= len(gpus):
            log(f"no GPU for remaining cells starting at "
                f"{cell_label}/seed{seed}; re-run this script after GPUs free.")
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
