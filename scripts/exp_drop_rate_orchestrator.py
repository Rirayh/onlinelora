#!/usr/bin/env python3
"""Exp-1 drop-rate sweep orchestrator (PI 2026-05-26).

Sweep: relora_random_drop with drop_rate in {0.0, 0.1, 0.25, 0.5, 0.75, 0.9}
Model: qwen3-8b on tulu3-sft, seed=42, AdamW (default optimizer).
Settings match exp_v1: total_steps=3000, merge_every=750, eval/ckpt=250.

Sanity: drop_rate=0.0 must reproduce exp_v1 relora_baseline within 0.5pp gsm8k.

Outputs: results/exp_drop_rate/qwen3-8b/tulu3-sft/dr<rate>/seed42/
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "exp_drop_rate"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "orchestrator.log"

PY_ESPO = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"

MODEL = "qwen3-8b"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
DATASET = "tulu3-sft"
SEED = 42
ATTN = "sdpa"

DROP_RATES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]


def _label_for_rate(r: float) -> str:
    s = f"{r:.2f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return f"dr{s}"


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
    for line in out.strip().splitlines():
        idx, used = [s.strip() for s in line.split(",")]
        rs.append({"idx": int(idx), "used_mb": int(used)})
    return rs


def free_gpus(threshold_mb: int = 1500) -> list[int]:
    return [g["idx"] for g in gpu_state() if g["used_mb"] < threshold_mb]


def build_jobs() -> list[dict]:
    out_root = ROOT / "results" / "exp_drop_rate" / MODEL / DATASET
    jobs = []
    for r in DROP_RATES:
        label = _label_for_rate(r)
        out_dir = out_root / label / f"seed{SEED}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "summary.json").exists():
            log(f"SKIP train (summary.json exists): {label}")
            continue
        jobs.append({
            "name": f"{MODEL}__{DATASET}__{label}",
            "label": label,
            "rate": r,
            "out_dir": out_dir,
        })
    return jobs


def launch_train(gpu: int, job: dict) -> int:
    log_path = LOG_DIR / f"{job['name']}.train.log"
    cmd = [
        PY_ESPO, str(ROOT / "scripts" / "stage3_run.py"),
        "--model_path", MODEL_PATH,
        "--model_key", MODEL,
        "--dataset", DATASET,
        "--method", "relora_random_drop",
        "--random_drop_rate", str(job["rate"]),
        "--total_steps", "3000",
        "--merge_every", "750",
        "--eval_every", "250",
        "--ckpt_every", "250",
        "--saliency_max_seq_len", "512",
        "--attn_implementation", ATTN,
        "--save_adapter",
        "--seed", str(SEED),
        "--out_root", str(job["out_dir"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    return proc.pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="",
                    help="Comma-sep GPU indices to use; empty=auto-detect free.")
    ap.add_argument("--max_parallel", type=int, default=6,
                    help="Max concurrent train jobs (default 6 for full sweep).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    jobs = build_jobs()
    log(f"Exp-1 drop-rate sweep: {len(jobs)} train jobs queued "
        f"({MODEL}/{DATASET}, seed={SEED})")
    log(f"  rates: {DROP_RATES}")

    if args.dry_run:
        for j in jobs:
            log(f"  {j['name']} rate={j['rate']}")
        return 0

    allowed = None
    if args.gpus:
        allowed = {int(x) for x in args.gpus.split(",") if x.strip()}

    running: dict[int, dict] = {}
    pids: dict[int, int] = {}
    queue = list(jobs)

    while queue or running:
        for gpu, pid in list(pids.items()):
            rc = subprocess.run(["kill", "-0", str(pid)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode
            if rc != 0:
                j = running.pop(gpu, None)
                pids.pop(gpu, None)
                if j:
                    ok = (j["out_dir"] / "summary.json").exists()
                    log(f"DONE train gpu={gpu} {j['name']} ok={ok}")

        idle = free_gpus()
        if allowed:
            idle = [g for g in idle if g in allowed]
        idle = [g for g in idle if g not in running]

        while queue and idle and len(running) < args.max_parallel:
            gpu = idle.pop(0)
            j = queue.pop(0)
            pid = launch_train(gpu, j)
            running[gpu] = j
            pids[gpu] = pid
            log(f"LAUNCH train gpu={gpu} {j['name']} rate={j['rate']} pid={pid}")

        if (LOG_DIR / ".STOP").exists():
            log("STOP signal; waiting for running jobs.")
            break

        time.sleep(args.poll)

    log("Exp-1 drop-rate sweep train phase finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
