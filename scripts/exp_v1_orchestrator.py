#!/usr/bin/env python3
"""Task 4 controlled-experiment orchestrator (priority slice).

Trains qwen3-8b on tulu3-sft with 7 methods, seed=42:
  M0: lora_vanilla                       (no-merge baseline)
  M1: relora_baseline                    (merge-all)
  M2: relora_random_drop                 (random gating)
  M3: relora_diag_gated_S3pos            (saliency, original buggy reset)
  M4: relora_diag_gated_S3pos_keepB      (saliency + keep-B fix; Task 2)
  M5: relora_diag_gated_S3pos_keepB +
      saliency_calib_set=gsm8k_train     (saliency + keep-B + OOD calib; Task 3)
  M6: cola                               (chain-of-LoRA / functional clone of relora_baseline)

Settings match Phase D Wave 1 (total_steps=3000, merge_every=750 for merge
methods; lora_vanilla and dora-style use total_steps=800 / merge=9999).

Outputs go to results/exp_v1/qwen3-8b/tulu3-sft/<method>[_calibgsm8k]/seed42/.
After train completes, eval is queued via p0_reeval_orchestrator (or run
directly here if --eval_after).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "exp_v1"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "orchestrator.log"

PY_ESPO  = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"
PY_RRENV = "/mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python"

MODEL = "qwen3-8b"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
DATASET = "tulu3-sft"
SEED = 42
ATTN = "sdpa"

# (label, method, extra_args). label is the directory name suffix.
EXP_GRID = [
    ("lora_vanilla",      "lora_vanilla",                  []),
    ("relora_baseline",   "relora_baseline",               []),
    ("relora_random_drop","relora_random_drop",            []),
    ("relora_S3pos",      "relora_diag_gated_S3pos",       []),
    ("relora_S3pos_keepB","relora_diag_gated_S3pos_keepB", []),
    ("relora_S3pos_keepB_calibgsm8k",
                          "relora_diag_gated_S3pos_keepB",
                          ["--saliency_calib_set", "gsm8k_train",
                           "--saliency_calib_n", "256"]),
    ("cola",              "cola",                          []),
]

NO_MERGE = {"lora_vanilla", "dora", "adalora"}


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
    out_root = ROOT / "results" / "exp_v1" / MODEL / DATASET
    jobs = []
    for label, method, extra in EXP_GRID:
        steps = 800 if method == "dora" else 3000
        merge = 9999 if method in NO_MERGE else 750
        out_dir = out_root / label / f"seed{SEED}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Skip if summary.json already present (already trained).
        if (out_dir / "summary.json").exists():
            log(f"SKIP train (summary.json exists): {label}")
            continue
        jobs.append({
            "name": f"{MODEL}__{DATASET}__{label}",
            "label": label,
            "method": method,
            "out_dir": out_dir,
            "steps": steps,
            "merge": merge,
            "extra": extra,
        })
    return jobs


def launch_train(gpu: int, job: dict) -> int:
    log_path = LOG_DIR / f"{job['name']}.train.log"
    cmd = [
        PY_ESPO, str(ROOT / "scripts" / "stage3_run.py"),
        "--model_path", MODEL_PATH,
        "--model_key", MODEL,
        "--dataset", DATASET,
        "--method", job["method"],
        "--total_steps", str(job["steps"]),
        "--merge_every", str(job["merge"]),
        "--eval_every", "250",
        "--ckpt_every", "250",
        "--saliency_max_seq_len", "512",
        "--attn_implementation", ATTN,
        "--save_adapter",
        "--seed", str(SEED),
        "--out_root", str(job["out_dir"]),
    ] + list(job["extra"])
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
    ap.add_argument("--max_parallel", type=int, default=3,
                    help="Max concurrent train jobs.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    jobs = build_jobs()
    log(f"Task 4 sweep: {len(jobs)} train jobs queued (qwen3-8b/tulu3-sft, seed=42)")

    if args.dry_run:
        for j in jobs:
            log(f"  {j['name']} method={j['method']} steps={j['steps']} merge={j['merge']} extra={j['extra']}")
        return 0

    allowed = None
    if args.gpus:
        allowed = {int(x) for x in args.gpus.split(",") if x.strip()}

    running: dict[int, dict] = {}
    pids: dict[int, int] = {}
    queue = list(jobs)

    while queue or running:
        # Reap finished
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

        # Launch new
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
            log(f"LAUNCH train gpu={gpu} {j['name']} method={j['method']} pid={pid}")

        if (LOG_DIR / ".STOP").exists():
            log("STOP signal; waiting for running jobs.")
            break

        time.sleep(args.poll)

    log("Task 4 train sweep finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
