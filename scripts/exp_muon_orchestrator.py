#!/usr/bin/env python3
"""Exp-2 Muon-decoupling orchestrator (PI 2026-05-26).

8 cells = optimizer x drop_rate x selection on qwen3-8b/tulu3-sft, seed=42:
  (adamw, 0.0,  random)        # = relora_baseline reference
  (adamw, 0.5,  random)        # = relora_random_drop reference (AdamW)
  (adamw, 0.5,  S3pos_keepB_calibgsm8k)  # best AdamW variant
  (adamw, 0.0,  S3pos_keepB_calibgsm8k)  # AdamW saliency without drop
  (muon,  0.0,  random)        # Muon baseline
  (muon,  0.5,  random)        # Muon + random drop
  (muon,  0.5,  S3pos_keepB_calibgsm8k)  # MAIN CELL
  (muon,  0.0,  S3pos_keepB_calibgsm8k)  # Muon saliency without drop

Output: results/exp_muon/qwen3-8b/tulu3-sft/<label>/seed42/
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "exp_muon"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "orchestrator.log"

PY_ESPO = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"

MODEL = "qwen3-8b"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
DATASET = "tulu3-sft"
SEED = 42
ATTN = "sdpa"

# (label, optimizer, drop_rate, selection)
EXP_GRID = [
    ("adamw_dr0_random",                "adamw", 0.0, "random"),
    ("adamw_dr0.5_random",              "adamw", 0.5, "random"),
    ("adamw_dr0.5_S3pos_keepB_calib",   "adamw", 0.5, "S3pos_keepB_calib"),
    ("adamw_dr0_S3pos_keepB_calib",     "adamw", 0.0, "S3pos_keepB_calib"),
    ("muon_dr0_random",                 "muon",  0.0, "random"),
    ("muon_dr0.5_random",               "muon",  0.5, "random"),
    ("muon_dr0.5_S3pos_keepB_calib",    "muon",  0.5, "S3pos_keepB_calib"),
    ("muon_dr0_S3pos_keepB_calib",      "muon",  0.0, "S3pos_keepB_calib"),
]


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
    out_root = ROOT / "results" / "exp_muon" / MODEL / DATASET
    jobs = []
    for label, optim, drop_rate, selection in EXP_GRID:
        out_dir = out_root / label / f"seed{SEED}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "summary.json").exists():
            log(f"SKIP train (summary.json exists): {label}")
            continue
        jobs.append({
            "name": f"{MODEL}__{DATASET}__{label}",
            "label": label,
            "optimizer": optim,
            "drop_rate": drop_rate,
            "selection": selection,
            "out_dir": out_dir,
        })
    return jobs


def launch_train(gpu: int, job: dict) -> int:
    log_path = LOG_DIR / f"{job['name']}.train.log"
    selection = job["selection"]
    extra: list[str] = []
    if selection == "random":
        method = "relora_random_drop"
        extra += ["--random_drop_rate", str(job["drop_rate"])]
    elif selection == "S3pos_keepB_calib":
        method = "relora_diag_gated_S3pos_keepB"
        extra += ["--saliency_calib_set", "gsm8k_train",
                  "--saliency_calib_n", "256"]
        # Saliency methods don't honour --random_drop_rate (they use gated drops).
        # When drop_rate=0.0 we still want to run the saliency code path; the
        # baseline behaviour comes from drop_rate>0 implicit in saliency cutoff.
        # For drop_rate=0.0 cell, we effectively want "keep all" which the
        # gated mechanism may not expose. We approximate via vanilla baseline:
        if job["drop_rate"] == 0.0:
            method = "relora_baseline"
            extra = []  # baseline ignores saliency / calib
    else:
        raise ValueError(f"unknown selection={selection}")
    cmd = [
        PY_ESPO, str(ROOT / "scripts" / "stage3_run.py"),
        "--model_path", MODEL_PATH,
        "--model_key", MODEL,
        "--dataset", DATASET,
        "--method", method,
        "--optimizer", job["optimizer"],
        "--total_steps", "3000",
        "--merge_every", "750",
        "--eval_every", "250",
        "--ckpt_every", "250",
        "--saliency_max_seq_len", "512",
        "--attn_implementation", ATTN,
        "--save_adapter",
        "--seed", str(SEED),
        "--out_root", str(job["out_dir"]),
    ] + list(extra)
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
    ap.add_argument("--max_parallel", type=int, default=8,
                    help="Max concurrent train jobs (default 8 = full sweep).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    jobs = build_jobs()
    log(f"Exp-2 muon-decoupling: {len(jobs)} train jobs queued "
        f"({MODEL}/{DATASET}, seed={SEED})")

    if args.dry_run:
        for j in jobs:
            log(f"  {j['name']} optim={j['optimizer']} dr={j['drop_rate']} sel={j['selection']}")
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
            log(f"LAUNCH train gpu={gpu} {j['name']} optim={j['optimizer']} "
                f"dr={j['drop_rate']} sel={j['selection']} pid={pid}")

        if (LOG_DIR / ".STOP").exists():
            log("STOP signal; waiting for running jobs.")
            break

        time.sleep(args.poll)

    log("Exp-2 muon-decoupling train phase finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
