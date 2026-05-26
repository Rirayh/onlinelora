#!/usr/bin/env python3
"""Exp-S2.5 schedule pilot orchestrator (PI 2026-05-26 v2 directive).

12 mandatory schedules on qwen3-8b/tulu3-sft, seed=42:
  const_0p25, const_0p5, const_0p75 (3 reused from Exp-1, listed but skipped
  if Exp-1 result dirs exist; the schedule_pilot summary points to them.)
  anneal_down, anneal_up, triangle_up_down, triangle_down_up,
  early_burst, late_burst, bookend_burst, extreme_alternate,
  random_schedule:seed=42 (12th).

Settings: total_steps=3000, merge_every=500 (= 6 events; per directive),
method=relora_random_drop, --drop_schedule <name>.

Output: results/exp_schedule/qwen3-8b/tulu3-sft/<schedule_label>/seed42/
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "exp_schedule"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "orchestrator.log"

PY_ESPO = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"

MODEL = "qwen3-8b"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
DATASET = "tulu3-sft"
SEED = 42
ATTN = "sdpa"

# (schedule_label, drop_schedule_arg). Empty arg means "skip; reuse Exp-1".
SCHEDULES: list[tuple[str, str]] = [
    ("const_0p25",         ""),  # reused from Exp-1 dr0.25
    ("const_0p5",          ""),  # reused from Exp-1 dr0.5
    ("const_0p75",         ""),  # reused from Exp-1 dr0.75
    ("anneal_down",        "anneal_down"),
    ("anneal_up",          "anneal_up"),
    ("triangle_up_down",   "triangle_up_down"),
    ("triangle_down_up",   "triangle_down_up"),
    ("early_burst",        "early_burst"),
    ("late_burst",         "late_burst"),
    ("bookend_burst",      "bookend_burst"),
    ("extreme_alternate",  "extreme_alternate"),
    ("random_schedule_s42","random_schedule:seed=42"),
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
    out_root = ROOT / "results" / "exp_schedule" / MODEL / DATASET
    jobs = []
    for label, sched_arg in SCHEDULES:
        out_dir = out_root / label / f"seed{SEED}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not sched_arg:
            log(f"REUSE {label} from Exp-1 (no train job needed)")
            continue
        if (out_dir / "summary.json").exists():
            log(f"SKIP train (summary.json exists): {label}")
            continue
        jobs.append({
            "name": f"{MODEL}__{DATASET}__{label}",
            "label": label,
            "drop_schedule": sched_arg,
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
        "--drop_schedule", job["drop_schedule"],
        "--total_steps", "3000",
        "--merge_every", "500",
        "--eval_every", "250",
        "--ckpt_every", "500",
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
    ap.add_argument("--max_parallel", type=int, default=7,
                    help="Max concurrent train jobs (default 7).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    jobs = build_jobs()
    log(f"Exp-S2.5 schedule pilot: {len(jobs)} train jobs queued "
        f"(plus 3 reused from Exp-1; total 12 schedules)")

    if args.dry_run:
        for j in jobs:
            log(f"  {j['name']} drop_schedule={j['drop_schedule']}")
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
            log(f"LAUNCH train gpu={gpu} {j['name']} sched={j['drop_schedule']} pid={pid}")

        if (LOG_DIR / ".STOP").exists():
            log("STOP signal; waiting for running jobs.")
            break

        time.sleep(args.poll)

    log("Exp-S2.5 schedule pilot train phase finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
