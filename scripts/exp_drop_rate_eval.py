#!/usr/bin/env python3
"""Post-train eval orchestrator for Exp-1 drop-rate sweep cells.

Mirrors `scripts/exp_v1_eval.py` but targets `results/exp_drop_rate/...`.

Watches for newly-finished cells (presence of `summary.json`) and:
  1. Merges adapter -> seed_dir/merged/ via scripts/merge_adapter.py (RRenv, CPU)
  2. Runs lm_eval --model vllm on the merged ckpt -> seed_dir/lm_eval/
  3. Skips cells already with a valid lm_eval/results_*.json

Usage:
  nohup /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python \
      scripts/exp_drop_rate_eval.py --gpus 1,2,3,4,5,6 --max_parallel 6 --poll 60 \
      > logs/exp_drop_rate/eval.stdout.log 2>&1 &
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
DAEMON_LOG = LOG_DIR / "eval_orchestrator.log"

PY_RRENV = "/mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python"

EXP_ROOT = ROOT / "results" / "exp_drop_rate" / "qwen3-8b" / "tulu3-sft"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"


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


def has_eval_result(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    return any(p.suffix == ".json" for p in out_dir.rglob("results_*.json"))


def merge_adapter(adapter_dir: Path, merged_dir: Path, log_path: Path) -> bool:
    sentinel = merged_dir / ".merge.done"
    if sentinel.exists() and (merged_dir / "config.json").exists():
        return True
    cmd = [
        PY_RRENV, str(ROOT / "scripts" / "merge_adapter.py"),
        "--base", MODEL_PATH,
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


def launch_eval(gpu: int, merged_dir: Path, out_dir: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_HOME"] = "/usr/local/cuda-12"
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


def discover_jobs() -> list[dict]:
    """Find Exp-1 cells with summary.json but no eval result yet."""
    jobs = []
    if not EXP_ROOT.exists():
        return jobs
    for label_dir in sorted(EXP_ROOT.iterdir()):
        if not label_dir.is_dir():
            continue
        seed_dir = label_dir / "seed42"
        summary = seed_dir / "summary.json"
        if not summary.exists():
            continue
        adapter = seed_dir / "adapter"
        if not adapter.exists():
            adapter = seed_dir / "checkpoints" / "best"
        if not adapter.exists():
            continue
        out_dir = seed_dir / "lm_eval"
        if has_eval_result(out_dir):
            continue
        jobs.append({
            "name": f"exp_drop_rate__qwen3-8b__tulu3-sft__{label_dir.name}",
            "label": label_dir.name,
            "adapter": adapter,
            "merged_dir": seed_dir / "merged",
            "out_dir": out_dir,
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="",
                    help="Comma-sep GPU indices (empty = any free)")
    ap.add_argument("--max_parallel", type=int, default=6)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    allowed = None
    if args.gpus:
        allowed = {int(x) for x in args.gpus.split(",") if x.strip()}

    running: dict[int, dict] = {}
    pids: dict[int, int] = {}
    handled: set[str] = set()

    log(f"exp_drop_rate_eval started (gpus={args.gpus or 'auto'}, "
        f"max_parallel={args.max_parallel})")

    while True:
        for gpu, pid in list(pids.items()):
            if pid <= 0:
                continue
            rc = subprocess.run(["kill", "-0", str(pid)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode
            if rc != 0:
                j = running.pop(gpu, None)
                pids.pop(gpu, None)
                if j:
                    ok = has_eval_result(j["out_dir"])
                    log(f"DONE gpu={gpu} {j['name']} ok={ok}")

        running_names = {j["name"] for j in running.values()}
        pending = [j for j in discover_jobs()
                   if j["name"] not in handled and j["name"] not in running_names]

        if pending:
            idle = free_gpus()
            if allowed:
                idle = [g for g in idle if g in allowed]
            idle = [g for g in idle if g not in running]
            for gpu in idle:
                if not pending or len(running) >= args.max_parallel:
                    break
                j = pending.pop(0)
                log_merge = LOG_DIR / f"{j['name']}.merge.log"
                log_eval = LOG_DIR / f"{j['name']}.eval.log"
                running[gpu] = j
                pids[gpu] = -1
                log(f"PREP gpu={gpu} {j['name']} (merge then vllm)")
                ok = merge_adapter(j["adapter"], j["merged_dir"], log_merge)
                if not ok:
                    log(f"FAIL merge: {j['name']}")
                    running.pop(gpu, None); pids.pop(gpu, None)
                    handled.add(j["name"])
                    continue
                pid = launch_eval(gpu, j["merged_dir"], j["out_dir"], log_eval)
                pids[gpu] = pid
                handled.add(j["name"])
                log(f"LAUNCH eval(vllm) gpu={gpu} {j['name']} pid={pid}")

        if args.once and not running:
            break

        if (LOG_DIR / ".STOP_EVAL").exists():
            log("STOP_EVAL signal; waiting for running.")
            break

        time.sleep(args.poll)

    log("exp_drop_rate_eval finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
