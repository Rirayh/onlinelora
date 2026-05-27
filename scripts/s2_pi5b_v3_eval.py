#!/usr/bin/env python3
"""lm-eval orchestrator for PI #5b 6-cell re-train results.

Each cell has a merged_final/ directory (post-all-merges base). This script
runs lm_eval (vllm backend) on each, in parallel across GPUs.

Tasks: gsm8k, hellaswag, arc_challenge (5-shot, same config as all prior evals).
Outputs: results/s2_pi5b_v3/qwen3-8b/tulu3-sft/<cell>/seed42/lm_eval/

Usage:
  nohup python scripts/s2_pi5b_v3_eval.py > logs/s2_pi5b_v3/eval_orch.log 2>&1 &
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "s2_pi5b_v3"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "eval_orchestrator.log"

PY_RRENV = "/mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python"
EXP_ROOT = ROOT / "results" / "s2_pi5b_v3" / "qwen3-8b" / "tulu3-sft"

CELLS = [
    "v1_S3pos",
    "v2_S3pos_IG_FDR",
    "random_dr0.5",
    "random_dr0.3",
    "relora_baseline",
    "lora_vanilla",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with DAEMON_LOG.open("a") as f:
        f.write(line + "\n")


def free_gpus(threshold_mb: int = 1500) -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    return [int(ln.split(",")[0].strip())
            for ln in out.strip().splitlines()
            if int(ln.split(",")[1].strip()) < threshold_mb]


def has_result(out_dir: Path) -> bool:
    return out_dir.exists() and any(out_dir.rglob("results_*.json"))


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
    out_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    return proc.pid


def main() -> int:
    log("=== PI #5b lm-eval orchestrator started ===")
    gpus = free_gpus()
    log(f"free GPUs: {gpus}")

    jobs = []
    for cell in CELLS:
        seed_dir = EXP_ROOT / cell / "seed42"
        merged_dir = seed_dir / "merged_final"
        lm_eval_dir = seed_dir / "lm_eval"
        if not (merged_dir / "model.safetensors.index.json").exists():
            log(f"SKIP {cell}: merged_final missing")
            continue
        if has_result(lm_eval_dir):
            log(f"SKIP {cell}: eval result already exists")
            continue
        jobs.append({"name": cell, "merged": merged_dir, "out": lm_eval_dir})

    if not jobs:
        log("All cells already have eval results.")
        return 0

    log(f"pending: {[j['name'] for j in jobs]}")

    if len(gpus) < len(jobs):
        log(f"WARN: {len(gpus)} GPUs for {len(jobs)} jobs; some will be queued")

    procs: list[tuple[str, int, int]] = []
    for i, job in enumerate(jobs):
        if i >= len(gpus):
            log(f"  no GPU for {job['name']}; skipping (re-run later)")
            continue
        gpu = gpus[i]
        log_path = LOG_DIR / f"{job['name']}.eval.log"
        pid = launch_eval(gpu, job["merged"], job["out"], log_path)
        procs.append((job["name"], gpu, pid))
        log(f"  LAUNCH {job['name']} gpu={gpu} pid={pid}")

    log(f"launched {len(procs)} evals. Monitoring...")

    # Poll until all done
    while True:
        time.sleep(60)
        alive = []
        for cell, gpu, pid in procs:
            try:
                os.kill(pid, 0)
                alive.append((cell, gpu, pid))
            except ProcessLookupError:
                done = has_result(EXP_ROOT / cell / "seed42" / "lm_eval")
                log(f"  {cell} (pid={pid}) exited; result={'OK' if done else 'MISSING'}")
        procs = alive
        if not procs:
            break
        log(f"  still running: {[c for c,_,_ in procs]}")

    log("=== all evals done ===")

    # Print score summary
    import json
    log("\n=== SCORE SUMMARY ===")
    rows = []
    for cell in CELLS:
        lm_eval_dir = EXP_ROOT / cell / "seed42" / "lm_eval"
        results = list(lm_eval_dir.rglob("results_*.json")) if lm_eval_dir.exists() else []
        if not results:
            rows.append(f"  {cell:25s}  NO RESULT")
            continue
        d = json.loads(results[0].read_text())
        r = d.get("results", {})
        gsm_s = r.get("gsm8k", {}).get("exact_match,strict-match", r.get("gsm8k", {}).get("exact_match", "?"))
        gsm_f = r.get("gsm8k", {}).get("exact_match,flexible-extract", "?")
        hsw   = r.get("hellaswag", {}).get("acc_norm,none", r.get("hellaswag", {}).get("acc_norm", "?"))
        arc   = r.get("arc_challenge", {}).get("acc_norm,none", r.get("arc_challenge", {}).get("acc_norm", "?"))
        def fmt(v):
            return f"{v*100:.2f}" if isinstance(v, float) else str(v)
        rows.append(f"  {cell:25s}  gsm8k_strict={fmt(gsm_s)}  gsm8k_flex={fmt(gsm_f)}  hellaswag={fmt(hsw)}  arc_c={fmt(arc)}")
    for row in rows:
        log(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
