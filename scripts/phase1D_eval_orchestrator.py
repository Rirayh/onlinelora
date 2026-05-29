#!/usr/bin/env python3
"""lm-eval orchestrator for Phase 1 robustness sweep + Phase D overtrain (PI #6).

Phase 1: results/phase1_robustness/qwen3-8b/tulu3-sft/<cell>/seed{42,43,44}/merged_final/
Phase D:  results/phase_d/qwen3-8b/tulu3-sft/<cell>/seed{42,43}/merged_final/

Tasks: gsm8k, hellaswag, arc_challenge, mmlu, ifeval (5-shot for all except
       ifeval which is 0-shot).

Output: lm_eval/ subdir inside each seed dir.

Usage (run after training completes):
  nohup python scripts/phase1D_eval_orchestrator.py [--gpus 0,1,2,...] \
      [--phase1] [--phaseD] > logs/phase1D_eval.log 2>&1 &
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
LOG_DIR = ROOT / "logs" / "phase1D_eval"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "eval_orchestrator.log"

PY_RRENV = "/mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python"

MODEL = "qwen3-8b"
DATASET = "tulu3-sft"

PHASE1_CELLS   = ["v1_S3pos", "random_dr0.5", "relora_baseline"]
PHASE1_SEEDS   = [42, 43, 44]
PHASED_CELLS   = ["lora_vanilla", "v1_S3pos"]
PHASED_SEEDS   = [42, 43]
PHASE1P5_CELLS = ["random_anneal_up", "random_anneal_down",
                  "random_triangle_up_down", "random_triangle_down_up"]
PHASE1P5_SEEDS = [42]

TASKS_5SHOT   = "gsm8k,hellaswag,arc_challenge,mmlu"
TASKS_0SHOT   = "ifeval"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with DAEMON_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


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


def launch_eval(gpu: int, merged_dir: Path, out_dir: Path,
                log_path: Path, tasks: str, fewshot: int) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_HOME"] = "/usr/local/cuda-12"
    cmd = [
        PY_RRENV, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args",
        (f"pretrained={merged_dir},dtype=bfloat16,"
         f"gpu_memory_utilization=0.85,max_model_len=4096,"
         f"trust_remote_code=True"),
        "--tasks", tasks,
        "--num_fewshot", str(fewshot),
        "--batch_size", "auto",
        "--log_samples",
        "--output_path", str(out_dir),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    return proc.pid


def collect_jobs(phase1: bool, phaseD: bool, phase1p5: bool = False) -> list[dict]:
    jobs = []
    if phase1:
        base = ROOT / "results" / "phase1_robustness" / MODEL / DATASET
        for cell in PHASE1_CELLS:
            for seed in PHASE1_SEEDS:
                seed_dir  = base / cell / f"seed{seed}"
                merged    = seed_dir / "merged_final"
                lm_dir    = seed_dir / "lm_eval"
                if not (merged / "config.json").exists():
                    log(f"SKIP phase1 {cell}/seed{seed}: merged_final missing")
                    continue
                if has_result(lm_dir):
                    log(f"SKIP phase1 {cell}/seed{seed}: result exists")
                    continue
                jobs.append({"label": f"p1/{cell}/s{seed}",
                             "merged": merged, "out": lm_dir})
    if phase1p5:
        base = ROOT / "results" / "phase1p5_schedule_ablation" / MODEL / DATASET
        for cell in PHASE1P5_CELLS:
            for seed in PHASE1P5_SEEDS:
                seed_dir  = base / cell / f"seed{seed}"
                merged    = seed_dir / "merged_final"
                lm_dir    = seed_dir / "lm_eval"
                if not (merged / "config.json").exists():
                    log(f"SKIP phase1p5 {cell}/seed{seed}: merged_final missing")
                    continue
                if has_result(lm_dir):
                    log(f"SKIP phase1p5 {cell}/seed{seed}: result exists")
                    continue
                jobs.append({"label": f"p1p5/{cell}/s{seed}",
                             "merged": merged, "out": lm_dir})
    if phaseD:
        base = ROOT / "results" / "phase_d" / MODEL / DATASET
        for cell in PHASED_CELLS:
            for seed in PHASED_SEEDS:
                seed_dir  = base / cell / f"seed{seed}"
                merged    = seed_dir / "merged_final"
                lm_dir    = seed_dir / "lm_eval"
                if not (merged / "config.json").exists():
                    log(f"SKIP phaseD {cell}/seed{seed}: merged_final missing")
                    continue
                if has_result(lm_dir):
                    log(f"SKIP phaseD {cell}/seed{seed}: result exists")
                    continue
                jobs.append({"label": f"pD/{cell}/s{seed}",
                             "merged": merged, "out": lm_dir})
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="",
                    help="Comma-sep GPU ids; empty=auto-detect free.")
    ap.add_argument("--phase1",   action="store_true", default=False)
    ap.add_argument("--phase1p5", action="store_true", default=False)
    ap.add_argument("--phaseD",   action="store_true", default=False)
    ap.add_argument("--all", action="store_true", default=False,
                    help="Eval phase1 + phase1p5 + phaseD.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    do_p1   = args.phase1   or args.all
    do_p1p5 = args.phase1p5 or args.all
    do_pD   = args.phaseD   or args.all
    if not do_p1 and not do_p1p5 and not do_pD:
        print("Specify --phase1, --phase1p5, --phaseD, or --all", file=sys.stderr)
        return 1

    log(f"=== Phase1/1p5/D eval orchestrator started "
        f"(phase1={do_p1} phase1p5={do_p1p5} phaseD={do_pD}) ===")

    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        gpus = free_gpus()
        log(f"auto-detected free GPUs: {gpus}")

    jobs = collect_jobs(do_p1, do_pD, do_p1p5)
    if not jobs:
        log("Nothing to eval.")
        return 0

    log(f"pending ({len(jobs)}): {[j['label'] for j in jobs]}")

    if len(gpus) < len(jobs):
        log(f"WARN: {len(gpus)} GPUs for {len(jobs)} jobs; "
            f"will launch first {len(gpus)}, re-run for rest.")

    procs: list[tuple[str, int, int]] = []
    for i, job in enumerate(jobs):
        if i >= len(gpus):
            log(f"no GPU for {job['label']}; re-run after GPUs free.")
            continue
        if args.dry_run:
            log(f"  DRY {job['label']} gpu={gpus[i]}")
            continue
        gpu = gpus[i]
        log_path = LOG_DIR / (job["label"].replace("/", "_") + ".eval.log")
        pid = launch_eval(gpu, job["merged"], job["out"], log_path,
                          TASKS_5SHOT + "," + TASKS_0SHOT, 5)
        procs.append((job["label"], gpu, pid))
        log(f"  LAUNCH {job['label']} gpu={gpu} pid={pid}")

    if args.dry_run:
        log("dry-run done.")
        return 0

    log(f"launched {len(procs)}. Monitoring...")
    while True:
        time.sleep(60)
        alive = []
        for label, gpu, pid in procs:
            try:
                os.kill(pid, 0)
                alive.append((label, gpu, pid))
            except ProcessLookupError:
                log(f"  {label} (pid={pid}) exited.")
        procs = alive
        if not procs:
            break
        log(f"  still running: {[l for l,_,_ in procs]}")

    log("=== all evals done ===")

    _print_summary(do_p1, do_p1p5, do_pD)
    return 0


def _print_summary(phase1: bool, phase1p5: bool, phaseD: bool) -> None:
    rows = []

    def _parse(seed_dir: Path, label: str) -> None:
        results = list((seed_dir / "lm_eval").rglob("results_*.json"))
        if not results:
            rows.append(f"  {label:40s}  NO RESULT")
            return
        d = json.loads(results[0].read_text())
        r = d.get("results", {})

        def g(task, key, fallback="?"):
            v = r.get(task, {}).get(key, r.get(task, {}).get(key.split(",")[0], fallback))
            return f"{v*100:.2f}" if isinstance(v, float) else str(v)

        gsm_s = g("gsm8k",        "exact_match,strict-match")
        gsm_f = g("gsm8k",        "exact_match,flexible-extract")
        hsw   = g("hellaswag",    "acc_norm,none")
        arc   = g("arc_challenge","acc_norm,none")
        mmlu  = g("mmlu",         "acc,none")
        ife   = g("ifeval",       "prompt_level_strict_acc,none")
        rows.append(f"  {label:40s}  gsm_s={gsm_s}  gsm_f={gsm_f}  "
                    f"hsw={hsw}  arc={arc}  mmlu={mmlu}  ifeval={ife}")

    if phase1:
        base = ROOT / "results" / "phase1_robustness" / MODEL / DATASET
        for cell in PHASE1_CELLS:
            for seed in PHASE1_SEEDS:
                _parse(base / cell / f"seed{seed}", f"p1/{cell}/s{seed}")
    if phase1p5:
        base = ROOT / "results" / "phase1p5_schedule_ablation" / MODEL / DATASET
        for cell in PHASE1P5_CELLS:
            for seed in PHASE1P5_SEEDS:
                _parse(base / cell / f"seed{seed}", f"p1p5/{cell}/s{seed}")
    if phaseD:
        base = ROOT / "results" / "phase_d" / MODEL / DATASET
        for cell in PHASED_CELLS:
            for seed in PHASED_SEEDS:
                _parse(base / cell / f"seed{seed}", f"pD/{cell}/s{seed}")

    log("\n=== SCORE SUMMARY ===")
    for row in rows:
        log(row)


if __name__ == "__main__":
    sys.exit(main())
