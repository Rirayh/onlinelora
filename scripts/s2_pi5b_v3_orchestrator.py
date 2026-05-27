#!/usr/bin/env python3
"""6-cell re-train orchestrator with PI #5b Option 3 ckpt semantics.

Per PI feedback #5b Action E.1: re-train the 6 most important cells with
--save_merged_final to capture the post-merge base for offline lm-eval.

Cells:
  1. v1_S3pos              relora_diag_gated_S3pos                          - primary
  2. v2_S3pos_IG_FDR       relora_diag_gated_S3pos + saliency_v2            - alt estimator
  3. random_dr0.5          relora_random_drop dr=0.5                        - best Exp-1 random
  4. random_dr0.3          relora_random_drop dr=0.3                        - moderate random
  5. relora_baseline       relora_baseline (no drop policy)                 - control
  6. lora_vanilla          lora_vanilla (no merging)                        - control

Settings: qwen3-8b/tulu3-sft, total_steps=3000, merge_every=750, seed=42.
Outputs: results/s2_pi5b_v3/qwen3-8b/tulu3-sft/<cell>/seed42/

Each cell ~9.5h on 1 GPU. With 6 GPUs in parallel, wall-clock ~10h.
Default exclude GPU 0 (reserved for smoke + eval pipeline); use GPUs 1-6.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "s2_pi5b_v3"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG = LOG_DIR / "orchestrator.log"

PY_ESPO = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"

MODEL = "qwen3-8b"
MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
DATASET = "tulu3-sft"
SEED = 42
ATTN = "sdpa"

CELLS = [
    ("v1_S3pos", "relora_diag_gated_S3pos", []),
    ("v2_S3pos_IG_FDR", "relora_diag_gated_S3pos",
     ["--saliency_estimator", "v2",
      "--saliency_v2_m_ig", "4",
      "--saliency_v2_alpha", "0.2"]),
    ("random_dr0.5", "relora_random_drop",
     ["--random_drop_rate", "0.5"]),
    ("random_dr0.3", "relora_random_drop",
     ["--random_drop_rate", "0.3"]),
    ("relora_baseline", "relora_baseline", []),
    ("lora_vanilla", "lora_vanilla", []),
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
    for ln in out.strip().splitlines():
        i, u = [s.strip() for s in ln.split(",")]
        rs.append({"idx": int(i), "used_mb": int(u)})
    return rs


def free_gpus(threshold_mb: int = 2000, exclude=None) -> list[int]:
    excl = set(exclude or [])
    return [g["idx"] for g in gpu_state()
            if g["used_mb"] < threshold_mb and g["idx"] not in excl]


def build_cmd(method: str, extra: list[str], out_dir: Path) -> list[str]:
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
        "--seed", str(SEED),
        "--out_root", str(out_dir),
    ]
    return base + extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="",
                    help="Comma-sep GPU ids; empty=auto-detect free.")
    ap.add_argument("--exclude_gpus", default="0",
                    help="GPUs to reserve (default: 0 for smoke/eval).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_root_base = ROOT / "results" / "s2_pi5b_v3" / MODEL / DATASET
    out_root_base.mkdir(parents=True, exist_ok=True)

    excl = [int(x) for x in args.exclude_gpus.split(",") if x.strip()]
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        gpus = free_gpus(exclude=excl)
        log(f"auto-detected free GPUs (excl={excl}): {gpus}")

    if len(gpus) < len(CELLS):
        log(f"WARN: have {len(gpus)} free GPUs but {len(CELLS)} cells; "
            f"will only launch first {len(gpus)}")

    pending = []
    for cell_label, method, extra in CELLS:
        out_dir = out_root_base / cell_label / f"seed{SEED}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "merged_final" / "config.json").exists():
            log(f"SKIP (merged_final exists): {cell_label}")
            continue
        pending.append((cell_label, method, extra, out_dir))

    if not pending:
        log("All cells have merged_final/; nothing to do.")
        return 0

    log(f"pending cells ({len(pending)}): {[p[0] for p in pending]}")

    procs = []
    for i, (cell_label, method, extra, out_dir) in enumerate(pending):
        if i >= len(gpus):
            log(f"ran out of GPUs after {i} cells; remaining: "
                f"{[p[0] for p in pending[i:]]}")
            break
        gpu = gpus[i]
        cmd = build_cmd(method, extra, out_dir)
        log_path = LOG_DIR / f"{cell_label}.train.log"
        log(f"launch {cell_label} on GPU {gpu} -> {log_path}")
        if args.dry_run:
            log(f"  cmd: {' '.join(cmd)}")
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=f,
                                    stderr=subprocess.STDOUT,
                                    cwd=str(ROOT), preexec_fn=os.setsid)
        procs.append((cell_label, gpu, proc.pid))

    if args.dry_run:
        log("dry-run done.")
        return 0

    log(f"launched {len(procs)} jobs:")
    for cell, gpu, pid in procs:
        log(f"  PID={pid} GPU={gpu} cell={cell}")
    log("orchestrator detaches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
