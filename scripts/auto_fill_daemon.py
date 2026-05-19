#!/usr/bin/env python3
"""Auto-fill daemon for Phase C scout + lm-eval.

Behavior
--------
- Monitors all 8 GPUs every CHECK_INTERVAL seconds.
- A GPU is "idle" if memory.used < IDLE_MEM_MB.
- A GPU must stay idle for IDLE_GRACE seconds before being scheduled.
- Two queues, in priority order:
    1. lm_eval queue: any (model, dataset, method) seed42 with summary.json AND
       missing lm_eval_v2/ directory. Auto-discovered each pass.
    2. train queue: Phase C scout cells (5 models x 2 datasets x 5 methods),
       skipped if summary.json already exists.
- Stop conditions:
    - both queues empty AND all GPUs idle for >= IDLE_GRACE  -> exit
    - manual: touch /tmp/auto_fill_daemon.STOP

Logs
----
- Daemon master log: logs/scout/_daemon.log
- Each launched job: logs/scout/<name>.log

Each launch updates /tmp/auto_fill_daemon.state.json with:
   {gpu: {pid, started_at, kind, name}}

Usage
-----
nohup python scripts/auto_fill_daemon.py > logs/scout/_daemon.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ============ Config ============
ROOT = Path("/mnt/cpfs/junlongke/onlinelora/lora_obd")
PY = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"
LOG_DIR = ROOT / "logs" / "scout"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHECK_INTERVAL = 30          # seconds between GPU polls
IDLE_GRACE = 300             # 5 min stable idle before launching
IDLE_MEM_MB = 1500           # GPU.memory.used below this = "idle"
PER_GPU_LAUNCH_COOLDOWN = 60 # don't relaunch on the same GPU for X seconds
STATE_FILE = Path("/tmp/auto_fill_daemon.state.json")
STOP_FILE = Path("/tmp/auto_fill_daemon.STOP")

MODEL_CFG = {
    "olmo2-7b":      ("/mnt/cpfs/junlongke/onlinelora/models/OLMo-2-7B",                     "sdpa"),
    "r1-distill-7b": ("/mnt/cpfs/junlongke/onlinelora/models/R1-Distill-Qwen-7B",            "sdpa"),
    "llama3-8b":     ("/mnt/cpfs/public_data/public_model/Meta-Llama-3-8B",                  "sdpa"),
    "gemma3-12b":    ("/mnt/cpfs/junlongke/onlinelora/models/gemma-3-12b-it",                "eager"),
    # existing models also eligible for missing eval discovery
    "qwen3-8b":      ("/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B",                   "sdpa"),
    "qwen25-7b":     ("/mnt/cpfs/public_data/public_model/Qwen/Qwen2.5-7B",                  "sdpa"),
    "mistral-7b":    ("/mnt/cpfs/public_data/public_model/Mistral/Mistral-7B-v0.3",          "sdpa"),
}

NEW_MODELS = ["olmo2-7b", "r1-distill-7b", "llama3-8b", "gemma3-12b"]
DATASETS = ["metamathqa-10k", "tulu3-sft"]
METHODS = ["lora_vanilla", "relora_baseline", "relora_diag_gated_S3pos",
           "relora_random_drop", "dora", "cola"]

# methods that don't need merge_every (use 9999)
NO_MERGE = {"lora_vanilla", "dora", "adalora"}

# methods whose adapter/ may have been contaminated by the P0 bug; we
# require lm_eval_v3/ for these (lm_eval_v2/ is stale post-fix). Other
# methods (lora_vanilla, dora, adalora) are unaffected -> v2/v1 still valid.
MERGE_METHODS_FOR_V3 = {
    "relora_baseline", "relora_diag_gated_S3pos", "relora_diag_gated_S3neg",
    "relora_random_drop", "relora_train_gated", "cola",
}

# models needing larger generation budget (R1-Distill emits long <think>
# scratchpads before the boxed answer).
EVAL_MAX_NEW_TOKENS = {
    "r1-distill-7b": 1024,
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============ GPU monitoring ============
def gpu_state() -> list[dict]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, used, free = [int(x.strip()) for x in line.split(",")]
        rows.append({"idx": idx, "used_mb": used, "free_mb": free})
    return rows


# ============ State ============
@dataclass
class State:
    gpu_idle_since: dict[int, float]      # gpu -> ts when first seen idle (or 0 if not idle)
    gpu_last_launch: dict[int, float]     # gpu -> ts of last launch on this gpu
    launched: dict[int, dict]             # gpu -> {pid, kind, name, started_at}

    @classmethod
    def fresh(cls) -> "State":
        return cls(
            gpu_idle_since={i: 0.0 for i in range(8)},
            gpu_last_launch={i: 0.0 for i in range(8)},
            launched={},
        )

    def save(self):
        STATE_FILE.write_text(json.dumps({
            "launched": {str(k): v for k, v in self.launched.items()},
            "ts": time.time(),
        }, indent=2))


# ============ Queue: missing lm_eval ============
def pending_lm_evals() -> list[dict]:
    """Return cells that have summary.json but missing the appropriate eval dir.

    Merge-based methods (relora_*, S3pos/neg, random_drop, train_gated, cola)
    require lm_eval_v3/ because v1/v2 were produced from a contaminated
    adapter/ (lora_B=0). Non-merge methods (lora_vanilla, dora, adalora)
    are unaffected -> any prior lm_eval/ or lm_eval_v2/ counts.
    """
    busy = _running_eval_outs()
    pending = []
    for model_dir in (ROOT / "results" / "stage3_v2").iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if model not in MODEL_CFG:
            continue
        for ds_dir in model_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            for method_dir in ds_dir.iterdir():
                seed_dir = method_dir / "seed42"
                if not seed_dir.exists():
                    continue
                if not (seed_dir / "summary.json").exists():
                    continue
                method = method_dir.name
                is_merge = method in MERGE_METHODS_FOR_V3
                if is_merge:
                    target_name = "lm_eval_v3"
                    has_eval = (seed_dir / target_name).is_dir()
                else:
                    target_name = "lm_eval_v2"
                    has_eval = any(
                        p.name in ("lm_eval_v3", "lm_eval_v2", "lm_eval")
                        for p in seed_dir.iterdir() if p.is_dir()
                    )
                if has_eval:
                    continue
                adapter = seed_dir / "adapter"
                if not adapter.exists():
                    continue
                out_dir = seed_dir / target_name
                if str(out_dir.resolve()) in busy:
                    continue
                pending.append({
                    "kind": "eval",
                    "model": model,
                    "dataset": ds_dir.name,
                    "method": method,
                    "adapter": str(adapter),
                    "out_dir": str(out_dir),
                    "name": f"eval-{model}-{ds_dir.name}-{method}",
                })
    return pending


# ============ Queue: training scout ============
def _running_out_roots() -> set[str]:
    """Discover --out_root args of currently running stage3_run.py procs to avoid relaunching same cell."""
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return set()
    roots = set()
    for line in out.splitlines():
        if "stage3_run.py" not in line:
            continue
        toks = shlex.split(line)
        if "--out_root" in toks:
            i = toks.index("--out_root")
            if i + 1 < len(toks):
                roots.add(os.path.realpath(toks[i + 1]))
    return roots


def _running_eval_outs() -> set[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return set()
    outs = set()
    for line in out.splitlines():
        if "lm_eval" not in line:
            continue
        toks = shlex.split(line)
        if "--output_path" in toks:
            i = toks.index("--output_path")
            if i + 1 < len(toks):
                outs.add(os.path.realpath(toks[i + 1]))
    return outs


def pending_trainings() -> list[dict]:
    busy = _running_out_roots()
    pending = []
    for model in NEW_MODELS:
        for ds in DATASETS:
            for method in METHODS:
                seed_dir = ROOT / "results" / "stage3_v2" / model / ds / method / "seed42"
                if (seed_dir / "summary.json").exists():
                    continue
                if str(seed_dir.resolve()) in busy:
                    continue
                pending.append({
                    "kind": "train",
                    "model": model,
                    "dataset": ds,
                    "method": method,
                    "out_dir": str(seed_dir),
                    "name": f"train-{model}-{ds}-{method}",
                })
    return pending


# ============ Launchers ============
def launch_train(gpu: int, job: dict) -> Optional[int]:
    model_path, attn = MODEL_CFG[job["model"]]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['name']}.log"
    method = job["method"]
    steps = 800 if method == "dora" else 3000
    merge = 9999 if method in NO_MERGE else 500
    cmd = [
        PY, str(ROOT / "scripts" / "stage3_run.py"),
        "--model_path", model_path,
        "--model_key", job["model"],
        "--dataset", job["dataset"],
        "--method", method,
        "--total_steps", str(steps),
        "--merge_every", str(merge),
        "--eval_every", "250",
        "--ckpt_every", "50",
        "--saliency_max_seq_len", "512",
        "--attn_implementation", attn,
        "--save_adapter",
        "--seed", "42",
        "--out_root", str(out_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    log(f"LAUNCH train GPU={gpu} {job['name']} pid={proc.pid} steps={steps} -> {log_path.name}")
    return proc.pid


def launch_eval(gpu: int, job: dict) -> Optional[int]:
    model_path, _attn = MODEL_CFG[job["model"]]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['name']}.log"
    cmd = [
        PY, "-m", "lm_eval",
        "--model", "hf",
        "--model_args",
        f"pretrained={model_path},peft={job['adapter']},dtype=bfloat16,attn_implementation=sdpa,trust_remote_code=True",
        "--tasks", "gsm8k,hellaswag,arc_challenge",
        "--num_fewshot", "5",
        "--batch_size", "4",
        "--log_samples",
        "--output_path", str(out_dir),
    ]
    mnt = EVAL_MAX_NEW_TOKENS.get(job["model"])
    if mnt:
        cmd.extend(["--gen_kwargs", f"max_new_tokens={mnt}"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    log(f"LAUNCH eval  GPU={gpu} {job['name']} pid={proc.pid} -> {log_path.name}")
    return proc.pid


# ============ Main loop ============
def main():
    log("=== auto_fill_daemon starting ===")
    log(f"CHECK_INTERVAL={CHECK_INTERVAL}s IDLE_GRACE={IDLE_GRACE}s IDLE_MEM_MB={IDLE_MEM_MB}")
    log(f"STOP file: {STOP_FILE}")
    state = State.fresh()
    all_idle_since = 0.0

    while True:
        if STOP_FILE.exists():
            log("STOP file detected, exiting.")
            break

        now = time.time()
        gpus = gpu_state()
        free_idle_gpus = []
        for g in gpus:
            idx = g["idx"]
            if g["used_mb"] < IDLE_MEM_MB:
                if state.gpu_idle_since[idx] == 0.0:
                    state.gpu_idle_since[idx] = now
                    log(f"GPU {idx} entered idle (used={g['used_mb']}MB)")
                if (now - state.gpu_idle_since[idx]) >= IDLE_GRACE \
                   and (now - state.gpu_last_launch[idx]) >= PER_GPU_LAUNCH_COOLDOWN:
                    free_idle_gpus.append(idx)
            else:
                if state.gpu_idle_since[idx] != 0.0:
                    log(f"GPU {idx} no longer idle (used={g['used_mb']}MB)")
                state.gpu_idle_since[idx] = 0.0

        eval_q = pending_lm_evals()
        train_q = pending_trainings()

        if not free_idle_gpus and not eval_q and not train_q:
            # nothing to do; check global idle
            if all(g["used_mb"] < IDLE_MEM_MB for g in gpus):
                if all_idle_since == 0.0:
                    all_idle_since = now
                elif (now - all_idle_since) >= IDLE_GRACE:
                    log("All GPUs idle and queues empty; exiting.")
                    break
            else:
                all_idle_since = 0.0
        else:
            all_idle_since = 0.0

        for gpu in free_idle_gpus:
            # priority: eval first
            if eval_q:
                job = eval_q.pop(0)
                pid = launch_eval(gpu, job)
            elif train_q:
                job = train_q.pop(0)
                pid = launch_train(gpu, job)
            else:
                continue
            state.gpu_idle_since[gpu] = 0.0
            state.gpu_last_launch[gpu] = now
            state.launched[gpu] = {
                "pid": pid, "kind": job["kind"], "name": job["name"],
                "started_at": now,
            }
            state.save()

        time.sleep(CHECK_INTERVAL)

    log("=== auto_fill_daemon exiting ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(0)
