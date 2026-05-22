#!/usr/bin/env python3
"""Phase D auto-fill daemon: Qwen3 + Qwen3.5 multi-size sweep.

Differences from auto_fill_daemon.py:
- Two python interpreters:
    espo  (transformers 4.52, peft 0.17.0) — Qwen3 dense (Qwen3ForCausalLM)
    RRenv (transformers 5.3.0, peft 0.19.1) — Qwen3.5 (Qwen3_5ForCausalLM)
- Only tulu3-sft (per directive; metamath skipped, merge-arms structurally abort).
- 5-arm method set: lora_vanilla, relora_baseline, relora_diag_gated_S3pos, dora, cola.
- Wave 1 priority order (small → big):
    qwen35-0p8b → qwen35-2b → qwen3-1p7b → qwen35-4b → qwen3-4b → qwen35-9b → qwen3-14b
  Wave 2 (no cola):
    qwen3-32b, qwen35-27b
- Per-cell evidence: train_loss, val_loss, ER, CR, CN, saliency_at_merge, dropped_components,
  summary.json, lm_eval_v3/, adapter under best/.
- Eval target dir: lm_eval_v3/ for ALL methods (Phase D is fresh; v2/v1 don't exist).
- DOES NOT TOUCH non-Qwen models (gemma3 / mistral / llama3 / r1-distill / olmo2 / qwen25).
- DOES NOT auto-fill Wave 2 — needs PI signoff. Wave 1 only by default.

Stop:    touch /tmp/phase_d_daemon.STOP
State:   /tmp/phase_d_daemon.state.json
Logs:    logs/scout/_phase_d_daemon.log + logs/scout/<job>.log
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ============ Config ============
ROOT = Path("/mnt/cpfs/junlongke/onlinelora/lora_obd")
PY_ESPO  = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"     # Qwen3 dense
PY_RRENV = "/mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python"    # Qwen3.5
LOG_DIR = ROOT / "logs" / "scout"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHECK_INTERVAL = 30
IDLE_GRACE = 300
IDLE_MEM_MB = 1500
PER_GPU_LAUNCH_COOLDOWN = 60

DAEMON_LOG = LOG_DIR / "_phase_d_daemon.log"
STATE_FILE = Path("/tmp/phase_d_daemon.state.json")
STOP_FILE = Path("/tmp/phase_d_daemon.STOP")

DATASET = "tulu3-sft"
METHODS = ["lora_vanilla", "relora_baseline", "relora_diag_gated_S3pos", "dora", "cola"]
NO_MERGE = {"lora_vanilla", "dora", "adalora"}

# Wave 1 priority order (per directive). Each model -> (path, env, attn).
# Env "rrenv" = Qwen3.5 (transformers 5.x), "espo" = Qwen3 dense.
WAVE1_ORDER = [
    ("qwen35-0p8b", "/mnt/cpfs/junlongke/onlinelora/models/qwen35-0p8b",  "rrenv", "sdpa"),
    ("qwen35-2b",   "/mnt/cpfs/junlongke/onlinelora/models/qwen35-2b",    "rrenv", "sdpa"),
    ("qwen3-1p7b",  "/mnt/cpfs/junlongke/onlinelora/models/qwen3-1p7b",   "espo",  "sdpa"),
    ("qwen35-4b",   "/mnt/cpfs/junlongke/onlinelora/models/qwen35-4b",    "rrenv", "sdpa"),
    ("qwen3-4b",    "/mnt/cpfs/junlongke/onlinelora/models/qwen3-4b",     "espo",  "sdpa"),
    ("qwen35-9b",   "/mnt/cpfs/junlongke/onlinelora/models/qwen35-9b",    "rrenv", "sdpa"),
    ("qwen3-14b",   "/mnt/cpfs/junlongke/onlinelora/models/qwen3-14b",    "espo",  "sdpa"),
]

# All Phase D models indexed for eval discovery (lm-eval doesn't care about env once
# adapter is saved — both can be eval'd in the env that trained them).
MODEL_CFG = {slug: (path, env, attn) for slug, path, env, attn in WAVE1_ORDER}

# ============ Logging ============
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with DAEMON_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============ GPU monitor ============
def gpu_state() -> list[dict]:
    """Return list of {idx, used_mb} for all visible GPUs."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=10
        )
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return []
    res = []
    for line in out.strip().splitlines():
        try:
            idx_s, mem_s = [t.strip() for t in line.split(",")]
            res.append({"idx": int(idx_s), "used_mb": int(mem_s)})
        except Exception:
            continue
    return res


# ============ Running-process discovery (dedupe) ============
def _ps_lines() -> list[list[str]]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True, timeout=10)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        try:
            rows.append(shlex.split(line))
        except Exception:
            pass
    return rows


def _running_train_out_roots() -> set[str]:
    roots = set()
    for toks in _ps_lines():
        if not any("stage3_run.py" in t for t in toks):
            continue
        if "--out_root" in toks:
            i = toks.index("--out_root")
            if i + 1 < len(toks):
                roots.add(os.path.realpath(toks[i + 1]))
    return roots


def _running_eval_outs() -> set[str]:
    outs = set()
    for toks in _ps_lines():
        if not any("lm_eval" in t for t in toks):
            continue
        if "--output_path" in toks:
            i = toks.index("--output_path")
            if i + 1 < len(toks):
                outs.add(os.path.realpath(toks[i + 1]))
    return outs


# ============ Per-job retry cap (avoid runaway loops on persistent failures) ============
MAX_LAUNCH_ATTEMPTS = 3
launch_attempts: dict[str, int] = {}      # job name -> count
launch_blacklist: set[str] = set()        # job name permanently skipped


def _preflight_check(job: dict) -> Optional[str]:
    """Return None if OK, else a string reason to skip. Cached failures go into blacklist.

    For Phase D evals we now run vLLM-on-merged-weights with the RRenv interpreter,
    so we need lm_eval + datasets(>=4) + vllm all importable in RRenv. Merge runs in
    espo, so we also check peft + transformers there.
    """
    if job["kind"] == "eval":
        # RRenv: lm_eval + datasets + vllm
        try:
            r = subprocess.run(
                [PY_RRENV, "-c",
                 "import lm_eval, datasets, vllm; "
                 "from datasets.features.features import _FEATURE_TYPES; "
                 "assert 'List' in _FEATURE_TYPES, 'datasets too old'; "
                 "print('rrenv ok')"],
                capture_output=True, timeout=60, text=True
            )
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").splitlines()[-1] if (r.stderr or r.stdout) else "unknown"
                return f"RRenv preflight failed: {tail}"
        except Exception as e:
            return f"RRenv preflight subprocess failed: {e}"

        # espo: peft + transformers (for merge)
        try:
            r = subprocess.run(
                [PY_ESPO, "-c", "import peft, transformers; print('espo ok')"],
                capture_output=True, timeout=30, text=True
            )
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").splitlines()[-1] if (r.stderr or r.stdout) else "unknown"
                return f"espo preflight failed: {tail}"
        except Exception as e:
            return f"espo preflight subprocess failed: {e}"
    return None


# ============ Queue: pending lm-evals (Phase D = lm_eval_v3/) ============
def pending_lm_evals() -> list[dict]:
    """For every Phase D cell with summary.json + adapter/, schedule lm_eval_v3/ if missing."""
    busy = _running_eval_outs()
    pending = []
    for slug, (path, env, attn) in MODEL_CFG.items():
        for method in METHODS:
            seed_dir = ROOT / "results" / "stage3_v2" / slug / DATASET / method / "seed42"
            if not (seed_dir / "summary.json").exists():
                continue
            adapter = seed_dir / "adapter"
            if not adapter.exists():
                continue
            out_dir = seed_dir / "lm_eval_v3"
            if out_dir.exists():
                # already evaluated; check if it has results json
                if any(out_dir.rglob("results_*.json")):
                    continue
            if str(out_dir.resolve()) in busy:
                continue
            name = f"eval-{slug}-{DATASET}-{method}"
            if name in launch_blacklist:
                continue
            if launch_attempts.get(name, 0) >= MAX_LAUNCH_ATTEMPTS:
                launch_blacklist.add(name)
                continue
            pending.append({
                "kind": "eval", "model": slug, "method": method,
                "model_path": path, "env": env, "attn": attn,
                "adapter": str(adapter), "out_dir": str(out_dir),
                "name": name,
            })
    return pending


# ============ Queue: pending Wave 1 trainings (priority by WAVE1_ORDER) ============
def pending_trainings() -> list[dict]:
    busy = _running_train_out_roots()
    pending = []
    for slug, path, env, attn in WAVE1_ORDER:
        for method in METHODS:
            seed_dir = ROOT / "results" / "stage3_v2" / slug / DATASET / method / "seed42"
            if (seed_dir / "summary.json").exists():
                continue
            if str(seed_dir.resolve()) in busy:
                continue
            pending.append({
                "kind": "train", "model": slug, "method": method,
                "model_path": path, "env": env, "attn": attn,
                "out_dir": str(seed_dir),
                "name": f"train-{slug}-{DATASET}-{method}",
            })
    return pending


# ============ Launchers ============
def _py_for(env: str) -> str:
    return PY_RRENV if env == "rrenv" else PY_ESPO


def launch_train(gpu: int, job: dict) -> Optional[int]:
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['name']}.log"
    method = job["method"]
    steps = 800 if method == "dora" else 3000
    merge = 9999 if method in NO_MERGE else 750  # Phase D uses 750 (4 stages of 750)
    cmd = [
        _py_for(job["env"]), str(ROOT / "scripts" / "stage3_run.py"),
        "--model_path", job["model_path"],
        "--model_key", job["model"],
        "--dataset", DATASET,
        "--method", method,
        "--total_steps", str(steps),
        "--merge_every", str(merge),
        "--eval_every", "250",
        "--ckpt_every", "250",
        "--saliency_max_seq_len", "512",
        "--attn_implementation", job["attn"],
        "--save_adapter",
        "--seed", "42",
        "--out_root", str(out_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    log(f"LAUNCH train GPU={gpu} env={job['env']} {job['name']} pid={proc.pid} steps={steps}")
    return proc.pid


def launch_eval(gpu: int, job: dict) -> Optional[int]:
    """vLLM-on-merged-weights eval launcher.

    1. Merge adapter into a private bf16 dump under <seed_dir>/merged/ (if not already).
       Merge is fast (~15s for 1.7B, scales linearly). Done with espo interpreter
       (so we avoid the RRenv transformers-5.x deepspeed CUDA_HOME quirk for now).
    2. Launch lm-eval with --model vllm pretrained=<merged>. No PEFT runtime, no
       DoRA-incompatibility, batch_size=auto for max throughput.
    """
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = out_dir.parent  # .../seed42
    merged_dir = seed_dir / "merged"
    log_path = LOG_DIR / f"{job['name']}.log"

    # ----- Step 1: merge if needed -----
    sentinel = merged_dir / ".merge.done"
    needs_merge = (not sentinel.exists()) or (not (merged_dir / "config.json").exists())
    if needs_merge:
        merge_log = LOG_DIR / f"{job['name']}.merge.log"
        merge_cmd = [
            PY_ESPO, str(ROOT / "scripts" / "merge_adapter.py"),
            "--base", job["model_path"],
            "--adapter", job["adapter"],
            "--out", str(merged_dir),
        ]
        m_env = os.environ.copy()
        m_env["CUDA_HOME"] = "/usr/local/cuda-12"
        m_env["CUDA_VISIBLE_DEVICES"] = ""  # merge on CPU; saves the GPU for vLLM
        log(f"MERGE  {job['name']} (cpu) -> {merged_dir}")
        try:
            with merge_log.open("w") as f:
                rc = subprocess.run(merge_cmd, env=m_env, stdout=f,
                                    stderr=subprocess.STDOUT, cwd=str(ROOT),
                                    timeout=900).returncode
        except subprocess.TimeoutExpired:
            log(f"MERGE TIMEOUT {job['name']} after 900s")
            return None
        if rc != 0:
            log(f"MERGE FAIL {job['name']} rc={rc} (see {merge_log})")
            return None

    # ----- Step 2: vLLM eval (always RRenv interpreter; vLLM 0.15.1 only there) -----
    cmd = [
        PY_RRENV, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args",
        f"pretrained={merged_dir},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,trust_remote_code=True",
        "--tasks", "gsm8k,hellaswag,arc_challenge",
        "--num_fewshot", "5",
        "--batch_size", "auto",
        "--log_samples",
        "--output_path", str(out_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), preexec_fn=os.setsid)
    log(f"LAUNCH eval(vllm) GPU={gpu} {job['name']} pid={proc.pid}")
    return proc.pid


# ============ State ============
def save_state(d: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(d, default=str, indent=2))
    except Exception:
        pass


# ============ Main loop ============
def main() -> None:
    log("=== phase_d_daemon starting ===")
    log(f"CHECK_INTERVAL={CHECK_INTERVAL}s IDLE_GRACE={IDLE_GRACE}s IDLE_MEM_MB={IDLE_MEM_MB}")
    log(f"STOP file: {STOP_FILE}")
    log(f"Models in priority order: {[s for s,_,_,_ in WAVE1_ORDER]}")

    gpu_idle_since: dict[int, float] = {}
    gpu_last_launch: dict[int, float] = {}
    launched: dict[int, dict] = {}
    all_idle_since = 0.0

    while True:
        if STOP_FILE.exists():
            log("STOP file detected, exiting.")
            break

        now = time.time()
        gpus = gpu_state()
        free_idle = []
        for g in gpus:
            idx = g["idx"]
            if g["used_mb"] < IDLE_MEM_MB:
                if gpu_idle_since.get(idx, 0.0) == 0.0:
                    gpu_idle_since[idx] = now
                    log(f"GPU {idx} entered idle (used={g['used_mb']}MB)")
                idle_for = now - gpu_idle_since.get(idx, now)
                cooldown_ok = (now - gpu_last_launch.get(idx, 0.0)) >= PER_GPU_LAUNCH_COOLDOWN
                if idle_for >= IDLE_GRACE and cooldown_ok:
                    free_idle.append(idx)
            else:
                if gpu_idle_since.get(idx, 0.0) != 0.0:
                    log(f"GPU {idx} no longer idle (used={g['used_mb']}MB)")
                gpu_idle_since[idx] = 0.0

        eval_q = pending_lm_evals()
        train_q = pending_trainings()

        if not free_idle and not eval_q and not train_q:
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

        for gpu in free_idle:
            job = None
            while eval_q:
                cand = eval_q.pop(0)
                reason = _preflight_check(cand)
                if reason is not None:
                    launch_attempts[cand["name"]] = launch_attempts.get(cand["name"], 0) + 1
                    log(f"SKIP {cand['name']}: {reason} (attempt {launch_attempts[cand['name']]}/{MAX_LAUNCH_ATTEMPTS})")
                    if launch_attempts[cand["name"]] >= MAX_LAUNCH_ATTEMPTS:
                        launch_blacklist.add(cand["name"])
                        log(f"BLACKLIST {cand['name']} after {MAX_LAUNCH_ATTEMPTS} preflight failures")
                    continue
                job = cand
                pid = launch_eval(gpu, job)
                break
            if job is None and train_q:
                job = train_q.pop(0)
                pid = launch_train(gpu, job)
            if job is None:
                continue
            launch_attempts[job["name"]] = launch_attempts.get(job["name"], 0) + 1
            if launch_attempts[job["name"]] >= MAX_LAUNCH_ATTEMPTS and job["kind"] == "eval":
                pass  # we'll let it run; on next scan if results json is missing it stays out
            gpu_idle_since[gpu] = 0.0
            gpu_last_launch[gpu] = now
            launched[gpu] = {
                "pid": pid, "kind": job["kind"], "name": job["name"],
                "started_at": now,
            }
            save_state({"launched": launched, "idle_since": gpu_idle_since,
                        "eval_q_size": len(eval_q), "train_q_size": len(train_q),
                        "blacklist": sorted(launch_blacklist),
                        "attempts": launch_attempts})

        time.sleep(CHECK_INTERVAL)

    log("=== phase_d_daemon exiting ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(0)
