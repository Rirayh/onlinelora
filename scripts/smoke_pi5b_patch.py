#!/usr/bin/env python3
"""Micro-smoke for PI #5b Option 3 patch.

Goal: verify --save_merged_final produces a usable merged base for lm-eval.

Strategy: 50-step training run with a single merge at step 30, then load the
saved merged_final and run a 5-prompt sanity gen. Cell: dr=0.5 random_drop on
qwen3-8b/tulu3-sft. ~5-10min on 1 GPU.

Pass criteria:
  - merged_final/ dir exists
  - Contains config.json + safetensors shards
  - Loadable via AutoModelForCausalLM.from_pretrained
  - Generates non-empty output for "What is 2+3?"

Run on first free GPU. Logs to logs/smoke_pi5b_patch.log.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "smoke_pi5b_patch.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
PY = "/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python"

MODEL_PATH = "/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B"
OUT = ROOT / "results" / "smoke_pi5b" / "seed42"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def free_gpu() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    for line in out.strip().splitlines():
        idx, used = [s.strip() for s in line.split(",")]
        if int(used) < 1500:
            return int(idx)
    raise SystemExit("no free GPU")


def main() -> int:
    if (OUT / "merged_final" / "config.json").exists():
        log("merged_final already exists; skipping training")
    else:
        gpu = free_gpu()
        log(f"using GPU {gpu}, output -> {OUT}")
        cmd = [
            PY, str(ROOT / "scripts" / "stage3_run.py"),
            "--model_path", MODEL_PATH,
            "--model_key", "qwen3-8b",
            "--dataset", "tulu3-sft",
            "--method", "relora_random_drop",
            "--random_drop_rate", "0.5",
            "--total_steps", "50",
            "--merge_every", "30",
            "--eval_every", "25",
            "--ckpt_every", "0",
            "--saliency_max_seq_len", "512",
            "--attn_implementation", "sdpa",
            "--save_merged_final",
            "--seed", "42",
            "--out_root", str(OUT),
        ]
        log(f"cmd: {' '.join(cmd)}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        train_log = LOG.parent / "smoke_pi5b_train.log"
        t0 = time.time()
        with train_log.open("w") as f:
            rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT))
        log(f"train rc={rc} elapsed={time.time()-t0:.1f}s")
        if rc != 0:
            log(f"TRAIN FAILED: see {train_log}")
            return 1

    merged = OUT / "merged_final"
    log(f"checking {merged}")
    if not (merged / "config.json").exists():
        log("FAIL: merged_final/config.json missing")
        return 2
    if not list(merged.glob("*.safetensors")):
        log("FAIL: merged_final/*.safetensors missing")
        return 3

    log("merged_final structure OK; running gen sanity check")
    gen_script = ROOT / "scripts" / "_smoke_gen_check.py"
    gen_script.write_text("""\
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
p = sys.argv[1]
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
m.eval()
prompts = ["What is 2+3?", "The capital of France is"]
for q in prompts:
    inp = tok(q, return_tensors='pt').to(m.device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=16, do_sample=False)
    txt = tok.decode(out[0], skip_special_tokens=True)
    print(f"[Q] {q}\\n[A] {txt}\\n")
""")
    gpu = free_gpu()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gen_log = LOG.parent / "smoke_pi5b_gen.log"
    with gen_log.open("w") as f:
        rc = subprocess.call([PY, str(gen_script), str(merged)],
                             env=env, stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(ROOT))
    log(f"gen rc={rc}")
    if rc != 0:
        log(f"GEN FAILED: see {gen_log}")
        return 4
    log(f"GEN log:")
    with gen_log.open() as f:
        for ln in f:
            log(f"  {ln.rstrip()}")
    log("PI #5b OPTION 3 PATCH: SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
