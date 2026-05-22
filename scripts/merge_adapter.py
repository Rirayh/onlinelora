#!/usr/bin/env python3
"""Merge a PEFT adapter (LoRA / DoRA) into its base model and save bf16.

Usage:
  python scripts/merge_adapter.py \
      --base /mnt/cpfs/junlongke/onlinelora/models/qwen3-1p7b \
      --adapter <results_dir>/adapter \
      --out    <results_dir>/merged

Notes:
  - Output dir is suitable for: lm-eval --model vllm --model_args pretrained=<out>
  - Idempotent: if `<out>/.merge.done` exists with same adapter mtime, skip.
  - Uses the espo interpreter (transformers 4.52, peft 0.17) to avoid the
    transformers 5.x deepspeed CUDA_HOME quirk in RRenv. CUDA_HOME is exported
    so the deepspeed probe inside accelerate.unwrap_model passes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    ada = Path(args.adapter)
    out = Path(args.out)

    if not base.is_dir():
        print(f"ERROR base not found: {base}", file=sys.stderr); return 2
    if not ada.is_dir() or not (ada / "adapter_model.safetensors").exists():
        print(f"ERROR adapter not found: {ada}", file=sys.stderr); return 2

    sentinel = out / ".merge.done"
    ada_sig = str(int((ada / "adapter_model.safetensors").stat().st_mtime))
    if sentinel.exists() and not args.force:
        try:
            saved = json.loads(sentinel.read_text())
            if saved.get("adapter_mtime") == ada_sig and (out / "config.json").exists():
                print(f"SKIP (already merged): {out}")
                return 0
        except Exception:
            pass

    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[merge] base={base}", flush=True)
    print(f"[merge] adapter={ada}", flush=True)
    print(f"[merge] out={out}", flush=True)

    m = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True
    )
    print(f"[merge] base loaded in {time.time()-t0:.1f}s", flush=True)

    m = PeftModel.from_pretrained(m, str(ada))
    print(f"[merge] adapter loaded in {time.time()-t0:.1f}s", flush=True)

    m = m.merge_and_unload()
    print(f"[merge] merge_and_unload done in {time.time()-t0:.1f}s", flush=True)

    m.save_pretrained(str(out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    tok.save_pretrained(str(out))
    print(f"[merge] saved in {time.time()-t0:.1f}s", flush=True)

    sentinel.write_text(json.dumps({
        "adapter": str(ada),
        "adapter_mtime": ada_sig,
        "base": str(base),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
