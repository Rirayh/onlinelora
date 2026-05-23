# Phase D LIVE STATUS — 2026-05-22 17:54

## Pipeline architecture (current)
- **Daemon**: `scripts/phase_d_daemon.py` PID 1916993 alive (restarted 17:21)
- **Eval backends** (split by env):
  - `env=espo` (Qwen3 dense: qwen3-1p7b/4b/14b): **vLLM-on-merged**
    - Step 1: `merge_adapter.py` (PY_RRENV, CUDA_HOME=/usr/local/cuda-12, CUDA_VISIBLE_DEVICES="")
      → `<seed_dir>/merged/` (bf16, ~3GB for 1.7B, ~80s)
    - Step 2: `lm_eval --model vllm pretrained=<merged> bs=auto gpu_mem_util=0.85`
  - `env=rrenv` (Qwen3.5: qwen35-{0p8b,2b,4b,9b,27b}): **HF-with-PEFT**
    - vLLM 0.15.1 does NOT support `Qwen3_5ForCausalLM` (hybrid Mamba/full-attn) — confirmed via pydantic validation error
    - Even `model_impl=transformers` fallback rejects it: "The Transformers implementation of 'Qwen3_5ForCausalLM' is not compatible with vLLM"
    - Stays on HF backend, but bs bumped: 0p8b=16, 2b=8, 4b=4, 9b=2

## Validated facts
- `merge_adapter.py` works on RRenv with `CUDA_HOME=/usr/local/cuda-12` (deepspeed probe needs it; merge runs CPU)
  - qwen35-0p8b/dora merged in 65s
  - qwen3-1p7b/dora merged in 105s (saved at `results/stage3_v2/qwen3-1p7b/tulu3-sft/dora/seed42/merged/`)
- vLLM smoke on `qwen3-1p7b/dora/merged` IN PROGRESS (PID 1919866 GPU=5, gpu_mem_util=0.4 to share with HF eval)
  - Engine init done at 17:50, doing actual eval now
  - HF baseline (qwen3-1p7b/dora full): gsm8k_strict=0.5201, flex=0.5262, hella=0.6314, arc=0.5205
  - Expected vLLM result ≈ HF baseline (validates merge correctness)
- numpy in RRenv: 2.4 → 2.2.6 (numba in vllm v1 spec_decode requires <2.3)
- datasets in RRenv: 3.6.0 → 4.8.5 (need 'List' feature for lm_eval gsm8k metadata)

## qwen3.5 vLLM blockers (DO NOT keep retrying)
- `Qwen3_5ForCausalLM` not in vLLM 0.15.1 supported architectures
- Architecture rename hack would fail: it's a hybrid linear-attn (Mamba-style in_proj_qkv/z/b/a) so structurally != Qwen3
- Decision: keep HF backend for all qwen35-*

## Done evals
- qwen3-1p7b: 5/5 (HF, espo) — ALREADY in lm_eval_v3/results_*.json
- qwen35-0p8b: 5/5 (HF, RRenv) — committed b8da94c
  | method | gsm8k_strict | gsm8k_flex | hella | arc |
  |---|---|---|---|---|
  | lora_vanilla | 0.2320 | 0.1713 | 0.4296 | 0.4232 |
  | relora_baseline | 0.2881 | 0.2889 | 0.4092 | 0.4061 |
  | S3pos | 0.3116 | 0.3124 | 0.4087 | 0.4019 |
  | dora | 0.3108 | 0.3124 | 0.4187 | 0.4172 |
  | cola | 0.2972 | 0.2972 | 0.4084 | 0.4070 |

## In flight (training)
- qwen35-2b: 5/5 trained — 4 lm-eval (HF) running on GPU 2/3/5/6 launched ~15:13-15:46 (will finish ~16:30-17:30)
  - actually one (lora_vanilla) already done v3=1
- qwen35-4b: training 4/5 (lora_vanilla 2000/3000, relora_baseline 1975/3000, S3pos 300/3000, dora 150/800)
  - rate: ~25 s/step (lora/relora) and 45 s/step (dora) — linear-attn slow fallback (no fla installed)
  - cola not yet launched

## Pending (training)
- qwen35-4b/cola
- qwen3-4b: 5 cells (env=espo, qwen3 dense — should be ~2-3x faster than qwen35-4b)
- qwen35-9b: 5 cells (rrenv, will be ~30-50s/step, 24-40h each)
- qwen3-14b: 5 cells (espo, ~10-15s/step likely)

## OPLoRA analysis task (assigned, NOT STARTED)
**Goal**: contrast experiment X-1 + X-2 (offline SVD vs online curvature subspace alignment)
- **No retraining**, pure offline SVD postprocessing on existing adapters
- Models: qwen3-1.7b, qwen3-8b, qwen35-0p8b, qwen35-2b
- Methods: lora_vanilla, dora, relora_diag_gated_S3pos
- Datasets: tulu3-sft (mathmix optional)
- For each (model, data, method, layer) emit JSON:
  ```
  {
    layer: "model.layers.0.self_attn.q_proj",
    rho_k: {8,16,32,64,128} -> float,           # OPLoRA metric
    subspace_overlap_left:  k -> float in [0,1],
    subspace_overlap_right: k -> float in [0,1],
    per_window_drift:       w=1..4 -> float in [0,1]   # X-2 only (windowed checkpoints)
  }
  ```
- Output figures:
  - Fig_A: ρ_k vs k, line per method
  - Fig_B: subspace overlap (avg over layers) vs k
  - Fig_C: per-window drift heatmap (windows × layers)
- Plan:
  1. `scripts/oplora_analysis.py`: load base W0 + adapter ΔW (or W0 + checkpoints/best/ for windows from periodic ckpts step_*); for each target_module compute SVD top-k subspace overlap with W0's top-k subspace
  2. ρ_k = ||P_⊥(W0_top_k) ΔW||_F / ||ΔW||_F  (OPLoRA metric: how much of ΔW falls outside W0's top-k subspace; higher = more "novel" direction)
  3. Subspace overlap_left[k] = ||U_W0_topk^T U_ΔW_topk||_F^2 / k   (Hotelling/principal angles)
  4. For X-2: walk `checkpoints/step_000250.../adapter_*.safetensors` to compute ΔW per window, then drift = 1 - overlap(W_w, W_{w+1})
  5. Plot with matplotlib; save under `analysis/oplora/{figures,jsons}/`

## Critical files (don't lose)
- `scripts/phase_d_daemon.py` (eval pipeline split)
- `scripts/merge_adapter.py` (PEFT merge utility)
- `scripts/run_lmeval_8parallel.sh` (older 8-task fanout, NOT used by daemon)
- Latest commit: b8da94c "Phase D: vLLM-on-merged eval pipeline + qwen35-0p8b 5/5 results"
- Branch: main, remote: github.com:Rirayh/onlinelora.git
- Daemon log: `logs/scout/_phase_d_daemon.log`
- Daemon stdout: `logs/phase_d_daemon.out`
- STOP file: `/tmp/phase_d_daemon.STOP`

## Next concrete actions
1. Wait for vLLM smoke (PID 1919866) to finish, compare result vs HF=0.5201
2. If vLLM agrees with HF (within ±0.01) → pipeline validated
3. Start `scripts/oplora_analysis.py`, run on qwen35-0p8b first (smallest, cheapest); do CPU-only with adapter_model.safetensors
4. Don't disturb daemon; OPLoRA analysis is CPU-bound

## Environment
- /mnt/cpfs/junlongke/miniconda3/envs/espo: transformers 4.52, peft 0.17 (Qwen3 dense only, NO vllm)
- /mnt/cpfs/junlongke/miniconda3/envs/RRenv: transformers 5.3, peft 0.19.1, vllm 0.15.1, datasets 4.8.5, numpy 2.2.6, lm_eval 0.4.12
- CUDA: /usr/local/cuda-12 (no nvcc binary); only need CUDA_HOME for deepspeed probe
