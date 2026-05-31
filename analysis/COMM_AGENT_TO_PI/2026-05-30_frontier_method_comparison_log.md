# Frontier Method Comparison Log - 2026-05-30

## Mission

User-level goal: decide whether our method has a defensible advantage over the strongest relevant methods available by the end of May 2026.

Working hypothesis to test:

- Our method is not just better than vanilla LoRA/ReLoRA; it must survive comparison against modern LoRA-family methods that use stronger initialization, rank allocation, dynamic rank growth/pruning, or full-rank random bases.
- If the advantage only holds against weak baselines, the claim must be softened or reframed.
- Every code iteration, failed run, bug, and result must be logged in enough detail to reproduce or audit later.

## Current Experiment State at Takeover of This Goal

Latest pushed commit before this log: `7bbcce4 Document PhaseD training completion`.

Running as of 2026-05-30 19:18 UTC:

- GPU0-3: PhaseD lm-eval via vLLM EngineCore, launched from `scripts/phase1D_eval_orchestrator.py --phaseD --gpus 0,1,2,3`.
- GPU4-5: Phase1.5 `random_anneal_down` seed43/seed44 training, still in progress.
- GPU6-7: idle.

Completed before this log:

- Phase1 n=3 showed v1_S3pos beats matched random on GSM8K by +3.18pp, paired t-test p=0.0479.
- Phase1.5 seed42 showed `random_anneal_down` is the strongest schedule threat among completed schedule cells.
- PhaseD training completed all four 10000-step jobs. Final validation loss strongly favors v1_S3pos over vanilla LoRA, but benchmark eval is still pending.

## Frontier Candidate Set, Cutoff 2026-05-30

Priority A - must compare if implementation cost is reasonable:

| Method family | Why it matters | First comparison target |
| --- | --- | --- |
| LoRA / QLoRA | canonical baseline; already partly covered by vanilla LoRA and ReLoRA variants | existing results plus clean table |
| ReLoRA | direct ancestor of our method; high-rank training via low-rank updates | existing baseline + PhaseD |
| AdaLoRA / ALoRA / ElaLoRA | closest conceptual competitor: adaptive budget/rank allocation by importance | implement or reuse library if available |
| PiSSA / SORSA | stronger SVD-based initialization and principal-component adaptation | implement PiSSA first; SORSA optional if code available |
| EVA | data-driven activation-SVD initialization and rank redistribution | high priority if PEFT version supports it or implementation is localizable |
| DoRA / BoRA | weight decomposition variants; common strong LoRA successors | DoRA first; BoRA lower priority unless easy |
| RandLoRA | challenges the low-rank bottleneck by full-rank random-basis updates | compare as a stress baseline if implementation available |
| TLoRA+ | April 2026 low-rank PEFT method; recent, but reported mainly on GLUE | scan code availability before treating as required |

Priority B - include only if claim expands beyond single-task SFT:

| Method family | Reason for lower priority |
| --- | --- |
| MoLE / MeteoRA / Meta-UCF | multi-adapter, mixture, or continual-learning settings; important if we claim continual/multi-task benefits, but not the first fair comparator for one-task TULU SFT |
| LoRA-FA / LoLoRA / PLoRA | primarily memory/throughput/hyperparameter orchestration; compare if we make efficiency claims rather than quality/stability claims |
| CAR-LoRA / Fit-LoRA | compression / model-portability oriented; not the same target setting |

Initial source anchors:

- LoRA: https://arxiv.org/abs/2106.09685
- ReLoRA: https://arxiv.org/abs/2307.05695
- AdaLoRA: https://arxiv.org/abs/2303.10512
- DoRA: https://arxiv.org/abs/2402.09353
- PiSSA: https://arxiv.org/abs/2404.02948
- CorDA: https://arxiv.org/abs/2406.05223
- EVA: https://arxiv.org/abs/2410.07170
- ElaLoRA: https://arxiv.org/abs/2504.00254
- RandLoRA: https://arxiv.org/abs/2502.00987
- TLoRA+: https://arxiv.org/abs/2604.13368
- Meta-UCF: https://openreview.net/forum?id=iNg5KL7eTC

## Fair Comparison Protocol

Primary model/data/eval setting:

- Model: `qwen3-8b`.
- Data: `tulu3-sft`.
- Seeds: 42, 43, 44 for final claims.
- Main task suite: GSM8K strict/flex, HellaSwag, ARC-Challenge, MMLU, IFEval.
- Training budgets:
  - Short SFT: 3000 steps, matching Phase1/Phase1.5.
  - Overtrain/stability: 10000 steps, matching PhaseD.
- Report all trainable parameter counts, wall-clock time, peak VRAM, final/best val loss, and benchmark scores.

Decision rule tiers:

1. Claim can be strong only if our method beats the best reasonable frontier baseline at n=3 on GSM8K and does not regress materially on broad metrics.
2. Claim can be moderate if our method is not best on raw score but has better overtraining stability, lower variance, or better score/compute tradeoff.
3. Claim must be negative or reframed if a schedule/rank/init baseline dominates at matched budget.

## Implementation Plan

Immediate:

1. Let current PhaseD eval finish; parse and summarize results.
2. Let Phase1.5 `random_anneal_down` seed43/44 finish; eval and recompute n=3 schedule threat.
3. Probe installed library support for DoRA, PiSSA, EVA, AdaLoRA without changing training code yet.
4. Add a frontier-baseline runner only after the current result queue is clean.

First baselines to implement/run:

1. DoRA at 3000 steps, n=3 if supported directly by PEFT.
2. PiSSA initialization for LoRA/ReLoRA-compatible layers, n=1 smoke then n=3.
3. AdaLoRA or ElaLoRA-like dynamic rank allocation, depending on code availability and implementation risk.
4. EVA if activation-SVD init can be implemented without destabilizing current stage3 runner.

## Bug and Run Logging Rules

Every meaningful run must have:

- Command line.
- Git commit hash.
- Conda env / Python executable.
- GPU ids and PIDs.
- Output root and log path.
- Expected stop condition.
- First 50 log lines check result.
- Final status: success, failed, killed, superseded, or invalid.
- If invalid: precise reason and whether outputs were deleted or quarantined.

Bug entry template:

```text
### BUG-YYYYMMDD-NN - short name
- Symptom:
- Repro command:
- Affected commit:
- Root cause:
- Fix commit:
- Validation:
- Residual risk:
```

No silent fixes. Any failed experiment that affects conclusions must be documented before the next claim is made.

## Open Risks

- PEFT/library support may lag the papers. If a method needs nontrivial custom implementation, run a tiny smoke first and do not mix it into final comparisons until validated.
- Some 2026 methods target continual learning, compression, or tuning throughput rather than single-task quality. These should not be treated as direct quality baselines unless our claim enters their setting.
- Current untracked `results/` tree is large and dirty by design. Do not commit it. Commit only code, reports, and small JSON summaries.
- `rg` is unavailable in this environment; use `grep`/`find` for local search unless `rg` becomes available.

## Next Concrete Checkpoint

Checkpoint A is complete when:

- PhaseD eval results are summarized and pushed.
- Phase1.5 anneal_down seed43/44 eval results are summarized and pushed.
- A small compatibility report says which frontier baselines are directly runnable in this repo/env.

Checkpoint B starts after that: implement and launch the first frontier baseline batch.

## Environment Compatibility Probe - 2026-05-30 19:20 UTC

Command summary:

```bash
/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python -c "import inspect, torch, transformers, peft; from peft import LoraConfig; print(torch.__version__, transformers.__version__, peft.__version__); print(inspect.signature(LoraConfig))"
```

Observed versions:

- `torch`: 2.6.0+cu124
- `transformers`: 4.52.0.dev0
- `peft`: 0.17.0

PEFT `LoraConfig` directly exposes:

- `use_dora`: direct DoRA path available.
- `init_lora_weights='pissa'`, `pissa_niter_[N]`: PiSSA initialization available.
- `init_lora_weights='eva'` plus `eva_config`: EVA path appears available.
- `init_lora_weights='corda'` plus `corda_config`: CorDA path appears available.
- `use_qalora`: QA/QALoRA-related support exists but not first priority.
- `rank_pattern` / `alpha_pattern`: can support static layerwise allocation baselines.

Current `scripts/stage3_run.py` already has method branches for:

- `dora` through `LoraConfig(..., use_dora=True)`.
- `adalora` through `AdaLoraConfig` if import succeeds.

Gap identified:

- `stage3_run.py` does not yet expose generic PEFT LoRA initialization choices (`pissa`, `eva`, `corda`, `olora`, `orthogonal`) as CLI methods.
- First code PR should add explicit methods such as `pissa`, `pissa_niter_16`, `eva`, and maybe `corda` only after a smoke run confirms PEFT's extra init data requirements.
- DoRA and AdaLoRA are the lowest-risk first frontier baselines because the current runner already has method choices for them.

Immediate baseline queue after current running jobs clear:

1. DoRA smoke: qwen3-8b/tulu3-sft, seed42, 250-500 steps, verify merge/save/eval path.
2. AdaLoRA smoke: same small budget; confirm rank scheduler does not conflict with ReLoRA merge logic. It should be treated as a non-ReLoRA baseline first.
3. Add PiSSA CLI support; run seed42 3000-step baseline if smoke passes.
4. Add EVA only after checking whether PEFT requires a dataloader-specific initialization pass.

## Run Launch - Frontier Baseline Pilot 2026-05-30 19:45 UTC

Purpose: use idle GPU6/7 while PhaseD eval and Phase1.5 stabilization are running. Start the two lowest-risk frontier baselines that current `stage3_run.py` already supports without code changes.

Git commit at launch: `e15a151`.

Shared config:

- Model: `/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B`
- Model key: `qwen3-8b`
- Dataset: `tulu3-sft`
- Seed: `42`
- Steps: `3000`
- Eval every: `250`
- Merge every: `750` (irrelevant for non-ReLoRA methods; both have `merge events scheduled at: []`)
- Checkpoints: disabled via `--ckpt_every 0`
- Attention: `sdpa`
- Save: `--save_merged_final`

Jobs launched:

| Method | GPU | PID | Log | Output root | Initial status |
| --- | ---: | ---: | --- | --- | --- |
| `dora` | 6 | `3200201` | `logs/frontier/dora.seed42.train.log` | `results/frontier_baselines/qwen3-8b/tulu3-sft/dora/seed42/` | initialized; 252 LoRA layers, 4032 components, 45.05M trainable params |
| `adalora` | 7 | `3200202` | `logs/frontier/adalora.seed42.train.log` | `results/frontier_baselines/qwen3-8b/tulu3-sft/adalora/seed42/` | initialized; 87.30M trainable params; rank-stat helper sees 0 standard LoRA handles |

Commands used:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python scripts/stage3_run.py \
  --model_path /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B \
  --model_key qwen3-8b --dataset tulu3-sft --method dora \
  --total_steps 3000 --merge_every 750 --eval_every 250 --ckpt_every 0 \
  --saliency_max_seq_len 512 --attn_implementation sdpa --save_merged_final \
  --seed 42 --out_root results/frontier_baselines/qwen3-8b/tulu3-sft/dora/seed42

CUDA_VISIBLE_DEVICES=7 /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python scripts/stage3_run.py \
  --model_path /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B \
  --model_key qwen3-8b --dataset tulu3-sft --method adalora \
  --total_steps 3000 --merge_every 750 --eval_every 250 --ckpt_every 0 \
  --saliency_max_seq_len 512 --attn_implementation sdpa --save_merged_final \
  --seed 42 --out_root results/frontier_baselines/qwen3-8b/tulu3-sft/adalora/seed42
```

Observed first-log check:

- DoRA reached data/model init, LoRA wrapping, step-0 rank stats, and no merge events.
- AdaLoRA reached data/model init and no merge events, but standard `get_lora_BA_handles()` returned 0 handles. This makes effective-rank/condition-number logging `nan`; training can still proceed because trainable params are present.

### BUG-20260530-01 - AdaLoRA rank-stat helper sees no standard LoRA handles

- Symptom: `stage3_run.py --method adalora` logs `#LoRA layers=0 #components=0`, then `mean_ER=nan mean_CN=nan` at step 0.
- Repro command: see AdaLoRA command above.
- Affected commit: `e15a151`.
- Root cause: current `src.model.get_lora_BA_handles()` appears tailored to standard LoRA modules and does not recognize PEFT AdaLoRA module structure.
- Impact: rank/effective-rank diagnostics are invalid for AdaLoRA, but the run still has 87.30M trainable parameters and can produce val/eval metrics.
- Fix plan: before using AdaLoRA diagnostic claims, extend the handle collector or explicitly mark AdaLoRA rank metrics as N/A in summaries.
- Current action: keep the run alive as a quality baseline pilot; do not use its rank-stat diagnostics for conclusions.


## Run Health Check - 2026-05-30 19:55 UTC

GPU6/GPU7 frontier pilots are occupying the remaining compute capacity; no extra baseline is launched at this checkpoint.

Observed state:

- DoRA seed42 PID `3200201`: process alive on GPU6, ~31.3 GiB memory, 100% util. Log confirms initialization, 252 LoRA layers, 4032 components, 45.05M trainable params, and no merge events. No step-25 line yet; keep monitoring before declaring stuck.
- AdaLoRA seed42 PID `3200202`: process alive on GPU7, ~32.0 GiB memory, 100% util. Reached step `25/3000` at 2026-05-30 19:51 UTC with train loss `1.9309`. `BUG-20260530-01` still applies to rank diagnostics only.

Operational rule added from this check: free memory is not sufficient reason to colocate another training job. Current A100 util is already saturated, so the next frontier experiment should wait for an actual GPU to free.

## Run Launch - 2026-05-31 08:10 UTC

Overnight frontier status:

- AdaLoRA seed42 completed training in `35454.0s`, final val loss `1.3689`, best logged val loss `1.3061` at step 1000. `merged_final/` was saved. Rank diagnostics remain invalid because `BUG-20260530-01` still applies.
- DoRA seed42 is still running and is much slower: step `600/3000` at 2026-05-31 07:46 UTC, best logged val loss `1.3106` at step 500.

New jobs:

| GPU | PID | Method | Seed | Purpose | Status |
| --- | ---: | --- | ---: | --- | --- |
| 2 | `3249642` | AdaLoRA | 42 | final benchmark eval | running |
| 3 | `3250178` | AdaLoRA | 43 | n=3 expansion | running; initialized, 87.30M trainable params |
| 4 | `3250040` | AdaLoRA | 44 | n=3 expansion | running |
| 5 | `3252651` | PiSSA-niter-16 | 42 | smoke for PEFT PiSSA support | running; pending validation |

Code change staged in working tree: `scripts/stage3_run.py` now exposes `pissa` and `pissa_niter_16` as PEFT LoRA initialization baselines with no ReLoRA merge events. Syntax check passed with `python -m py_compile scripts/stage3_run.py`.

Operational notes:

- DoRA seed expansion is deferred because seed42 ETA is too high.
- AdaLoRA seed expansion is preferred because seed42 completed overnight and produces a viable quality baseline, although rank diagnostics must be marked N/A.
- PiSSA is treated as an initialization baseline. It should only be promoted from smoke to 3000-step run after the smoke confirms model wrapping, rank stats, and at least one eval point.

## Run Launch - PiSSA Full Seed42 2026-05-31 08:12 UTC

PiSSA-niter-16 smoke passed the initialization and training-loop checks by 2026-05-31 08:11 UTC:

- 252 LoRA layers, 4032 components.
- 43.65M trainable params.
- Step-0 rank stats valid: `mean_ER=1926.51`, `mean_CN=8.22e+04`.
- Reached step `15/50` without error.

Based on that, launched full `pissa_niter_16` seed42 on GPU7:

- PID: `3258837`
- Log: `logs/frontier/pissa_niter_16.seed42.train.log`
- Output: `results/frontier_baselines/qwen3-8b/tulu3-sft/pissa_niter_16/seed42/`
- Config: qwen3-8b/tulu3-sft, 3000 steps, eval every 250, no ReLoRA merge events, `--save_merged_final`.

This fills the last idle GPU. Current frontier training coverage now includes AdaLoRA n=3 in progress, DoRA seed42 in progress, and PiSSA-niter-16 seed42 in progress.
