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

