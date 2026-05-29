# Agent Runtime Progress - 2026-05-29 16:11 UTC

## Repo / Cloud

- Repo: `/mnt/cpfs/junlongke/onlinelora/lora_obd`
- Branch: `main`
- Remote: `origin git@github.com:Rirayh/onlinelora.git`
- Previous pushed takeover commit: `de2bba3`

This note records runtime progress only. Large generated outputs under `results/` and live logs are intentionally not committed.

## Current GPU Occupancy

As of 2026-05-29 16:11 UTC all 8 GPUs are occupied.

| GPU | PID(s) | Job |
| --- | --- | --- |
| 0 | `3005198` | PhaseD train `lora_vanilla/seed42` |
| 1 | `3005199` | PhaseD train `lora_vanilla/seed43` |
| 2 | `3005200` | PhaseD train `v1_S3pos/seed42` |
| 3 | `3005201` | PhaseD train `v1_S3pos/seed43` |
| 4 | `3069656`, `3071323` | Phase1 eval `random_dr0.5/seed42`, vLLM EngineCore |
| 5 | `3069657`, `3071328` | Phase1 eval `random_dr0.5/seed43`, vLLM EngineCore |
| 6 | `3048918` | Phase1 clean rerun train `relora_baseline/seed44` |
| 7 | `3069658`, `3071335` | Phase1 eval `random_dr0.5/seed44`, vLLM EngineCore |

## Phase1 Clean Rerun

The contaminated duplicate `relora_baseline/seed44` run was moved aside earlier. The clean rerun is active:

- PID: `3048918`
- Output root: `results/phase1_robustness/qwen3-8b/tulu3-sft/relora_baseline/seed44`
- Latest observed progress: `step=950/3000` at 2026-05-29 16:08 UTC
- First merge completed at step 750; validation and post-merge validation both logged normally.

Estimated completion remains late 2026-05-29 UTC, followed by lm-eval after `merged_final` is available.

## Phase1 / Phase1.5 Eval Progress

The eval orchestrator patch from `de2bba3` fixed:

- `NLTK_DATA` routing to `/mnt/cpfs/junlongke/nltk_data`
- `Popen`-based child monitoring
- false running-state reports from zombie/exited children

Completed after the fix:

- `p1/v1_S3pos/s42`
- `p1/v1_S3pos/s43`
- `p1/v1_S3pos/s44`

Currently running:

- `p1/random_dr0.5/s42` on GPU 4
- `p1/random_dr0.5/s43` on GPU 5
- `p1/random_dr0.5/s44` on GPU 7

Remaining after the current eval batch:

- `p1/relora_baseline/s42`
- `p1/relora_baseline/s43`
- `p1/relora_baseline/s44` after clean rerun finishes
- `p1p5/random_anneal_up/s42`
- `p1p5/random_anneal_down/s42`
- `p1p5/random_triangle_up_down/s42`
- `p1p5/random_triangle_down_up/s42`

## Eval Scheduling Note

`scripts/phase1D_eval_orchestrator.py` is still a one-shot launcher: it launches at most one batch for currently free GPUs and exits after those children finish. It does not own a persistent queue.

To avoid idle GPU gaps, a temporary autodrain loop was started:

- PID: `3072459`
- Log: `logs/phase1D_eval/p1_p1p5_autodrain_20260529_1606.log`
- Behavior: reruns the orchestrator periodically so newly free GPUs can pick up remaining pending evals.

This is operationally useful but should be replaced by a real queue/drain mode in the orchestrator if long eval campaigns continue.

## PhaseD

Active PhaseD training jobs:

- `lora_vanilla/seed42`
- `lora_vanilla/seed43`
- `v1_S3pos/seed42`
- `v1_S3pos/seed43`

All four are still long-running 10k-step jobs and appeared healthy in `nvidia-smi`.

## Watchpoints

- Do not commit `results/`, live `logs/`, `.phase1_seed44_rerun.pid`, or local incident timestamp files.
- Check `logs/phase1D_eval/*.eval.log` if any eval exits nonzero.
- Check `logs/phase1/relora_baseline.seed44.train.log` for the clean rerun.
- Before final Phase1 decision, ensure `relora_baseline/seed44` clean `merged_final` exists and its lm-eval completed.
