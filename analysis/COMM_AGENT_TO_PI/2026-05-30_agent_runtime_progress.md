# Agent Runtime Progress - 2026-05-30 00:08 UTC

## Repo / Cloud

- Repo: `/mnt/cpfs/junlongke/onlinelora/lora_obd`
- Branch: `main`
- Remote: `origin git@github.com:Rirayh/onlinelora.git`
- Previous pushed progress commit: `dff3ac9`

This note records runtime state only. Large generated outputs under `results/` and live logs are intentionally not committed.

## Health Summary

Status after inspection: running, with one scheduling gap remediated.

- PhaseD training is healthy on GPUs 0-3.
- Phase1 `relora_baseline/seed44` clean rerun completed successfully and saved `merged_final`.
- Phase1 `random_dr0.5` evals completed and produced result JSON files.
- GPUs 4-7 were idle at first inspection even though Phase1/Phase1.5 evals were still pending. This was due to the one-shot eval orchestrator not draining the queue after the previous batch.
- A new eval batch was started on GPUs 4-7, and a longer temporary autodrain loop is now active for the remaining Phase1.5 jobs.

## Current GPU Occupancy

As of 2026-05-30 00:08 UTC all 8 GPUs are occupied.

| GPU | PID(s) | Job |
| --- | --- | --- |
| 0 | `3005198` | PhaseD train `lora_vanilla/seed42` |
| 1 | `3005199` | PhaseD train `lora_vanilla/seed43` |
| 2 | `3005200` | PhaseD train `v1_S3pos/seed42` |
| 3 | `3005201` | PhaseD train `v1_S3pos/seed43` |
| 4 | `3109659`, `3111061` | Phase1 eval `relora_baseline/seed42`, vLLM EngineCore |
| 5 | `3109660`, `3111062` | Phase1 eval `relora_baseline/seed43`, vLLM EngineCore |
| 6 | `3109661`, `3111071` | Phase1 eval `relora_baseline/seed44`, vLLM EngineCore |
| 7 | `3109662`, `3111060` | Phase1.5 eval `random_anneal_up/seed42`, vLLM EngineCore |

## Phase1 Clean Rerun

The clean rerun for `relora_baseline/seed44` completed:

- Training completed at 2026-05-29 22:37 UTC.
- Final validation loss: `1.6833`.
- `summary.json` was written.
- `merged_final` was saved at:
  `results/phase1_robustness/qwen3-8b/tulu3-sft/relora_baseline/seed44/merged_final`

It is now being evaluated on GPU 6.

## Phase1 Eval Progress

Completed result JSONs now exist for:

- `p1/v1_S3pos/s42`
- `p1/v1_S3pos/s43`
- `p1/v1_S3pos/s44`
- `p1/random_dr0.5/s42`
- `p1/random_dr0.5/s43`
- `p1/random_dr0.5/s44`

Currently running:

- `p1/relora_baseline/s42` on GPU 4
- `p1/relora_baseline/s43` on GPU 5
- `p1/relora_baseline/s44` on GPU 6

After these finish, Phase1 will have the required v1/random/baseline eval coverage for the 3-seed decision.

## Phase1.5 Eval Progress

Currently running:

- `p1p5/random_anneal_up/s42` on GPU 7

Still pending:

- `p1p5/random_anneal_down/s42`
- `p1p5/random_triangle_up_down/s42`
- `p1p5/random_triangle_down_up/s42`

A longer temporary autodrain loop is active:

- PID: `3110194`
- Log: `logs/phase1D_eval/p1_p1p5_autodrain_20260530_0006.log`
- Behavior: every 10 minutes, rerun `phase1D_eval_orchestrator.py --phase1 --phase1p5` for up to 12 cycles, so newly free GPUs pick up remaining pending evals.

## PhaseD Training

All four PhaseD jobs are live and making progress:

- `lora_vanilla/seed42`: latest observed `step=5175/10000` at 2026-05-30 00:08 UTC.
- `lora_vanilla/seed43`: latest observed `step=5175/10000` at 2026-05-30 00:06 UTC.
- `v1_S3pos/seed42`: latest observed `step=5125/10000` at 2026-05-30 00:07 UTC.
- `v1_S3pos/seed43`: latest observed `step=5125/10000` at 2026-05-30 00:06 UTC.

Rough ETA at the current rate is around 2026-05-30 15:00-16:00 UTC for PhaseD training completion, before PhaseD eval.

## Watchpoints

- `scripts/phase1D_eval_orchestrator.py` is still effectively one-shot; the temporary autodrain loop is operational mitigation, not a permanent scheduler.
- Check `logs/phase1D_eval/*.eval.log` for nonzero eval exits.
- Check `logs/phaseD/*.train.log` for PhaseD progress and final `merged_final` creation.
- Do not commit `results/`, live `logs/`, `.phase1_seed44_rerun.pid`, or local incident timestamp files.

---

# Update - 2026-05-30 13:42 UTC

## Pulled PI Feedback #8

Pulled `origin/main` to `977436f` and received `analysis/COMM_PI_TO_AGENT/2026-05-30_pi_feedback_8_phase1p5_seed_stabilize.md`.

## New Phase 1.5 Jobs Launched

Launched the highest-priority stabilization batch from section A1:

| GPU | PID | Job |
| --- | ---: | --- |
| 4 | `3165503` | Phase1.5 train `random_anneal_down/seed43` |
| 5 | `3165504` | Phase1.5 train `random_anneal_down/seed44` |

Logs:

- `logs/phase1p5/random_anneal_down.seed43.train.log`
- `logs/phase1p5/random_anneal_down.seed44.train.log`

Both jobs reached step 0 diagnostics and confirmed `drop_schedule 'anneal_down' -> [0.75, 0.65, 0.55, 0.45]`.

## GPU State

As of 2026-05-30 13:41 UTC:

- GPUs 0-3: PhaseD training still healthy.
- GPUs 4-5: new Phase1.5 `random_anneal_down` seed43/44 training.
- GPUs 6-7: idle by design; PI #8 only required two new training jobs at this point.

## Phase 1 Decision Analysis

Regenerated Phase 1 paired-t outputs:

- `analysis/results_v3/phase1_summary.json`
- `analysis/COMM_AGENT_TO_PI/2026-05-30_phase1_decision.md`

Result: Phase 1 alone satisfies the selection-vs-random rule (`+3.18pp`, `p=0.0479`), but Phase 2 remains held until Phase 1.5 n=3 evals finish.

## Remaining Work

- Wait for `random_anneal_down/seed43` and `seed44` `merged_final/`.
- Eval both Phase1.5 seeds on the same suite.
- Recompute Phase1.5 n=3 decision per PI section A4.
- Low-priority hygiene still pending: delete older duplicate `random_anneal_up` result JSON and replace the temporary eval autodrain with `--drain`.

