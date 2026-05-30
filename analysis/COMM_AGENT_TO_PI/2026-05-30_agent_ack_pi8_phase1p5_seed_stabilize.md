# Agent ACK - PI Feedback #8 Phase 1.5 Seed Stabilization

**Date**: 2026-05-30 13:42 UTC
**ACK token**: `ACK_pi_feedback_8_phase1p5_seed_stabilize`
**Feedback file**: `analysis/COMM_PI_TO_AGENT/2026-05-30_pi_feedback_8_phase1p5_seed_stabilize.md`

## Actions Taken

- Pulled `origin/main` and received PI feedback #8 at commit `977436f`.
- Updated `scripts/phase1p5_train_orchestrator.py` to support targeted `--cells` and `--seeds` launches while preserving the seed42/all-cells default path.
- Launched Phase 1.5 `random_anneal_down` seed43 and seed44 with the same 3000-step config as seed42 and `--save_merged_final`.
- Regenerated Phase 1 paired-t analysis and fixed the report writer so the headline comparison rows preserve comparator names and metric labels.

## New Training Jobs

| Cell | Seed | PID | GPU | Log | Output root |
| --- | ---: | ---: | ---: | --- | --- |
| `random_anneal_down` | 43 | `3165503` | 4 | `logs/phase1p5/random_anneal_down.seed43.train.log` | `results/phase1p5_schedule_ablation/qwen3-8b/tulu3-sft/random_anneal_down/seed43/` |
| `random_anneal_down` | 44 | `3165504` | 5 | `logs/phase1p5/random_anneal_down.seed44.train.log` | `results/phase1p5_schedule_ablation/qwen3-8b/tulu3-sft/random_anneal_down/seed44/` |

Both logs reached model/data initialization, LoRA setup, step 0 diagnostics, and the expected anneal-down schedule:

`drop_schedule 'anneal_down' -> per-event rates: [0.75, 0.65, 0.55, 0.45]`

## Current Scheduling Decision

- `random_triangle_down_up/seed42` finished below `random_anneal_down/seed42` on gsm8k strict (`72.55 < 77.94`), so no triangle-down-up seed43/44 training is required under section A2.
- Phase 2 remains held pending Phase 1.5 n=3 results and evals under section A4.
- GPUs 6 and 7 are intentionally idle after this launch; only two new trainings were required.

## Phase 1 Analysis Output

Generated and ready to commit:

- `analysis/results_v3/phase1_summary.json`
- `analysis/COMM_AGENT_TO_PI/2026-05-30_phase1_decision.md`

Phase 1 decision remains `PROCEED_TO_PHASE2` by the Phase 1 rule, but execution is held until the Phase 1.5 n=3 comparison lands.
