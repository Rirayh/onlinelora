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

---

# Update - 2026-05-30 18:55 UTC

## PhaseD Training Completed

All four PhaseD 10000-step training jobs completed and saved `merged_final/`:

| Cell | Seed | Final val loss | Best val loss | Completed UTC | merged_final |
| --- | ---: | ---: | ---: | --- | --- |
| `lora_vanilla` | 42 | 3.7051 | 1.3116 | 2026-05-30 15:14 | `results/phase_d/qwen3-8b/tulu3-sft/lora_vanilla/seed42/merged_final/` |
| `lora_vanilla` | 43 | 3.7439 | 1.3136 | 2026-05-30 15:11 | `results/phase_d/qwen3-8b/tulu3-sft/lora_vanilla/seed43/merged_final/` |
| `v1_S3pos` | 42 | 1.9068 | 1.3117 | 2026-05-30 15:30 | `results/phase_d/qwen3-8b/tulu3-sft/v1_S3pos/seed42/merged_final/` |
| `v1_S3pos` | 43 | 1.9263 | 1.3135 | 2026-05-30 15:31 | `results/phase_d/qwen3-8b/tulu3-sft/v1_S3pos/seed43/merged_final/` |

Immediate read: after 10000 steps, vanilla LoRA is heavily overfit by validation loss, while v1_S3pos remains much lower. Final benchmark eval is still required before making a PhaseD claim.

## PhaseD Eval Launched

Started PhaseD lm-eval on the four completed merged models:

- Orchestrator PID: `3189161`
- Log: `logs/phase1D_eval/phaseD_eval_20260530_1853.log`
- Jobs launched:
  - `pD/lora_vanilla/s42` PID `3189162` GPU0
  - `pD/lora_vanilla/s43` PID `3189163` GPU1
  - `pD/v1_S3pos/s42` PID `3189164` GPU2
  - `pD/v1_S3pos/s43` PID `3189165` GPU3

No PhaseD `lm_eval/results_*.json` existed at launch time.

## Phase1.5 Stabilization Still Running

PI #8 Phase1.5 `random_anneal_down` seed43/44 jobs are still healthy on GPU4/5. Latest observed progress at 2026-05-30 18:49 UTC:

- seed43: step `1625/3000`, completed merge event 2 at step 1500, realised drop rate `0.6451`.
- seed44: step `1625/3000`, completed merge event 2 at step 1500, realised drop rate `0.6486`.

Next expected result: Phase1.5 seed43/44 `merged_final/`, then run their lm-eval and recompute Phase1.5 n=3 verdict.

---

# Update - 2026-05-30 19:50 UTC

## Idle GPUs Filled with Frontier Baseline Pilots

Started two frontier baseline pilot trainings on the previously idle GPUs 6 and 7:

| GPU | PID | Job | Log |
| --- | ---: | --- | --- |
| 6 | `3200201` | `frontier/dora/seed42`, 3000-step qwen3-8b/tulu3-sft | `logs/frontier/dora.seed42.train.log` |
| 7 | `3200202` | `frontier/adalora/seed42`, 3000-step qwen3-8b/tulu3-sft | `logs/frontier/adalora.seed42.train.log` |

Both jobs use `--save_merged_final` and output under `results/frontier_baselines/qwen3-8b/tulu3-sft/<method>/seed42/`.

Current GPU allocation after launch:

- GPU0-3: PhaseD benchmark eval via vLLM.
- GPU4-5: Phase1.5 `random_anneal_down` seed43/44 training.
- GPU6: DoRA frontier baseline seed42 training.
- GPU7: AdaLoRA frontier baseline seed42 training.

First-log check:

- DoRA initialized normally: 252 LoRA layers, 4032 components, 45.05M trainable params.
- AdaLoRA initialized and has 87.30M trainable params, but current rank-stat helper reports 0 standard LoRA handles, causing `mean_ER=nan` / `mean_CN=nan`. This is logged as `BUG-20260530-01` in the frontier comparison log. The run remains useful as a quality baseline, but its rank diagnostics are invalid until fixed or marked N/A.


---

# Update - 2026-05-30 19:55 UTC

## GPU Utilization Check

No idle GPU remains. `nvidia-smi` at 2026-05-30 19:52 UTC showed all eight A100s at high utilization:

- GPU0-3: PhaseD `lm_eval` via vLLM, each using ~76.4 GiB and 96-100% GPU util.
- GPU4: Phase1.5 `random_anneal_down` seed43, PID `3165503`, ~31.9 GiB and 100% util.
- GPU5: Phase1.5 `random_anneal_down` seed44, PID `3165504`, ~32.6 GiB and 100% util.
- GPU6: DoRA frontier baseline seed42, PID `3200201`, ~31.3 GiB and 100% util.
- GPU7: AdaLoRA frontier baseline seed42, PID `3200202`, ~32.0 GiB and 100% util.

Decision: do not launch additional jobs just because GPU4-7 still have free memory. Compute utilization is already saturated; colocating another training job would likely slow the active jobs and increase OOM risk.

## Runtime Health

Latest observed progress:

- PhaseD eval orchestrator still reports all four PhaseD eval jobs running through 2026-05-30 19:52 UTC. No `results_*.json` files are complete yet under `results/phase_d/`.
- Phase1.5 `random_anneal_down` seed43 reached step `1950/3000` at 2026-05-30 19:50 UTC.
- Phase1.5 `random_anneal_down` seed44 reached step `1975/3000` at 2026-05-30 19:53 UTC.
- AdaLoRA frontier baseline reached step `25/3000` at 2026-05-30 19:51 UTC, train loss `1.9309`.
- DoRA frontier baseline has completed initialization and is consuming GPU, but has not emitted a step-25 line yet. Keep monitoring; current state is consistent with a slow first compiled training segment rather than a confirmed failure.

Next action when a GPU frees: if PhaseD eval completes first, use freed GPUs to evaluate Phase1.5 seed43/44 once their `merged_final/` directories exist. If a training GPU frees first, extend frontier baselines with PiSSA support and a seed42 run after a smoke check.

---

# Update - 2026-05-31 08:10 UTC

## Overnight Results and GPU Refill

At 2026-05-31 07:59 UTC, GPUs 0-5 and 7 were free; only DoRA seed42 was still running on GPU6.

Completed since the previous checkpoint:

- PhaseD benchmark eval finished at 2026-05-30 20:23 UTC.
- Phase1.5 `random_anneal_down` seed43/44 training finished and wrote `summary.json` plus `merged_final/`.
- AdaLoRA seed42 finished training, wrote `summary.json`, and saved `merged_final/`.

### PhaseD Eval Summary

| Cell | Seed | gsm8k strict | gsm8k flex | HellaSwag | ARC-C | MMLU | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lora_vanilla` | 42 | 87.57 | 88.25 | 76.07 | 66.47 | 74.78 | 25.51 |
| `lora_vanilla` | 43 | 87.57 | 88.25 | 76.07 | 66.47 | 74.78 | 25.51 |
| `v1_S3pos` | 42 | 77.03 | 77.18 | 79.00 | 66.72 | 73.17 | 42.33 |
| `v1_S3pos` | 43 | 76.65 | 77.03 | 79.41 | 65.10 | 73.24 | 44.36 |

Read: PhaseD does not give a single-axis win. Vanilla LoRA wins GSM8K after long overtraining, while `v1_S3pos` has much better IFEval and slightly better HellaSwag. This should be framed as a tradeoff / overtraining-regime diagnostic, not as a clean dominance result.

## New Jobs Launched

### Eval Jobs

Started with `setsid env ...` after plain background `nohup` failed to keep lm-eval alive in this execution wrapper.

| GPU | PID | Job | Log |
| --- | ---: | --- | --- |
| 0 | `3248737` | Phase1.5 `random_anneal_down` seed43 lm-eval | `logs/phase1p5_eval/random_anneal_down.seed43.eval.log` |
| 1 | `3249570` | Phase1.5 `random_anneal_down` seed44 lm-eval | `logs/phase1p5_eval/random_anneal_down.seed44.eval.log` |
| 2 | `3249642` | Frontier AdaLoRA seed42 lm-eval | `logs/frontier_eval/adalora.seed42.eval.log` |

### Frontier Training Jobs

| GPU | PID | Job | Log |
| --- | ---: | --- | --- |
| 3 | `3250178` | AdaLoRA seed43, 3000 steps | `logs/frontier/adalora.seed43.train.log` |
| 4 | `3250040` | AdaLoRA seed44, 3000 steps | `logs/frontier/adalora.seed44.train.log` |
| 5 | `3252651` | PiSSA-niter-16 seed42 smoke | `logs/frontier/pissa_niter_16.seed42.smoke.log` |
| 6 | `3200201` | DoRA seed42 continues | `logs/frontier/dora.seed42.train.log` |

Current note: DoRA seed42 is healthy but very slow. It reached step `600/3000` at 2026-05-31 07:46 UTC, with best observed val loss `1.3106` at step 500. Do not expand DoRA seeds until seed42 completes or a shorter-budget policy is agreed.
