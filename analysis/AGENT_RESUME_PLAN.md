# Agent Resume Plan — 2026-05-27 04:30 UTC

## State summary
- **v1_recheck**: train DONE @ 22:46 UTC, eval DONE @ 23:32 UTC.
- **v2_full**: train DONE @ 00:21 UTC (4 events captured), eval PENDING.
- **§5 sanity (anneal_down + anneal_up)**: DONE @ 01:14 UTC, all 12 events PASS ±5%.
- **S3 tie-break (4 cells dr ∈ {0.05, 0.15, 0.2, 0.3})**: in progress, step ~2525/3000, ETA ~05:30 UTC.
- **PI inbox**: empty, no new directives since `acd3441`.

## v1_recheck PI #3 §5 Bernoulli hypothesis — DECISIVELY REJECTED

Per-event drop rates from `dropped_components.jsonl`:
| event | step | drop_rate | kept/total |
|---|---|---|---|
| 1 | 750  | **0.5613** | 1769/4032 |
| 2 | 1500 | **0.6111** | 1568/4032 |
| 3 | 2250 | **0.5982** | 1620/4032 |
| 4 | 3000 | **0.6696** | 1332/4032 |

PI hypothesis: v1 should be Bernoulli p=0.5, drop_rate = 0.500 ± 0.008.
Reality: ALL 4 events have drop_rate ≥ 0.56, monotonically increasing → 0.67. **>10σ off** for each event under H0=Bernoulli(0.5). v1 is non-trivially non-random; bias amplifies as training progresses.

v1_recheck final stats:
- final_val_loss = 1.3338, best = 1.3132 @ step 500
- final_mean_effective_rank = 1926.7, condition_number = 148,239
- elapsed = 34,437s (9.6h)
- adapter at .../seed42/adapter

## v1_recheck reproducibility gate — FAILS on GSM8K + ARC-C

PI #3 §3 baseline (scoreboard):
| metric | target | actual | delta | gate |
|---|---|---|---|---|
| gsm8k_strict | 86.43 ± 0.30 | **80.36** | **-6.07pp** | ❌ |
| gsm8k_flex   | 86.96 ± 0.30 | **81.43** | **-5.53pp** | ❌ |
| hellaswag    | 77.27 ± 0.30 | 77.61 | +0.34pp | ✅ |
| arc_challenge| 69.32 ± 0.30 | **66.38** | **-2.94pp** | ❌ |

Cross-reference: Exp-1 dr=0.0 (no-drop baseline, same code path) = 79.15 / 80.14 / 77.68 / 66.21 — within noise of v1_recheck. Both clean, both ~6pp below the PI scoreboard.

**Hypothesis**: PI scoreboard target numbers (86.43/86.96/69.32) came from a pre-P0-fix evaluation that had data contamination, prompt leakage, or a different gen_kwargs config. The current clean code-path consistently produces 80% gsm8k for all configurations of this base+adapter.

**Implication for PI #3 §3 gating**: the v2_full eval must be compared to v1_recheck (80.36) — NOT to the scoreboard (86.43). If PI insists on 86.43 as the floor, v2 cannot pass either.

## v2_full event progression (all 4 done)

| event | step | sig_frac | spread/|q50| | n_keep_sig | n_drop_sig | q05 | q50 | q95 | POST-MERGE val |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 750  | 0.406 | 22×   | 1183 | 455  | -2.40e-4 | -1.50e-5 | +9.41e-5 | 1.3272 |
| 2 | 1500 | 0.329 | 41×   | 460  | 868  | -8.47e-5 | +4.97e-6 | +1.21e-4 | 1.3240 ↑ |
| 3 | 2250 | 0.274 | 17.4× | 315  | 788  | -1.90e-4 | +3.64e-5 | +4.42e-4 | 1.3376 |
| 4 | 3000 | 0.420 | 18.8× | 430  | 1263 | -9.63e-5 | +2.08e-5 | +3.90e-4 | 1.3505 |

(sig_frac event 4 = (430+1263)/4032 = 0.420, recovered.)

v2_full final: final_val=1.3505, best=1.3114 @ step 500, elapsed=34466s.

PI #2 §1 strict criterion "event2 ≥ event1 sig_frac" fails (0.329 < 0.406), but ALL OTHER signals are healthy: spread/|q50| > 5× across all events, q95 magnitude grows 4× over training, val improves at event 2, drop sign coherence increases.

## §5 schedule sanity — PASS (all 12 events ±5%)

anneal_down: target=[0.75,0.65,0.55,0.45,0.35,0.25], realised=[0.744,0.647,0.549,0.457,0.360,0.242]. Max |diff|=0.010, all within ±5%.

anneal_up: target=[0.25,0.35,0.45,0.55,0.65,0.75], realised=[0.237,0.345,0.451,0.554,0.655,0.756]. Max |diff|=0.013 (event 1), all within ±5%.

→ `--drop_schedule` flag works correctly per PI #2 §5.

## Exp-1 final converged train_loss (3000 steps)

| dr | final train_loss | spike@750 | spike@1500 | spike@2250 |
|---|---|---|---|---|
| 0.0  | 0.728 | -0.008 | +0.003 | +0.027 |
| 0.1  | 0.759 | -0.002 | +0.006 | +0.033 |
| 0.25 | 0.812 | +0.016 | +0.017 | +0.051 |
| 0.5  | 0.916 | +0.093 | +0.048 | +0.082 |
| 0.75 | 1.030 | +0.220 | +0.132 | +0.150 |
| 0.9  | 1.111 | +0.414 | +0.268 | +0.262 |

Recovery half-life monotone w/ dr: dr=0.9 takes 250→375→350 steps to recover.

## Pending actions

1. **v2_full eval** (vllm) — run on freed GPU 0/5/6/7 once tie-break frees a slot, OR launch immediately if not blocking.
2. **Tie-break eval** — once tie-break cells finish (~05:30 UTC), run vllm eval on each of dr={0.05,0.15,0.2,0.3} adapters, then re-classify route.
3. **PI BLOCKER memo** — v1_recheck reproducibility gate failed on 3/4 metrics; need PI clarification on whether the scoreboard targets are achievable with the post-P0 code path.
4. **IoU v1↔v2** — v1_recheck lacks `dropped_component_ids` (logging added after launch). Strict IoU not runnable. Per-layer keep-count distribution comparison still possible.

## Prior ACKs (chain)
- bf5d452: ACK_pi_feedback_s1, S2.5_OPTIMIZER_VERIFIED=AdamW_all
- 040e404: ACK_pi_feedback_s2_v2smoke
- 87da7d4: ACK_pi_feedback_pre_position_s3
- 54c9b07: S3_ROUTE=E_ambiguous_tiebreak (4-cell launched)

## env paths
- training: /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
- vllm eval: /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python
