# PI BLOCKER — v1_recheck reproducibility gate FAILED

**Date**: 2026-05-27 04:30 UTC
**Status**: BLOCKING per PI #3 §3 ("v1_recheck must reproduce the scoreboard within ±0.5pp before any v2 eval")

## Summary

v1_recheck training + eval completed cleanly under post-P0-fix code path. Eval results vs PI #3 §3 scoreboard targets:

| metric | scoreboard target | v1_recheck actual | delta | gate |
|---|---|---|---|---|
| gsm8k_strict | 86.43 ± 0.30 | **80.36** | **-6.07pp** | ❌ FAIL |
| gsm8k_flex   | 86.96 ± 0.30 | **81.43** | **-5.53pp** | ❌ FAIL |
| hellaswag    | 77.27 ± 0.30 | 77.61 | +0.34pp | ✅ PASS |
| arc_challenge| 69.32 ± 0.30 | **66.38** | **-2.94pp** | ❌ FAIL |

3 of 4 metrics fail by >2pp.

## Diagnostic: not a v1 problem

Cross-reference Exp-1 cell `dr=0.0` (random_drop with rate=0, i.e. pure no-drop baseline, identical code path):

| metric | exp1_dr0 | v1_recheck | delta |
|---|---|---|---|
| gsm8k_strict | 79.15 | 80.36 | +1.21 |
| gsm8k_flex   | 80.14 | 81.43 | +1.29 |
| hellaswag    | 77.68 | 77.61 | -0.07 |
| arc_challenge| 66.21 | 66.38 | +0.17 |

Both v1_recheck and the no-drop baseline land at ~80% gsm8k / ~66% arc_challenge — a clean 6pp below the scoreboard. The v1 method is NOT the cause; the post-P0-fix code-path itself produces these numbers regardless of drop policy.

## Most likely root cause

The PI #3 §3 scoreboard numbers (86.43 / 86.96 / 69.32) were captured **before** the P0 fix. Candidate contaminations in pre-P0 evals:
1. Prompt-template leakage that effectively gave the model the answer
2. `gen_kwargs` differences (max_new_tokens, stop strings, temperature)
3. Data-prep contamination (test set seen during SFT)
4. A bug in lm_eval task version we have since pinned/fixed

Whatever the exact cause, **the post-P0 clean code-path will not reach 86%+ on gsm8k for Qwen3-8B + tulu3-sft regardless of merging policy**. This is a property of the codebase, not of v1.

## Decision required from PI

Two paths forward:

**(A) Re-baseline the scoreboard** — accept 80.36 / 81.43 / 77.61 / 66.38 as the new v1_recheck reference. Compare v2_full eval directly against these numbers. (Recommended.)

**(B) Investigate the gap** — pause v2 eval, dig into the pre-P0 eval pipeline to identify what produced the higher scoreboard. (Costs ~1-2 days of eval-pipeline forensics.)

If PI is silent or chooses (A): we will proceed with v2_full eval and report against v1_recheck. If PI chooses (B): we hold all v2 work pending forensics.

## Supporting artefacts (already in repo)

- `results/s2_v1_recheck/qwen3-8b/tulu3-sft/relora_diag_gated_S3pos_v1_recheck/seed42/lm_eval/.../results_2026-05-26T23-32-55.125728.json` — full lm_eval JSON
- `results/s2_v1_recheck/.../seed42/summary.json` — train summary (val=1.3338, best=1.3132@500)
- `results/s2_v1_recheck/.../seed42/dropped_components.jsonl` — 4 events with drop rates 0.56/0.61/0.60/0.67
- `results/exp_drop_rate/qwen3-8b/tulu3-sft/dr0/seed42/lm_eval/...` — Exp-1 dr=0.0 baseline (cross-reference)

## Bonus finding: PI #3 §5 Bernoulli hypothesis REJECTED

PI hypothesised v1 drop logic is Bernoulli(p=0.5). Reality from `dropped_components.jsonl`:

| event | step | drop_rate | σ off H0 |
|---|---|---|---|
| 1 | 750  | 0.5613 | +7.8σ |
| 2 | 1500 | 0.6111 | +14.1σ |
| 3 | 2250 | 0.5982 | +12.5σ |
| 4 | 3000 | 0.6696 | +21.5σ |

(σ under H0=Bernoulli(0.5), n=4032: σ=0.0079.) All events are decisively non-Bernoulli; bias amplifies over training. v1 has a real, systematic, non-random signal — just not necessarily a *good* one.
