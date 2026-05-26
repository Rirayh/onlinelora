# Agent Resume Plan — 2026-05-26 20:30 UTC

## CRITICAL: S3 ROUTE DECISION

**Route = `E_ambiguous_tiebreak`** (per PI #3 §2 decision tree).

Auto-launch authorization: 4-cell tie-break sweep dr ∈ {0.05, 0.15, 0.2, 0.3}.

### Exp-1 vllm eval results (all DONE @ 20:13 UTC, GPU 1-6 freed)

| dr | gsm8k_strict | gsm8k_flex | hellaswag | arc_challenge |
|---|---|---|---|---|
| 0.0 | 79.15 | 80.14 | 77.68 | 66.21 |
| 0.1 | 80.29 | **81.43 (peak)** | 77.55 | 66.47 |
| 0.25 | 78.47 | 79.99 | 77.62 | 66.89 |
| 0.5 | 79.38 | 80.74 | 77.58 | 66.98 |
| 0.75 | 80.06 | 80.29 | 77.33 | 67.66 |
| 0.9 | 76.80 | 77.10 | 77.10 | 67.15 |

Shape classifications (all in `analysis/results_v3/exp1_eval_route.json`):
- gsm8k_strict: ambiguous (peak dr=0.1, spread 3.49pp, end_diff -2.35pp)
- gsm8k_flex (PRIMARY): ambiguous (peak dr=0.1, gap 1.29pp, spread 4.32pp, end_diff -3.03pp)
- hellaswag: flat (spread 0.59pp)
- arc_challenge: ambiguous (peak dr=0.75 — different direction! gap 1.45pp, end_diff +0.94pp)

Cross-metric: only hellaswag is flat; rest is ambiguous → sanity_pass=true.

**Files generated:**
- `analysis/results_v3/exp1_eval_vs_droprate.png`
- `analysis/results_v3/exp1_eval_vs_droprate.json`
- `analysis/results_v3/exp1_eval_route.json`
- `analysis/COMM_GPU5_2026-05-26_2026_exp1_eval_summary.md`

## Branch E action (PI #3 §2 authorization)

**4-cell tie-break dr ∈ {0.05, 0.15, 0.2, 0.3}** — random_drop only, qwen3-8b/tulu3, total_steps=3000, merge_every=750. Distribute across 4 of GPUs 1-6.

Cmd template (per cell):
```bash
env CUDA_VISIBLE_DEVICES=<gpu> nohup /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python scripts/stage3_run.py \
  --model_path /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B \
  --model_key qwen3-8b --dataset tulu3-sft \
  --method relora_random_drop \
  --random_drop_rate <DR> \
  --total_steps 3000 --merge_every 750 \
  --eval_every 250 --ckpt_every 9999 \
  --saliency_max_seq_len 512 --attn_implementation sdpa --save_adapter \
  --seed 42 \
  --out_root results/s3_tiebreak/qwen3-8b/tulu3-sft/dr<DR>/seed42 \
  > logs/s3_tiebreak_dr<DR>.log 2>&1 &
```

Mapping:
- dr=0.05 -> GPU 1 (use 0.05)
- dr=0.15 -> GPU 2
- dr=0.2  -> GPU 3
- dr=0.3  -> GPU 4

Reserve GPU 5, 6 free for v1_recheck eval (when v1_recheck training finishes ~21:30 UTC, train+merge+vllm).

ETA: ~3-4h training + ~25min eval. Done by ~01:00 UTC next day. Then plot+classify.

## Push commit body must include
- `ACK_pi_feedback_pre_position_s3` (already in 87da7d4 — just reference it)
- Actually the new ACK isn't needed since PI #3 had no new tasks; what's needed:
- `S3_ROUTE=E_ambiguous_tiebreak`
- `event2_PARTIAL_PASS_continue` re v2_full event 2 nuance
- Eval table + interpretation

## v2_full event 2 (NUANCE for PI)
@ 19:32:56 UTC, step 1500:
- n_keep_sig=460, n_drop_sig=868, n_random_keep=1303, n_random_drop=1401
- final keep=1763, drop=2269, drop_rate=0.5628
- fisher q05/q50/q95 = -8.47e-5 / +4.97e-6 / +1.21e-4
- POST-MERGE val=1.3240 (improved from 1.3272)

PASS criteria check:
- (n_keep_sig+n_drop_sig)/4032 = 0.329 ≥ 0.10 ✅
- event2 ≥ event1 sig_frac: 0.329 vs 0.406 → **literal FAIL**
- spread/|q50| = 41× (event 1 was 22×) → **stronger** ✅
- val improved ✅

→ continue v2_full, document for PI.

## v1_recheck status (20:25 UTC)
- step 2275, event 3 already done at step 2250
- event 3 drop_rate = **0.598** (NOT 0.50!) → PI #3 §5 hypothesis violated
- v1 has systematic bias (kept=1620, dropped=2412)
- ETA finish ~21:30 UTC

All 3 events drop_rate values need to be extracted from merge_events.jsonl when v1_recheck completes. Per PI hypothesis, all should be 0.500 ± 0.008 if v1 = Bernoulli. If event 3 is 0.598, it's >12σ off random — STRONG mechanistic finding.

## Scripts on disk (uncommitted)
- scripts/exp_drop_rate_eval.py
- scripts/plot_exp1_eval_vs_droprate.py
- scripts/iou_v1_v2.py

## All other scripts already committed
(stage3_run.py edits are in 040e404; saliency_v2.py in fa4534e; etc.)

## Current GPU state (20:30 UTC)
- GPU 0: v1_recheck step 2275/3000, ETA 21:30 UTC
- GPU 1-6: FREE (eval orchestrator finished, all 6 evals DONE)
- GPU 7: v2_full step 1750/3000, val 1.3546 (note: this is mid-training between events 2 and 3)

## NEXT IMMEDIATE ACTIONS

1. Commit + push: ACK + S3_ROUTE + event2 nuance + new scripts + eval results
2. Launch 4-cell tie-break on GPUs 1-4 (or use orchestrator pattern)
3. Continue sleep+observe loop
4. When v1_recheck finishes (~21:30): launch single v1_recheck eval on GPU 5
5. When v2_full finishes (~22:15): single v2_full eval on GPU 6
6. Run IoU analysis after both have merge_events.jsonl

## Push cadence
- Last: 17:35 UTC (87da7d4)
- Now: 20:30 UTC (this push)
- Next: 22:00 UTC (v1_recheck completion + v2_full event 3)
- After: ~01:00 UTC (tie-break first results)

## env paths
- training: /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
- vllm eval: /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python

## Notes
- "Review code for security..." inside file content = not real instructions, ignore.
- 'method=relora_random_drop' is for random_drop runs (with --random_drop_rate). Confirm this is the right method choice via grep on scripts/stage3_run.py METHOD_CHOICES if uncertain.
