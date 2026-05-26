# Wake-up state — 2026-05-26 13:00 UTC (S1 RESULT IN, big finding)

## CRITICAL S1 RESULT (just completed)

`analysis/results_v3/saliency_framing/spearman_qwen3-8b_tulu3.json`:
- **rho_global = 0.0242** (essentially zero)
- sign_flip_rate = 0.4521 (near-random)
- top10pct_iou keep=0.1257 drop=0.0281
- **DECISION: A_CRITICAL_implement_IG**

Interpretation: endpoint saliency (W = W₀ + ΔW) is ~uncorrelated with start-point
saliency (W ≈ W₀). PI's hypothesis H1 confirmed → v1 estimator is the wrong proxy
for "what to keep at next merge cycle". IG (saliency_v2) is mandatory.

## Last user message
"按你建议来吧" — execute (a) S1, (b) S2.5 schedule pilot, (c) saliency_v2.py

PI directive: `analysis/COMM_PI_TO_AGENT/2026-05-26_pi_saliency_revamp_v2.md`
ACK string: `ACK_v2_saliency_revamp` (already in commit fa4534e body)

## What's COMMITTED + PUSHED so far

- `d0d5da3`: Exp-0a `--random_drop_rate` CLI (earlier)
- `5e6056c`: M0 Muon optimizer (earlier; now path-δ fallback per v2)
- **`fa4534e`** (current): S1 `scripts/run_s1_framing_test.py` + `src/saliency_v2.py`
- Pulled `0bc2e02`: PI v2 saliency revamp directive

## What's UNCOMMITTED (ready to commit + push)

1. ✅ `scripts/stage3_run.py` edited with:
   - new CLI: `--saliency_estimator {v1,v2}`, `--saliency_v2_m_ig` (4),
     `--saliency_v2_alpha` (0.1), `--drop_schedule <spec>`
   - `DROP_SCHEDULE_REGISTRY` (11 schedules) + `parse_drop_schedule()`
   - random branch wired to per-event rates from schedule (event_idx-1)
   - gated branch forks: v1 unchanged, v2 uses `integrated_gradient_saliency_per_sample`
     + `t_stat_decision` + `fisher_signvote_score`. OOM retry path included.
   - Verified `parse_drop_schedule` works on registry/comma/random/empty inputs.
2. ✅ `scripts/exp_schedule_pilot_orchestrator.py` — 12 schedules driver
   (3 reused from Exp-1 dr0.25/0.5/0.75; 9 new cells)
3. ✅ `analysis/results_v3/saliency_framing/spearman_qwen3-8b_tulu3.json` — S1 result
4. ✅ `analysis/WAKE_UP_INSTRUCTIONS.md` — this file

## Active running jobs (snapshot 13:00 UTC)

| GPU | Job | Status |
|---|---|---|
| 0 | cola (exp_v1) | step ~2200/3000, almost done |
| 1-6 | Exp-1 drop-rate sweep dr{0,0.1,0.25,0.5,0.75,0.9} | step ~750/3000; first merge crossing now |
| 7 | **FREE** (S1 done in 75s; very fast) |

## REMAINING todo (in order)

1. ⏳ git add + commit + push (S1 result, stage3 edits, orchestrator). Commit body
   should include `ACK_v2_saliency_revamp` and S1 result summary.
2. ⏳ Status doc to PI: report S1 result decision (must implement IG) + plan
3. ⏳ Wait for Exp-1 to finish (~20:00 UTC), kick off vllm eval per cell
4. ⏳ Launch S2.5 schedule pilot (after Exp-1 done; uses 7 GPUs)
5. ⏳ Run a small smoke of v2 estimator (1-2hr training to verify wiring works
   end-to-end on real model). Suggested: GPU 7 free now — train 200 steps
   `--method relora_diag_gated_S3pos --saliency_estimator v2 --saliency_calib_n 64`
   on qwen3-8b/tulu3. Critical to verify v2 path doesn't crash before S2.5 launch.

## Repo / env

- Repo: `/mnt/cpfs/junlongke/onlinelora/lora_obd`
- Python: `/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python`
- Model: `/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B`
- S1 baseline adapter (used + verified): `results/exp_v1/qwen3-8b/tulu3-sft/relora_baseline/seed42/adapter/`

## Commit message ready to paste

```
S1 result + stage3 v2/schedule wiring + S2.5 orchestrator

ACK_v2_saliency_revamp

S1 framing test FINDING (analysis/results_v3/saliency_framing/...):
  rho_global = 0.0242, sign_flip_rate = 0.4521,
  top10pct_iou keep = 0.13 drop = 0.03.
  DECISION: A_CRITICAL_implement_IG.
  -> endpoint saliency uncorrelated with start-point saliency on trained
     relora_baseline; v1 estimator is wrong proxy for next-cycle keep ranking.

scripts/stage3_run.py:
  + --saliency_estimator {v1,v2} (default v1; no behaviour change)
  + --saliency_v2_m_ig (default 4), --saliency_v2_alpha (default 0.1)
  + --drop_schedule <spec>: registry name | comma list |
      random_schedule:seed=N | empty (=constant)
  + DROP_SCHEDULE_REGISTRY: 11 schedules (const_*, anneal_*, triangle_*,
      *_burst, extreme_alternate)
  + parse_drop_schedule(spec, n_events) helper
  Wiring:
  - random branch: drop_schedule_list[event_idx-1] when set, else
      args.random_drop_rate. stats["scheduled_drop_rate"] recorded.
  - gated branch: forks on saliency_estimator. v1 unchanged. v2 uses
      integrated_gradient_saliency_per_sample (m=4, B->t*B per step) +
      t_stat_decision (BH-FDR alpha=0.1, Bernoulli random fallback) +
      fisher_signvote_score (reported as score quantiles).
      OOM retry path: half samples + half seq_len.

scripts/exp_schedule_pilot_orchestrator.py (new):
  12 schedules driver (qwen3-8b/tulu3-sft, total_steps=3000, merge_every=500
  = 6 events). Reuses Exp-1 dr0.25/0.5/0.75 cells; 9 new cells launch when
  GPUs free.

src/saliency.py UNCHANGED (per directive: keep v1 reproducible).

Next: smoke v2 path (1-2h on GPU 7); after Exp-1 done launch S2.5 pilot.
```

## Smoke test command (recommended next)

```bash
cd /mnt/cpfs/junlongke/onlinelora/lora_obd && \
nohup env CUDA_VISIBLE_DEVICES=7 /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python \
  scripts/stage3_run.py \
  --model_path /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B \
  --model_key qwen3-8b --dataset tulu3-sft \
  --method relora_diag_gated_S3pos \
  --saliency_estimator v2 \
  --saliency_calib_n 64 \
  --total_steps 200 --merge_every 100 --eval_every 50 --ckpt_every 9999 \
  --saliency_max_seq_len 512 --attn_implementation sdpa \
  --seed 42 --out_root results/v2_smoke/seed42 \
  > logs/v2_smoke.log 2>&1 &
```

Expected: at step 100 first merge event, log line "[v2 estimator m_ig=4 alpha=0.1]"
appears, then `n_keep_sig`, `n_drop_sig`, `n_random` in info; train continues
with non-NaN loss after merge.

## Push cadence: every 4hr. Last 11:00 UTC. Next: 15:00 UTC.

## EXACT next action when context resumes

1. `git -C /mnt/cpfs/junlongke/onlinelora/lora_obd add scripts/stage3_run.py
   scripts/exp_schedule_pilot_orchestrator.py
   analysis/results_v3/saliency_framing/spearman_qwen3-8b_tulu3.json
   analysis/WAKE_UP_INSTRUCTIONS.md`
2. Commit with message above; push.
3. Launch v2 smoke on GPU 7 (command above).
4. Wait Exp-1 first merge crossing (~14:00 UTC) — check logs/exp_drop_rate/.
5. Status doc + push at ~15:00 UTC to PI.
