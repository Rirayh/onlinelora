# PI Feedback #7 — Phase 1.5 schedule ablation (parallel to Phase 1)

**Date**: 2026-05-28
**Severity**: Story-decisive — separates "saliency adds value" from "v1's drop-rate shape adds value"
**Relationship to #6**: ADDS Phase 1.5 in parallel; does not modify Phase 1 / Phase 2 / D.

---

## TL;DR

Realized while reviewing #6: v1's effective drop_rate is monotonically
increasing 0.56 → 0.61 → 0.60 → 0.67 across the 4 events (from
v1_recheck/dropped_components.jsonl). This **shape** alone may explain part
of v1's gsm8k advantage, independent of which components saliency picks.

DROP_SCHEDULE_REGISTRY in `scripts/stage3_run.py:189-201` already has all
required schedules implemented but **only `anneal_up` and `anneal_down` have
been run, and only at 300-step smoke**. The triangle / burst / late /
extreme variants have never been trained end-to-end at full 3000 steps.

This directive: **Phase 1.5** — 5-cell parallel ablation on qwen3-8b that
disambiguates "v1 wins because saliency picks well" from "v1 wins because
its drop-rate shape happens to be anneal_up-like".

## Why this matters

If `random_anneal_up` (purely random selection but with v1's drop-rate
shape) reaches v1's gsm8k ≈79.53% within 1.5pp:
> **v1 saliency adds no measurable selection benefit beyond drop-rate
> scheduling.** The +2.5pp v1 vs random_dr=0.5 in #5b is then attributable
> to schedule shape, not selection quality. Paper pivots to
> "drop-rate scheduling matters more than which directions get dropped".

If `random_anneal_up` is meaningfully below v1 (>=2pp):
> v1 selection is genuinely informative beyond schedule. Paper story stays
> as in #6 with this ablation as a strong supporting claim.

This is **the cleanest single experiment** to determine where v1's
contribution actually comes from. It must run before paper-write.

## Phase 1.5 — schedule ablation

### Setup
- model: qwen3-8b (single model — Phase 2 broadens to olmo2/llama3)
- dataset: tulu3-sft
- training: total_steps=3000, merge_every=750, --save_merged_final, lr/optim
  identical to s2_pi5b_v3
- seed: 42 (single seed; Phase 1.5 is a discrimination test, not a
  significance test — if the result is interesting we can add seeds 43/44 in
  a follow-up)

### 5 cells (parallel-launchable on 5 GPUs, ~10h wall-clock each)

| cell | spec | purpose |
|---|---|---|
| `random_anneal_up` | `--method relora_random_drop --drop_schedule anneal_up` | mimics v1 drop shape with random selection |
| `random_anneal_down` | `--drop_schedule anneal_down` | reverse shape, control |
| `random_triangle_up_down` | `--drop_schedule triangle_up_down` | middle-burst |
| `random_triangle_down_up` | `--drop_schedule triangle_down_up` | endpoint-burst |
| `random_const_0p5` (already have from #5b) | n/a, reuse | flat-shape baseline |

**Note**: v1_S3pos is already trained from #5b — reuse `analysis/results_v3/`
adapter for comparison. No need to retrain v1.

### Eval
Same lm-eval pipeline as #5b: gsm8k_strict + gsm8k_flex + hellaswag + arc_challenge,
on `merged_final/` ckpt. Add mmlu (5-shot) + ifeval if Phase 1 has them by
the time Phase 1.5 evals.

### Output
- `analysis/results_v3/phase1p5_schedule_ablation/{cell}/seed42/scores.json`
- `analysis/results_v3/phase1p5_summary.json` (one row per cell, sorted by gsm8k)

### Analysis & decision rule

After all 5 schedules eval, compute:
```
delta_v1_vs_best_random_schedule = v1_gsm8k - max(random_*_gsm8k)
```

**Interpretation**:
- `delta_v1_vs_best_random_schedule >= 2.0pp`:
  v1 saliency adds value beyond schedule. Paper claim "saliency-aware
  selection beats matched-shape random" is supported. Continue with #6
  Phase 1 / Phase 2.
- `0 <= delta < 2.0pp`:
  v1 weakly above schedule. Saliency contributes but small. Paper softens
  to "saliency provides a modest improvement over schedule-matched
  random."
- `delta < 0` (some random schedule beats v1):
  **Story flips.** Paper repositions to "drop-rate scheduling drives
  ReLoRA gsm8k recovery; saliency selection has no measurable benefit
  on this benchmark." This is still publishable (negative result on
  saliency + positive result on schedule) but is a different paper.

Push verdict to `analysis/COMM_AGENT_TO_PI/{date}_phase1p5_decision.md`
with the gsm8k ranking table and the interpretation above.

## Compute budget

| item | cells | GPU-h | wall @ 5 GPU |
|---|---|---|---|
| Phase 1.5 train | 4 new × ~10h = 40 GPU-h | ~10h | concurrent Phase 1 |
| Phase 1.5 eval | 4 cells × vllm batch | ~5 GPU-h | <2h |
| **total** | | **~45** | **fits inside Phase 1 24h window** |

Total cumulative budget after Phase 1.5: 90 (#6 Phase 1) + 12 (#6 D) + 45
(this) = ~147 GPU-h, ~24h on 8 GPUs if Phase 1 + 1.5 + D run concurrently.

## Reporting

After Phase 1.5 completes (concurrently with #6 Phase 1 + D):
- `analysis/results_v3/phase1p5_schedule_ablation/` raw scores
- `analysis/results_v3/phase1p5_summary.json` ranking
- `analysis/COMM_AGENT_TO_PI/{date}_phase1p5_decision.md` with interpretation

Do not block on PI ack — push results and continue with #6 Phase 2 if both
Phase 1 and Phase 1.5 indicate proceed.

## What stays unchanged from #6

- Phase 1 (qwen3-8b × {v1, random_dr0.5, baseline} × 3 seeds + mmlu + ifeval): RUN
- Phase 2 (olmo2-7b + llama3-8b × same 3 cells × 2 seeds): conditional on Phase 1
- D (vanilla overtrain 10k step, n=2): RUN
- Phase 1 decision rule: 1.5pp + paired-t p<0.10 vs random_dr=0.5

Phase 1.5 RUNS REGARDLESS of Phase 1 outcome. Even if Phase 1 fails
(v1 ~ random_dr=0.5), Phase 1.5 reveals whether ANY random schedule
recovers gsm8k — which is itself a finding.

## ACK

`ACK_pi_feedback_7_phase1p5_schedule_ablation`

Within 4h: Phase 1.5 launched on dedicated GPUs (concurrent with Phase 1
+ D). State GPU assignments.

Within 24h: Phase 1.5 results pushed alongside Phase 1 + D.

---

## Personal note

This is the experiment I should have requested in #6 alongside Phase 1.
The user (PI) flagged it: "随机 drop schedule 形状一样能不能赶上 v1?"
The intuition is clean and the experiment costs very little in wall-clock
(5 GPUs × 10h, parallelizable with Phase 1).

If `random_anneal_up` reaches 78pp on gsm8k → v1 saliency adds nothing
beyond schedule and we re-write. If it stays below 76pp → v1 selection is
real. Single experiment, decisive answer.

The 11 schedules in DROP_SCHEDULE_REGISTRY have been sitting unused for
weeks; this finally exercises them at full scale.
