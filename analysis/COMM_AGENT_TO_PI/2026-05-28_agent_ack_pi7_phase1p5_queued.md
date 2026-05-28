# Agent ACK: PI Feedback #7 Phase 1.5 Schedule Ablation

**Date**: 2026-05-28
**Re**: `2026-05-28_pi_feedback_7_phase1p5_schedule_ablation.md`
**Tag**: ACK_pi_feedback_7_phase1p5_schedule_ablation

---

## ACK

Confirmed. Phase 1.5 understood: 4 schedule variants (random selection,
varying drop-rate shape) to disambiguate "saliency adds value" from
"v1's anneal_up-like shape adds value".

---

## Execution Status

### Phase 1.5 — QUEUED, auto-launches when 4 GPUs free

Orchestrator `scripts/phase1p5_train_orchestrator.py` (PID 2875635) is
polling every 120s. All 8 GPUs currently occupied by Phase 1. Phase 1.5
will launch on the first 4 GPUs that become free (~19:00 UTC when Phase 1
jobs complete).

4 cells:
| cell | method | drop_schedule |
|------|--------|--------------|
| random_anneal_up | relora_random_drop | anneal_up → [0.25,0.35,0.45,0.55] |
| random_anneal_down | relora_random_drop | anneal_down → [0.75,0.65,0.55,0.45] |
| random_triangle_up_down | relora_random_drop | triangle_up_down → [0.25,0.45,0.65,0.65] |
| random_triangle_down_up | relora_random_drop | triangle_down_up → [0.75,0.55,0.35,0.35] |

Reused (no retraining):
- `random_const_0p5` = `results/s2_pi5b_v3/.../random_dr0.5/seed42/`
- `v1_S3pos` = `results/s2_pi5b_v3/.../v1_S3pos/seed42/`

Config: total_steps=3000, merge_every=750, seed=42, --save_merged_final.
Output: `results/phase1p5_schedule_ablation/qwen3-8b/tulu3-sft/<cell>/seed42/`

Estimated launch: ~19:00 UTC → completion ~05:00 UTC May 29 →
lm-eval ~07:00 UTC May 29 → Phase 1.5 decision pushed ~08:00 UTC May 29.

---

## Scripts added

| script | purpose |
|--------|---------|
| `scripts/phase1p5_train_orchestrator.py` | polls + launches when 4 GPUs free |
| `scripts/phase1p5_decision_analysis.py` | ranking table + delta + verdict + writes phase1p5_decision.md |

---

## Decision rule implemented

```
delta_v1_vs_best_random_schedule = v1_gsm8k - max(random_*_gsm8k)

>= 2.0pp   -> SALIENCY_ADDS_VALUE         (paper story stays)
[0, 2.0)   -> SALIENCY_WEAKLY_ADDS_VALUE  (soften claim)
< 0        -> SCHEDULE_DOMINATES_STORY_FLIP
```

Phase 1.5 runs regardless of Phase 1 outcome (per §D directive).

---

## Timeline (combined Phase 1 + 1.5 + D)

| time (UTC) | event |
|------------|-------|
| ~19:00 May 28 | Phase 1 training done; Phase 1.5 launches on freed GPUs |
| ~19:00 May 28 | Phase 1 lm-eval starts (phase1D_eval_orchestrator.py --phase1) |
| ~24:00 May 28 | Phase 1 lm-eval done → phase1_summary.json + phase1_decision.md pushed |
| ~05:00 May 29 | Phase 1.5 training done |
| ~07:00 May 29 | Phase 1.5 lm-eval done → phase1p5_summary.json + phase1p5_decision.md pushed |
| ~02:00 May 30 | Phase D training done → summary pushed |
