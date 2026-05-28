# Agent ACK: PI Feedback #6 Robustness Sweep + Story Pivot

**Date**: 2026-05-28
**Re**: `2026-05-28_pi_feedback_6_robustness_sweep.md`
**Tag**: ACK_pi_feedback_6_robustness_sweep

---

## ACK

Confirmed. Story pivot, decision rule, and compute plan understood. Executing now.

---

## Execution Status

### Phase 1 (§C.1) — LAUNCHED at 09:32 UTC

qwen3-8b / tulu3-sft × 3 cells × 3 seeds = 9 cells. 8 launched immediately on
GPUs 0-7; `relora_baseline/seed44` queued (will auto-launch when any GPU frees,
~9.5h from now).

| cell | seed | GPU | PID |
|------|------|-----|-----|
| v1_S3pos | 42 | 0 | 2863584 |
| v1_S3pos | 43 | 1 | 2863585 |
| v1_S3pos | 44 | 2 | 2863586 |
| random_dr0.5 | 42 | 3 | 2863587 |
| random_dr0.5 | 43 | 4 | 2863588 |
| random_dr0.5 | 44 | 5 | 2863589 |
| relora_baseline | 42 | 6 | 2863590 |
| relora_baseline | 43 | 7 | 2863591 |
| relora_baseline | 44 | — | queued |

Config identical to s2_pi5b_v3: total_steps=3000, merge_every=750,
eval_every=250, --save_merged_final, lr/optim unchanged.

Estimated completion: ~09:32 + ~9.5h = **~19:00 UTC today**.
Then lm-eval (6 tasks: gsm8k, hellaswag, arc_challenge, mmlu, ifeval, arc_c)
on all 9 merged_final/ → ~4-5h → **~24:00 UTC today**.

Output: `results/phase1_robustness/qwen3-8b/tulu3-sft/<cell>/seed{42,43,44}/`

### Phase D (§D) — queued, launches when Phase 1 GPUs free (~19:00 UTC)

4 cells: lora_vanilla × {42,43} + v1_S3pos × {42,43}, 10000 steps each.
Orchestrator `scripts/phaseD_train_orchestrator.py` ready; will launch
automatically when 4 GPUs become available.

Estimated Phase D completion: ~19:00 + ~31h = **~26h from now (Phase D is
the slow path)**. PI estimate of 12 GPU-h appears to assume 4 parallel GPUs
× 3h wall-clock — however qwen3-8b at 10k steps is closer to 31h/GPU.
Flagging this: Phase D will complete ~50h from now, not 24h.
Phase 1 lm-eval completes in ~24h as requested.

---

## Scripts added

| script | purpose |
|--------|---------|
| `scripts/phase1_train_orchestrator.py` | Phase 1 training launch |
| `scripts/phaseD_train_orchestrator.py` | Phase D over-train launch |
| `scripts/phase1D_eval_orchestrator.py` | lm-eval for phase1 + phaseD (6 tasks) |
| `scripts/phase1_decision_analysis.py` | mean/std/paired-t + decision rule + writes phase1_decision.md |

---

## Story pivot: understood

New framing locked in documentation and analysis scripts:
> Saliency-aware ReLoRA closes 53% of the gsm8k gap between ReLoRA baseline
> and vanilla LoRA, while preserving ReLoRA's hellaswag advantage.

3 headline numbers to validate with error bars:
1. v1 vs relora_baseline gsm8k Δ (+9.25pp seed42, need n=3)
2. v1 vs random_dr0.5 gsm8k Δ (+2.5pp seed42 — the decision number)
3. v1 vs lora_vanilla hellaswag Δ (+1.9pp seed42)

Decision rule implemented in `phase1_decision_analysis.py`:
PROCEED iff Δ(v1,random) ≥ 1.5pp AND p < 0.10 (paired t, n=3).

---

## Reporting timeline

| time (UTC) | deliverable |
|------------|-------------|
| ~19:00 today | Phase 1 training done; lm-eval starts |
| ~24:00 today | Phase 1 lm-eval done; `phase1_summary.json` + `phase1_decision.md` pushed |
| ~19:00+31h ≈ 2026-05-30 02:00 | Phase D training done; lm-eval + summary pushed |

Will push Phase 1 decision before launching Phase 2 per §C.1 directive.
