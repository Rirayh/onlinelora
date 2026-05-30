# Phase 1 Robustness Decision — 2026-05-30

**ACK_pi_feedback_6_robustness_sweep** (Phase 1 results)

## Decision: `PROCEED_TO_PHASE2`

> v1 vs random_dr0.5 gsm8k delta=+3.18pp >= 1.5pp AND p=0.0479 < 0.1. Selection signal confirmed.

## Phase 1 Score Table (qwen3-8b / tulu3-sft, n=3 seeds)

| cell | gsm_strict | gsm_flex | hellaswag | arc_c | mmlu | ifeval |
|------|-----------|----------|-----------|-------|------|--------|
| v1_S3pos | 79.00±0.65 | 79.18±0.70 | 78.33±0.19 | 68.17±0.53 | 74.92±0.27 | 41.77±1.92 |
| random_dr0.5 | 75.81±0.99 | 76.22±0.82 | 79.41±0.12 | 66.52±0.94 | 74.26±0.16 | 39.68±3.15 |
| relora_baseline | 71.19±0.53 | 71.67±0.74 | 79.10±0.17 | 63.42±0.99 | 72.72±0.20 | 28.96±3.35 |

## Paired t-tests

| comparison | metric | delta | t | p | sig? |
|-----------|--------|-------|---|---|------|
| primary: method vs baseline | gsm8k_strict | +7.81pp | 20.880 | 0.0023 | YES |
| KEY: selection vs random (decision rule) | gsm8k_strict | +3.18pp | 4.403 | 0.0479 | YES |

## 3 Headline Deltas (95% CI via ±2*SE)

- **primary: method vs baseline**: v1_S3pos=79.00% vs relora_baseline=71.19%, delta=+7.81pp (95% CI approx ±0.73pp), p=0.0023
- **KEY: selection vs random (decision rule)**: v1_S3pos=79.00% vs random_dr0.5=75.81%, delta=+3.18pp (95% CI approx ±1.42pp), p=0.0479

## Recommended Phase 2 Model Order

1. olmo2-7b (instruct) — different architecture, best cross-arch test
2. llama3-8b (instruct) — widely used baseline in NLP benchmarks

Ready to launch Phase 2 on PI ack.
