# Phase1.5 n=3 Decision - 2026-05-31

## Verdict

`v1_S3pos` still has a GSM8K advantage over the strongest schedule-matched random arm, but the advantage is small and not decisive at n=3.

- `v1_S3pos` vs `random_anneal_down` on GSM8K strict: `+1.01pp` mean paired delta, paired t-test `p=0.0914`.
- `random_anneal_down` beats `v1_S3pos` on IFEval by `+2.90pp` mean paired delta.
- `random_anneal_down` also has a small HellaSwag edge: `+0.75pp`, paired t-test `p=0.0338`.

Interpretation: saliency selection still appears useful for GSM8K recovery, but drop-rate schedule is a major confound and may be better for instruction-following behavior. The paper claim should be softened from a broad advantage to a metric-specific one.

## n=3 Summary

| Cell | n | GSM8K strict | IFEval | HellaSwag |
| --- | ---: | ---: | ---: | ---: |
| `v1_S3pos` | 3 | 79.00 +/- 0.65 | 41.77 +/- 1.92 | 78.33 |
| `random_anneal_down` | 3 | 77.99 +/- 0.38 | 44.67 +/- 2.24 | 79.07 |
| `random_const_0p5` | 3 | 75.82 +/- 0.99 | 39.68 +/- 3.15 | 79.41 |
| `relora_baseline` | 3 | 71.19 +/- 0.53 | 28.96 +/- 3.35 | 79.10 |

Seed42-only schedule probes remain useful as scouts but are no longer decision-grade:

| Cell | n | GSM8K strict | IFEval | HellaSwag |
| --- | ---: | ---: | ---: | ---: |
| `random_anneal_up` | 1 | 73.39 | 27.54 | 79.36 |
| `random_triangle_up_down` | 1 | 74.98 | 38.63 | 79.22 |
| `random_triangle_down_up` | 1 | 72.55 | 42.70 | 79.47 |

## Paired Deltas: `v1_S3pos - random_anneal_down`

| Metric | seed42 | seed43 | seed44 | Mean delta | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K strict | +1.67 | +0.68 | +0.68 | +1.01 | 0.0914 |
| IFEval | -4.81 | -2.40 | -1.48 | -2.90 | 0.1000 |
| HellaSwag | -0.99 | -0.76 | -0.50 | -0.75 | 0.0338 |
| MMLU | -0.09 | +0.32 | +0.53 | +0.25 | 0.2935 |

## Consequence for the Method Story

The strongest defensible statement now is:

> Saliency-guided component selection improves GSM8K over schedule-matched random by about 1pp at n=3, but the effect is borderline and does not dominate all downstream metrics. Anneal-down scheduling is a strong baseline and is better on IFEval.

This does not kill the method. It does narrow the claim: the advantage is task-sensitive, not universal.

## Files

- Aggregate JSON: `analysis/results_v3/phase1p5_n3_and_frontier_partial.json`
- Source result roots:
  - `results/phase1_robustness/qwen3-8b/tulu3-sft/`
  - `results/phase1p5_schedule_ablation/qwen3-8b/tulu3-sft/random_anneal_down/`
