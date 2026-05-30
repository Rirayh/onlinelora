# Phase1 / Phase1.5 Eval Results - 2026-05-30 04:25 UTC

## Summary

Phase1 eval coverage is now complete for the three main cells:

- `v1_S3pos`: seeds 42/43/44 complete.
- `random_dr0.5`: seeds 42/43/44 complete.
- `relora_baseline`: seeds 42/43/44 complete, including the clean rerun of seed44.

Phase1.5 schedule ablation is partially complete:

- Complete: `random_anneal_up`, `random_anneal_down`, `random_triangle_up_down`.
- Running: `random_triangle_down_up` on GPU 4.

Current GPU underfill is expected: GPUs 5/6/7 are idle because only one Phase1.5 eval remains pending/running. PhaseD continues on GPUs 0-3.

## Phase1 Per-Seed Scores

Scores are percentages from the latest `lm_eval` result JSON per cell/seed.

| Cell | Seed | GSM8K strict | GSM8K flex | HellaSwag | ARC-C | MMLU | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_S3pos` | 42 | 79.61 | 79.76 | 78.13 | 67.75 | 74.67 | 40.67 |
| `v1_S3pos` | 43 | 78.32 | 78.39 | 78.51 | 68.77 | 74.87 | 43.99 |
| `v1_S3pos` | 44 | 79.08 | 79.38 | 78.34 | 68.00 | 75.21 | 40.67 |
| `random_dr0.5` | 42 | 76.35 | 76.80 | 79.41 | 66.21 | 74.43 | 41.59 |
| `random_dr0.5` | 43 | 76.42 | 76.57 | 79.53 | 65.78 | 74.11 | 36.04 |
| `random_dr0.5` | 44 | 74.68 | 75.28 | 79.29 | 67.58 | 74.25 | 41.40 |
| `relora_baseline` | 42 | 71.72 | 72.48 | 79.01 | 63.57 | 72.57 | 30.31 |
| `relora_baseline` | 43 | 71.19 | 71.49 | 79.00 | 62.37 | 72.65 | 31.42 |
| `relora_baseline` | 44 | 70.66 | 71.04 | 79.30 | 64.33 | 72.95 | 25.14 |

## Phase1 Mean Scores

| Cell | GSM8K strict | GSM8K flex | HellaSwag | ARC-C | MMLU | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_S3pos` | 79.00 | 79.18 | 78.33 | 68.17 | 74.92 | 41.77 |
| `random_dr0.5` | 75.82 | 76.22 | 79.41 | 66.52 | 74.26 | 39.68 |
| `relora_baseline` | 71.19 | 71.67 | 79.10 | 63.42 | 72.72 | 28.96 |

Interpretation:

- `v1_S3pos` is the strongest Phase1 cell on GSM8K, ARC-C, MMLU, and IFEval.
- `random_dr0.5` is slightly stronger on HellaSwag, but weaker on the other five reported metrics.
- `relora_baseline` is materially behind both `v1_S3pos` and `random_dr0.5` on GSM8K, ARC-C, MMLU, and IFEval.

## Phase1.5 Schedule Ablation

Seed 42 only.

| Schedule | Status | GSM8K strict | GSM8K flex | HellaSwag | ARC-C | MMLU | IFEval |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_anneal_up` | complete | 73.39 | 74.37 | 79.36 | 64.59 | 73.71 | 27.54 |
| `random_anneal_down` | complete | 77.94 | 78.54 | 79.12 | 67.24 | 74.75 | 45.47 |
| `random_triangle_up_down` | complete | 74.98 | 76.50 | 79.22 | 66.55 | 74.21 | 38.63 |
| `random_triangle_down_up` | running | - | - | - | - | - | - |

Notes:

- `random_anneal_down` is currently the best Phase1.5 schedule among completed runs.
- `random_anneal_up` has duplicate result JSONs from a scheduling retry; this summary uses the latest result file.
- `random_triangle_down_up` was running at 2026-05-30 04:25 UTC on GPU 4.

## Runtime State

Active as of inspection:

- GPUs 0-3: PhaseD training continues.
- GPU 4: `p1p5/random_triangle_down_up/s42` eval running.
- GPUs 5-7: idle because no additional Phase1/Phase1.5 pending evals remain.

## Result Artifacts

Raw result JSONs remain under `results/` and are intentionally not committed. This file is the lightweight commit artifact.
