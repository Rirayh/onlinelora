# Phase D Triage — gemma3-12b in-flight at daemon stop (2026-05-21 12:36)

PI directive: freeze all non-Qwen work. Triage rule: <30% kill, >70% keep, 30-70% case-by-case.
Additional override: only the 3 P0.4 cleanup cells are worth completing on tulu3.

| PID | cell | step | progress | decision | reason |
|---|---|---|---|---|---|
| 1724122 | gemma3-12b/mm/dora | 475/800 | 59% | **KILL** | mm-eval explicitly skipped per P0.3, no value in finishing |
| 1733368 | gemma3-12b/tulu3-sft/lora_vanilla | 1175/3000 | 39% | **KEEP** | listed in P0.4 cleanup |
| 1735950 | gemma3-12b/tulu3-sft/relora_baseline | 1075/3000 | 36% | **KEEP** | listed in P0.4 cleanup |
| 1759988 | gemma3-12b/tulu3-sft/relora_random_drop | 75/3000 | 2.5% | **KILL** | not in P0.4 cleanup, far below 30% |
| 1762026 | gemma3-12b/tulu3-sft/relora_diag_gated_S3pos | 500/3000 | 17% | **KEEP** | listed in P0.4 cleanup, restart cost too high |
| 1762763 | gemma3-12b/tulu3-sft/dora | <100 | <5% | **KILL** | not in P0.4 cleanup |

GPUs freed by kills: 1, 2, 4 (already idle 0,7 too) → 5 GPUs available for P1 downloads / smokes.

Survivors (estimated finish):
- 1733368 lora_vanilla: ~10h more (was ~30s/step, 1825 steps left)
- 1735950 relora_baseline: ~10h more (1925 steps left, plus extra merge events)
- 1762026 S3pos: ~22h more (2500 steps left)

Strategy: let the 3 cleanup trainings continue on GPU 3/5/6. Use GPU 0,1,2,4,7 for P1 download + smoke (downloads CPU-bound), then begin P2 Wave 1 small-model trainings as GPUs free up.
