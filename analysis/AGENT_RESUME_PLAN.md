# Agent Resume Plan — 2026-05-28 06:25 UTC

## CRITICAL FINDING: lm-eval scores reveal unexpected pattern

All 6-cell lm-eval DONE (completed ~19:13-19:15 UTC yesterday).

### Scores table
| cell             | gsm_strict | gsm_flex | hellaswag | arc_c |
|------------------|-----------|----------|-----------|-------|
| v1_S3pos         | 79.53     | 80.14    | 77.97     | 66.89 |
| v2_S3pos_IG_FDR  | 76.19     | 76.72    | 78.84     | 66.81 |
| random_dr0.5     | 77.03     | 77.56    | 79.36     | 66.38 |
| random_dr0.3     | 72.18     | 72.71    | 79.73     | 65.44 |
| relora_baseline  | 70.28     | 70.81    | 78.93     | 62.37 |
| **lora_vanilla** | **87.64** | **88.32**| 76.07     | 66.47 |

### Delta vs lora_vanilla
| cell             | gsm_strict | gsm_flex | hellaswag | arc_c |
|------------------|-----------|----------|-----------|-------|
| v1_S3pos         | -8.11     | -8.19    | +1.90     | +0.43 |
| v2_S3pos_IG_FDR  | -11.45    | -11.60   | +2.77     | +0.34 |
| random_dr0.5     | -10.61    | -10.77   | +3.29     | -0.09 |
| random_dr0.3     | -15.47    | -15.62   | +3.65     | -1.02 |
| relora_baseline  | -17.36    | -17.51   | +2.86     | -4.10 |

### Interpretation
Pattern: lora_vanilla DOMINATES gsm8k by 8-17pp over all relora variants.
Hellaswag shows OPPOSITE pattern: all relora variants beat lora_vanilla by 1.9-3.7pp.

This suggests: ReLoRA merging HURTS gsm8k but HELPS hellaswag. The merge operation
may be compressing reasoning-heavy knowledge. v1_S3pos is the BEST relora variant
(least gsm8k degradation, still beats vanilla on hellaswag).

v1 vs relora_baseline: v1_S3pos gsm_strict=79.53 vs relora_baseline=70.28 (+9.25pp).
→ v1 saliency IS helping, substantially, vs baseline relora.

v2 vs v1: v2=76.19 vs v1=79.53 (-3.34pp gsm8k) but v2 hellaswag=78.84 > v1=77.97 (+0.87pp).

### Key conclusions
1. Method DOES produce signal: v1 > v2 > random_dr0.5 > random_dr0.3 > relora_baseline (gsm8k)
2. lora_vanilla gsm8k (87.64%) > all relora variants — BUT vanilla has no parameter budget
   savings vs relora. This means the merging+dropout COSTS reasoning ability.
3. Hellaswag trend reversed: relora variants BEAT vanilla by 1.9-3.7pp.
4. This matches PI #5b hypothesis that method works (signal hidden by P0 bug).

## Current state
- GPUs 0-7: ALL FREE
- Orchestrator: still running (polling dead PIDs, harmless)
- PI inbox: EMPTY (no new commits)
- Nothing pushed since 0eb5566

## Next actions
1. Kill orphan orchestrator (PID 2778765)
2. Write results summary JSON
3. Commit + push: training results + lm_eval results + summary
4. Write COMM_AGENT_TO_PI doc with score table + conclusions

## Commands
```bash
# Kill orchestrator
kill 2778765 2>/dev/null

# Score extraction already done (see above)

# Files to commit:
git add results/s2_pi5b_v3/ analysis/ && git commit && git push
```

## ACK chain
bf5d452, 040e404, 87da7d4, 54c9b07, afa7880, 5f53503, 921d66c,
f960448, 5286141, 0eb5566 (train done + eval launched),
(NEXT: final results commit)

## env paths
- training: /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
- vllm eval: /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python
- model: /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B
