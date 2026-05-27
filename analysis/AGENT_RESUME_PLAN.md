# Agent Resume Plan — 2026-05-27 18:40 UTC

## Active state: 6-cell lm-eval IN FLIGHT on GPUs 0-5

All 6 cells trained + merged_final saved. lm-eval launched 18:35 UTC.
ETA per cell ~45-60min → all done ~19:30-19:45 UTC.

| GPU | cell             | PID (vllm sub) | eval log                              |
|-----|------------------|----------------|---------------------------------------|
| 0   | v1_S3pos         | 2779947        | logs/s2_pi5b_v3/v1_S3pos.eval.log    |
| 1   | v2_S3pos_IG_FDR  | 2779952        | logs/s2_pi5b_v3/v2_S3pos_IG_FDR.eval.log |
| 2   | random_dr0.5     | 2779957        | logs/s2_pi5b_v3/random_dr0.5.eval.log |
| 3   | random_dr0.3     | 2779973        | logs/s2_pi5b_v3/random_dr0.3.eval.log |
| 4   | relora_baseline  | 2779978        | logs/s2_pi5b_v3/relora_baseline.eval.log |
| 5   | lora_vanilla     | 2779983        | logs/s2_pi5b_v3/lora_vanilla.eval.log |

Orchestrator PID: 2778765, log: logs/s2_pi5b_v3/eval_orchestrator.log

## Training results (post-merge val_loss, significant signal already visible)
| cell             | final_val | best_val | best_step |
|------------------|-----------|----------|-----------|
| v1_S3pos         | 1.3298    | 1.3134   | 500       |
| v2_S3pos_IG_FDR  | 1.3477    | 1.3126   | 250       |
| random_dr0.5     | 1.4438    | 1.3122   | 500       |
| random_dr0.3     | 1.5459    | 1.3112   | 500       |
| relora_baseline  | 1.6943    | 1.3141   | 250       |
| lora_vanilla     | 1.7829    | 1.3124   | 500       |

Note: final_val_loss is the POST-ALL-MERGES val_loss (correct metric now).
v1_S3pos (1.330) is substantially better than relora_baseline (1.694) and
lora_vanilla (1.783). This is the first clean signal we've ever seen.

## After eval done
1. Check results/s2_pi5b_v3/.../lm_eval/results_*.json for all 6 cells
2. Parse scores (gsm8k_strict, gsm8k_flex, hellaswag, arc_challenge)
3. Build comparison table; PI acceptance = >5pp delta v1/v2 vs baseline
4. git add results/s2_pi5b_v3/ scripts/s2_pi5b_v3_eval.py analysis/ && commit + push

## Commands to check progress
```bash
# eval progress
for c in v1_S3pos v2_S3pos_IG_FDR random_dr0.5 random_dr0.3 relora_baseline lora_vanilla; do
  echo -n "$c: "
  tail -3 /mnt/cpfs/junlongke/onlinelora/lora_obd/logs/s2_pi5b_v3/${c}.eval.log 2>/dev/null | grep -oE "(gsm8k|Running|Completed)" | head -1
done

# orchestrator status
tail -10 /mnt/cpfs/junlongke/onlinelora/lora_obd/logs/s2_pi5b_v3/eval_orchestrator.log

# results present?
for c in v1_S3pos v2_S3pos_IG_FDR random_dr0.5 random_dr0.3 relora_baseline lora_vanilla; do
  echo -n "$c: "
  ls /mnt/cpfs/junlongke/onlinelora/lora_obd/results/s2_pi5b_v3/qwen3-8b/tulu3-sft/$c/seed42/lm_eval/ 2>/dev/null | head -1
done
```

## Key paths
- training results: results/s2_pi5b_v3/qwen3-8b/tulu3-sft/<cell>/seed42/
- merged models: .../seed42/merged_final/ (16.4GB per cell, 4 shards)
- lm_eval output: .../seed42/lm_eval/

## ACK chain
bf5d452, 040e404, 87da7d4, 54c9b07, afa7880, 5f53503, 921d66c,
f960448, 5286141 (ACK_5b + 6-cell train launched),
(next: final results + commit + push)

## env paths
- training: /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
- vllm eval: /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python
- model: /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B
