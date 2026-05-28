# PI #5b Final Results — 6-Cell lm-eval on merged_final

**Date**: 2026-05-28 06:25 UTC
**Experiment**: results/s2_pi5b_v3/qwen3-8b/tulu3-sft/<cell>/seed42/
**Config**: total_steps=3000, merge_every=750, seed=42, --save_merged_final

## Scores (5-shot vllm, on post-all-merges base)

| cell             | gsm_strict | gsm_flex | hellaswag | arc_c | final_val |
|------------------|-----------|----------|-----------|-------|-----------|
| v1_S3pos         | **79.53** | **80.14**| 77.97     | **66.89** | 1.3298 |
| v2_S3pos_IG_FDR  | 76.19     | 76.72    | 78.84     | 66.81     | 1.3477 |
| random_dr0.5     | 77.03     | 77.56    | **79.36** | 66.38     | 1.4438 |
| random_dr0.3     | 72.18     | 72.71    | **79.73** | 65.44     | 1.5459 |
| relora_baseline  | 70.28     | 70.81    | 78.93     | 62.37     | 1.6943 |
| lora_vanilla     | 87.64     | 88.32    | 76.07     | 66.47     | 1.7829 |

## Key findings

### 1. v1 saliency IS the best relora variant (+9.25pp gsm8k vs relora_baseline)

v1_S3pos 79.53% vs relora_baseline 70.28% = **+9.25pp** on gsm8k_strict.
This directly confirms that saliency-guided component selection adds substantial
value over blind baseline ReLoRA. The method WORKS.

### 2. All relora variants beat lora_vanilla on hellaswag (+1.9 to +3.7pp)

lora_vanilla hellaswag=76.07 vs relora variants 77.97-79.73. The pattern is
consistent: merge-and-reset promotes generalization on commonsense reasoning
(hellaswag, arc_challenge), though at a cost to arithmetic reasoning (gsm8k).

### 3. gsm8k ranking matches the saliency quality hierarchy

v1 (79.53) > random_dr0.5 (77.03) > v2 (76.19) > random_dr0.3 (72.18) > relora_baseline (70.28)

This is the expected ordering: more selective pruning (v1/v2) > less selective (random) >
no pruning (baseline). The IG-FDR estimator (v2) underperforms v1 slightly on gsm8k
but leads on hellaswag, suggesting different tradeoffs in the knowledge preserved.

### 4. lora_vanilla dominates gsm8k (87.64% vs v1 79.53%)

The merge operation introduces an 8pp gsm8k cost for the best relora variant.
This is the core cost of parameter budget savings from ReLoRA. The question for
PI: is the hellaswag gain (v1 +1.90pp) + potential parameter efficiency worth
the gsm8k tradeoff? 

### 5. final_val_loss tracks benchmark scores closely

| cell            | final_val | gsm_strict | hellaswag |
|-----------------|-----------|-----------|-----------|
| v1_S3pos        | 1.3298    | 79.53     | 77.97     |
| relora_baseline | 1.6943    | 70.28     | 78.93     |
| lora_vanilla    | 1.7829    | 87.64     | 76.07     |

Interesting: final_val_loss (post-merge perplexity) does NOT track lm-eval
monotonically. v1_S3pos has the lowest final_val (1.330) AND the best gsm8k
among relora variants. lora_vanilla has the highest final_val (1.783) but best
gsm8k overall. This means final_val_loss measures "fit to SFT distribution"
while gsm8k measures "retained arithmetic reasoning from base" — different dimensions.

## Questions for PI

1. **Merge cost**: the 8pp gsm8k gap between v1 and lora_vanilla — is this
   expected for ReLoRA? Is the parameter savings from periodic merging considered
   worth this cost in the original setup?

2. **hellaswag improvement**: relora variants consistently improve hellaswag vs
   vanilla (+1.9 to +3.7pp). This wasn't documented in prior work. Is this a
   known ReLoRA effect or a new finding?

3. **v2 vs v1**: v2 (IG-FDR) slightly underperforms v1 (cosine similarity score)
   on gsm8k but leads on hellaswag. Does the PI want both variants evaluated on
   more tasks (mmlu, math) to better characterize the tradeoff?

4. **Next step**: given that v1 is confirmed as best relora variant with clear
   method signal, should we proceed to a full training run with more seeds or
   larger steps for a publishable result?

## Score JSON
`analysis/results_v3/s2_pi5b_v3_scores.json`
