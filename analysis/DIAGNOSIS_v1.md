# DIAGNOSIS_v1 — P0 contamination + method bug + saliency source

Author: PI directive (2026-05-24); analysis by lora-obd-agent.

---

## 1. P0 BUG: lora_B=0 contamination (FIXED in commit b7d07dc, 2026-05-19 15:45 UTC)

### Root cause
- `scripts/stage3_run.py:925` (pre-fix) called `model.save_pretrained(adapter_dir)`
  AFTER the final ReLoRA merge event zeroed `lora_B`.
- All merge-method cells (relora_*, cola) ended up with `adapter/` containing
  `lora_B = 0` → downstream lm_eval = base model score, not learned adapter.

### Evidence (PI's bit-identical reading)
- `mistral-7b/metamathqa/{relora_diag_gated_S3pos, relora_random_drop}/lm_eval/...` 
  both produce gsm8k strict-match = `0.36467020470053071` to 16 decimals.
- Mistral-7B base 8-shot GSM8K ≈ 35.4% (paper) ≈ 36.47% → both runs are base.

## 2. PI's target models for Task 1 re-eval: ['mistral-7b', 'qwen25-7b', 'qwen3-8b']

**Total cells scanned**: 115

- Tainted (merge-method + has pre-fix eval): **24**
- Needs ANY post-fix v3 (none today): 29
  - within PI targets: 10
- Needs **vLLM-on-merged** v3 (PI strict req): **67**
  - within PI targets: **20**

Existing post-fix `lm_eval_v3/` results all use `--model hf` backend and PI
requires `--model vllm` (vLLM-on-merged). Hence the larger 67-cell scope.

## 3. Method bug found by PI: identical reset for kept / dropped components

`scripts/stage3_run.py:305-306` (merge_and_reset_lora):

```python
nn.init.kaiming_uniform_(h.A, a=math.sqrt(5))
nn.init.zeros_(h.B)
```
- Both kept-by-saliency and dropped components are reset IDENTICALLY.
- Saliency only affects the W_base position (which components were merged in),
  not the next training segment's direction.
- This wipes out the gating signal for downstream learning.

**Fix (Task 2)**: add `keep_B_after_merge=True` variant that preserves the
B columns of kept components and only resets the dropped ones.

## 4. saliency_source default is wrong distribution

Default `saliency_source='val'` (SFT val), but eval is OOD GSM8K.
Importance ranking is ranked under wrong distribution.

**Fix (Task 3)**: add `--saliency_calib_set {gsm8k_train, hellaswag_val}`.

---

## Cells needing vLLM-on-merged v3 re-eval — PI target models

| model | dataset | method | adapter source | existing v3 backend |
|---|---|---|---|---|
| mistral-7b | metamathqa-10k | relora_baseline | best/ | (none) |
| mistral-7b | metamathqa-10k | relora_diag_gated_S3neg | best/ | (none) |
| mistral-7b | metamathqa-10k | relora_diag_gated_S3pos | adapter | hf |
| mistral-7b | metamathqa-10k | relora_random_drop | adapter | hf |
| mistral-7b | metamathqa-10k | relora_train_gated | adapter | hf |
| qwen25-7b | metamathqa-10k | relora_baseline | best/ | (none) |
| qwen25-7b | metamathqa-10k | relora_diag_gated_S3neg | best/ | (none) |
| qwen25-7b | metamathqa-10k | relora_diag_gated_S3pos | adapter | hf |
| qwen25-7b | metamathqa-10k | relora_random_drop | best/ | (none) |
| qwen25-7b | metamathqa-10k | relora_train_gated | best/ | (none) |
| qwen3-8b | metamathqa-10k | relora_baseline | best/ | (none) |
| qwen3-8b | metamathqa-10k | relora_diag_gated_S3neg | best/ | (none) |
| qwen3-8b | metamathqa-10k | relora_diag_gated_S3pos | adapter | hf |
| qwen3-8b | metamathqa-10k | relora_random_drop | best/ | (none) |
| qwen3-8b | metamathqa-10k | relora_train_gated | best/ | (none) |
| qwen3-8b | tulu3-sft | relora_baseline | adapter | hf |
| qwen3-8b | tulu3-sft | relora_diag_gated_S3neg | adapter | hf |
| qwen3-8b | tulu3-sft | relora_diag_gated_S3pos | adapter | hf |
| qwen3-8b | tulu3-sft | relora_random_drop | adapter | hf |
| qwen3-8b | tulu3-sft | relora_train_gated | adapter | hf |

## Cells needing vLLM-on-merged v3 re-eval — other models (Wave 1 + legacy)

| model | dataset | method | adapter source | existing v3 backend |
|---|---|---|---|---|
| gemma3-12b | metamathqa-10k | cola | best/ | (none) |
| gemma3-12b | metamathqa-10k | relora_baseline | best/ | (none) |
| gemma3-12b | metamathqa-10k | relora_diag_gated_S3pos | best/ | (none) |
| gemma3-12b | metamathqa-10k | relora_random_drop | best/ | (none) |
| gemma3-12b | tulu3-sft | relora_baseline | best/ | (none) |
| gemma3-12b | tulu3-sft | relora_diag_gated_S3pos | best/ | (none) |
| llama3-8b | metamathqa-10k | cola | best/ | (none) |
| llama3-8b | metamathqa-10k | relora_baseline | best/ | (none) |
| llama3-8b | metamathqa-10k | relora_diag_gated_S3pos | adapter | hf |
| llama3-8b | metamathqa-10k | relora_random_drop | best/ | (none) |
| llama3-8b | tulu3-sft | cola | adapter | hf |
| llama3-8b | tulu3-sft | relora_baseline | adapter | hf |
| llama3-8b | tulu3-sft | relora_diag_gated_S3pos | adapter | hf |
| llama3-8b | tulu3-sft | relora_random_drop | adapter | hf |
| olmo2-7b | metamathqa-10k | cola | best/ | (none) |
| olmo2-7b | metamathqa-10k | relora_baseline | best/ | (none) |
| olmo2-7b | metamathqa-10k | relora_diag_gated_S3pos | adapter | hf |
| olmo2-7b | metamathqa-10k | relora_random_drop | best/ | (none) |
| olmo2-7b | tulu3-sft | cola | adapter | hf |
| olmo2-7b | tulu3-sft | relora_baseline | adapter | hf |
| olmo2-7b | tulu3-sft | relora_diag_gated_S3pos | adapter | hf |
| olmo2-7b | tulu3-sft | relora_random_drop | adapter | hf |
| qwen3-14b | tulu3-sft | cola | best/ | (none) |
| qwen3-14b | tulu3-sft | relora_diag_gated_S3pos | best/ | (none) |
| qwen3-1p7b | tulu3-sft | cola | adapter | hf |
| qwen3-1p7b | tulu3-sft | relora_baseline | adapter | hf |
| qwen3-1p7b | tulu3-sft | relora_diag_gated_S3pos | adapter | hf |
| qwen35-0p8b | tulu3-sft | cola | adapter | hf |
| qwen35-0p8b | tulu3-sft | relora_baseline | adapter | hf |
| qwen35-0p8b | tulu3-sft | relora_diag_gated_S3pos | adapter | hf |
| qwen35-2b | tulu3-sft | cola | adapter | hf |
| qwen35-2b | tulu3-sft | relora_baseline | adapter | hf |
| qwen35-2b | tulu3-sft | relora_diag_gated_S3pos | adapter | hf |
| qwen35-4b | tulu3-sft | cola | adapter | hf |
| qwen35-4b | tulu3-sft | relora_baseline | adapter | hf |
| qwen35-4b | tulu3-sft | relora_diag_gated_S3pos | best/ | (none) |
| qwen35-9b | tulu3-sft | cola | adapter | hf |
| qwen35-9b | tulu3-sft | relora_baseline | adapter | hf |
| qwen35-9b | tulu3-sft | relora_diag_gated_S3pos | best/ | (none) |
| r1-distill-7b | metamathqa-10k | cola | best/ | (none) |
| r1-distill-7b | metamathqa-10k | relora_baseline | best/ | (none) |
| r1-distill-7b | metamathqa-10k | relora_diag_gated_S3pos | adapter | hf |
| r1-distill-7b | metamathqa-10k | relora_random_drop | best/ | (none) |
| r1-distill-7b | tulu3-sft | cola | adapter | hf |
| r1-distill-7b | tulu3-sft | relora_baseline | adapter | hf |
| r1-distill-7b | tulu3-sft | relora_diag_gated_S3pos | adapter | hf |
| r1-distill-7b | tulu3-sft | relora_random_drop | adapter | hf |

## Action plan

1. **Task 1 (this commit)**: write `scripts/find_p0_tainted_evals.py` (done).
   Output: `analysis/p0_tainted_manifest.json` + this DIAGNOSIS_v1.md.
2. **Task 1 (next commit)**: rename pre-fix `lm_eval/` to `lm_eval_PRE_P0_FIX_TAINTED/`,
   re-eval all `needs_vllm_v3` cells via vLLM-on-merged.
3. **Task 2** (separate commit): keep_B_after_merge variant.
4. **Task 3** (same commit as Task 2): OOD calib saliency loader.
5. **Task 4** (separate commit): controlled experiment 7-method × 2-data × 2-model × 3-seed.

---

## Notes on Wave 1 (Phase D) status

Wave 1 training (qwen3-{1p7b,4b,14b}, qwen35-{0p8b,2b,4b,9b}) was launched ~2026-05-21,
AFTER the b7d07dc fix landed. So adapter/ files are clean (lora_B != 0).
HOWEVER, the `lm_eval_v3/` results in Wave 1 cells used `--model hf` backend
(not vLLM-on-merged), and qwen3-* cells did use vLLM-on-merged (correct).
Per PI's strict 'all evals via vLLM-on-merged' rule, qwen35-* HF evals also need redo.
Listed in 'other models' table above.
