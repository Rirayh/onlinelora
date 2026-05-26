# S1 Framing Test — Cross-Model Summary

**Date:** 2026-05-26
**PI directive:** `2026-05-26_pi_feedback_on_s1.md` §1 BLOCKING
**Pass condition:** `rho_global < 0.15` on ≥ 2/3 models

## Method

ε-scaled-B trick: at endpoint W = W0 + ΔW (loaded adapter), set B := ε·B (ε = 1e-3).
- forward = base + ε·B@A·scaling, so for tiny ε the network behavior ≈ base model
- ε is a multiplicative scalar on saliency → Spearman ρ invariant
- Compare s_end (computed at full B) vs s_eps (computed at ε·B) per LoRA component

n_calib = 256, max_len = 512, dataset = tulu3-sft.

## Results

| model | adapter | n_components | rho_global | p_global | sign_flip_rate | top10pct_iou_keep | top10pct_iou_drop |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3-8b | exp_v1/relora_baseline/seed42 | 4032 | **0.0242** | 0.124 | 0.452 | 0.126 | 0.028 |
| r1-distill-7b | stage3_v2/relora_baseline/seed42 | 3136 | **-0.0090** | 0.613 | 0.475 | 0.120 | 0.026 |
| qwen3-4b | stage3_v2/relora_baseline/seed42 | 4032 | **-0.0357** | 0.023 | 0.553 | 0.123 | 0.027 |

## Verdict

**PASS — 3/3 models satisfy `|rho_global| < 0.15`**.

All three rho_global values are within ±0.04 of zero, far below the 0.15 threshold. Sign-flip rates near 0.5 (random chance) and top-10% IoU near 0.10 (random chance for 10% subset) confirm:

> The first-order saliency at the endpoint W = W0 + ΔW carries **no signal** about which components were helpful at the start W = W0.

V1 estimator (computed at endpoint) is therefore an incorrect proxy for the start-point saliency that the rank-recycling theory requires. This is the cross-model evidence that justifies the IG-based v2 estimator.

**DECISION (all three models):** `A_CRITICAL_implement_IG`

## Cross-model signal stability

The near-zero ρ + ~0.5 sign-flip rate is consistent across:
- Architectural variants (Qwen3 GQA vs R1-Distill)
- Model scale (4B / 7B / 8B)
- Training source (exp_v1 sweep vs stage3_v2 baselines)

This rules out a single-model artifact and confirms the framing problem is generic to LoRA endpoint gradients.
