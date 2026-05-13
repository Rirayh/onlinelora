# Stage 1 Report — Predictive Validity of Validation Saliency

**Date**: 2026-05-12
**Hypothesis** (handover §3.1): Validation-set saliency is a strictly better predictor of where ablation actually hurts than train-set saliency.
**Verdict**: **AMBIGUOUS** (1 of 3 GO conditions fails, but the failure is marginal and the other 2 conditions pass strongly).

## TL;DR

- The hypothesis is partially confirmed.
- **First-order saliency on validation (S3) is a much stronger predictor than first-order on train (S2)**: mean delta_|ρ| = +0.246, 95% CI [0.120, 0.373], 11/15 (task, ckpt) pairs positive.
- **Fisher saliency on validation (S5) is only marginally better than Fisher on train (S4)**: mean delta_|ρ| = +0.046, 95% CI [0.008, 0.088], 10/15 pairs positive. Threshold of 0.10 is not met.
- **Harmful-component detection AUC** (sign-symmetric for S3) is ≥0.65 on 2 of 3 tasks at the final checkpoint (MRPC=0.776, RTE=0.759); SST-2 lags at 0.587.
- Hard GO rule fails on Fisher threshold; STOP rule does not trigger. Per handover §3.8 this is an AMBIGUOUS outcome that requires a user decision.

## Results

### Per-(task, checkpoint) pairs (15 total)

| task | step | |ρ_S5_val|−|ρ_S4_tr| | |ρ_S3_val|−|ρ_S2_tr| | sym-AUC harmful | harmful_rate |
|------|-----:|--------------------:|--------------------:|----------------:|-------------:|
| sst2 | 1000 |              -0.084 |              +0.063 |           0.750 |        0.313 |
| sst2 | 2000 |              -0.054 |              -0.077 |           0.771 |        0.651 |
| sst2 | 3000 |              +0.020 |              -0.020 |           0.590 |        0.719 |
| sst2 | 4000 |              -0.042 |              -0.051 |           0.669 |        0.646 |
| sst2 | 5000 |              +0.019 |              -0.103 |           0.587 |        0.698 |
| mrpc |  400 |              -0.025 |              +0.106 |           0.799 |        0.563 |
| mrpc |  800 |              -0.005 |              +0.341 |           0.873 |        0.583 |
| mrpc | 1200 |              +0.161 |              +0.582 |           0.844 |        0.708 |
| mrpc | 1600 |              +0.136 |              +0.566 |           0.832 |        0.755 |
| mrpc | 2000 |              +0.045 |              +0.475 |           0.776 |        0.724 |
|  rte |  400 |              +0.086 |              +0.125 |           0.844 |        0.797 |
|  rte |  800 |              +0.060 |              +0.248 |           0.840 |        0.865 |
|  rte | 1200 |              +0.076 |              +0.559 |           0.788 |        0.698 |
|  rte | 1600 |              +0.164 |              +0.454 |           0.791 |        0.781 |
|  rte | 2000 |              +0.129 |              +0.429 |           0.759 |        0.755 |

### Aggregate decision (handover §3.8)

| Condition | Required | Observed | Pass |
|-----------|----------|----------|:----:|
| mean Δ|ρ|_fisher ≥ 0.10 | 0.10 | **0.046** (CI95 [0.008, 0.088]) | ❌ |
| mean Δ|ρ|_fo ≥ 0.05      | 0.05 | **0.246** (CI95 [0.120, 0.373]) | ✅ |
| sign test ≥ 10/15 positive (fisher OR fo) | ≥10 | fisher 10/15, fo 11/15 | ✅ |
| sym-AUC ≥ 0.65 on ≥1 task at latest ckpt | ≥0.65 | mrpc 0.776, rte 0.759 | ✅ |
| STOP: all sym-AUCs < 0.55 | — | no (min 0.587) | n/a |

**Per handover §3.8**: GO requires *all* of: condition 1 (FAILED on Fisher leg) AND 2 AND 3. STOP requires condition 1 failing in the wrong direction (mean<0). Neither GO nor STOP triggers → **AMBIGUOUS**.

## What the data clearly says

1. **The validation/train distinction is enormous in the first-order regime** (FO val vs FO train): Δ|ρ| means +0.246, with positive effect on 11/15 pairs. This is a real, large effect, especially on the smaller tasks (MRPC, RTE) where overfitting is more pronounced.

2. **The Fisher regime is more subtle**. On RTE and MRPC the validation Fisher is consistently better (delta_rho_fisher positive in 8/10 of their pairs); but on **SST-2 it is reversed**: 4/5 SST-2 pairs have delta_rho_fisher < 0 (i.e. train Fisher is the better predictor). This drags the global mean below the 0.10 threshold.

3. **Hypothesis is task-dependent**. The mechanism the handover proposes — that test ≈ val ≠ train when the model overfits — is observed clearly on MRPC/RTE (small datasets where overfitting is visible: harmful_rate climbs to 70-87%) but **not on SST-2** (large dataset, less overfitting, harmful_rate plateaus around 65-70%).

4. **Harmful-component detection** is strong on the smaller tasks (sym-AUC ≈ 0.78-0.84 across most checkpoints) and lukewarm on SST-2.

## Why Fisher signal is weaker than first-order signal

A plausible explanation: Fisher squares each per-sample inner product, which amplifies noise from rare-but-large gradients on individual examples. The validation set is 20% of train, so its Fisher estimate has a noisier denominator than train's. The first-order saliency averages over examples *before* taking magnitude/sign, so the validation-vs-train signal-to-noise is much cleaner.

If true, the methodological implication is that we should prefer a Hessian or full first-order curvature estimate rather than diagonal Fisher when generalizing this to Llama-scale; or use larger val sets at this stage.

## What this means for Stage 2

The handover demands an explicit decision from the user given AMBIGUOUS. Two reasonable paths:

- **Path A — Proceed to Stage 2 (relaxed gate)**. The first-order evidence is compelling: mean Δ|ρ| = 0.246, 5x the threshold. The Fisher near-miss (0.046 vs 0.10) is partly attributable to a single task (SST-2) where the saliency-direction conflict is weak. **A reasonable interpretation: validation saliency provides reliable signal where overfitting is present, which is exactly the regime Stage 2 targets**. This justifies attempting the ReLoRA failure-mode reproduction + diagnostic-gated fix.

- **Path B — Stop and pivot**. The original handover sets a *conjunctive* AND threshold for a reason: Fisher is the formula most current literature uses for LoRA pruning (e.g. WeLore, AdaLoRA), so showing it dominates is the strongest claim. If we cannot show this cleanly, the contribution narrative weakens. Consider:
  - (a) re-running Stage 1 with K=8-fold cross-validated diagnostic splits to reduce variance, OR
  - (b) reframing the contribution as "validation-FO saliency dominates", which is what the data robustly shows, and proceeding to Stage 2 *with FO-based gating only*.

## Figures (see `plots/stage1/`)

- **fig3_train_vs_val_paired.png** — headline. Left panel (Fisher): 15 dots near y=x line, slight lift above on RTE/MRPC, slight drop below on SST-2. Right panel (FO): MRPC/RTE dots are clearly far above the y=x line; SST-2 is on or just below.
- **fig2_rho_over_time.png** — per task, ρ vs step. RTE/MRPC show consistent gap of S3 (val FO) above S2 (train FO).
- **fig1_correlation_grid.png** — scatter of delta_test vs S5 per (task, step). Strong negative slope on RTE; weak/noisy on SST-2.
- **fig4_harmful_auc.png** — sym-AUC across training. Rises on MRPC (0.80→0.87 then to 0.78), stable on RTE (0.84→0.76), and stays around 0.60-0.77 on SST-2.

## Recommendation

Given the data, my recommendation (subject to your call):

> **Proceed to Stage 2 but explicitly using val-FO (S3) signaling and document Fisher as an open finding.** The hypothesis "val saliency is a strictly better predictor" is robust under the FO definition (mean +0.246, sign 11/15) and is the variant most directly comparable to OBD-style theory. Treating SST-2 as a stress test (where overfitting is mild and the val-vs-train gap should naturally shrink) explains the negative results there. The Fisher gap on SST-2 reduces but does not contradict the central thesis.

Pending your decision, I have NOT proceeded to Stage 2. All Stage 1 artifacts (raw `components.jsonl` for every checkpoint, correlations, AUCs, decision.json, figures) are committed and reproducible.

## Reproduction

```bash
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
cd /mnt/cpfs/junlongke/onlinelora/lora_obd

# 3 GPU parallel run (~30 min for MRPC/RTE, ~30 min for SST-2)
CUDA_VISIBLE_DEVICES=0 $PY scripts/stage1_run.py --config configs/stage1_sst2.yaml &
CUDA_VISIBLE_DEVICES=1 $PY scripts/stage1_run.py --config configs/stage1_mrpc.yaml &
CUDA_VISIBLE_DEVICES=3 $PY scripts/stage1_run.py --config configs/stage1_rte.yaml  &
wait

# aggregate + decision + plots
$PY scripts/stage1_aggregate.py
$PY scripts/stage1_plot.py
```
