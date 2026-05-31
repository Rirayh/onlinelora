# Frontier Method Partial Results - 2026-05-31

## Completed So Far

### AdaLoRA seed42

AdaLoRA seed42 completed training and final lm-eval.

| Method | Seed | GSM8K strict | GSM8K flex | HellaSwag | ARC-C | MMLU | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adalora` | 42 | 87.57 | 88.25 | 76.07 | 66.47 | 74.78 | 25.32 |

Important caveat: AdaLoRA rank diagnostics are invalid in the current runner (`#LoRA layers=0`, `mean_ER=nan`) because `get_lora_BA_handles()` does not recognize PEFT AdaLoRA handles. The quality metric is still usable, but any rank/condition-number claims for AdaLoRA are N/A.

The benchmark profile is very similar to the long LoRA overtrain profile: very high GSM8K, low IFEval. Treat this as a capability/behavior tradeoff, not an unconditional win.

## Running Frontier Jobs

| Method | Seed | PID | Status at 2026-05-31 13:48 UTC | Notes |
| --- | ---: | ---: | --- | --- |
| `adalora` | 43 | `3250178` | step `1725/3000` | best val loss so far `1.3068` at step 1250 |
| `adalora` | 44 | `3250040` | step `1750/3000` | best val loss so far `1.3083` at step 750 |
| `dora` | 42 | `3200201` | still running | much slower than AdaLoRA/PiSSA; seed expansion deferred |
| `pissa_niter_16` | 42 | `3258837` | step `1750/3000` | overfitting trend: val loss rose from `1.4114` at step 250 to `1.7193` at step 1750 |
| `pissa` | 42 | `3281699` | launched on GPU5 | plain PiSSA control for PiSSA-niter-16 |

## Current Read

- AdaLoRA looks strong on GSM8K but weak on IFEval, matching the overtrain/behavior tradeoff seen in PhaseD.
- PiSSA-niter-16 is technically running correctly, but early validation loss suggests aggressive overfitting on this setup. Final eval is still needed before rejecting it.
- DoRA is too slow to expand until seed42 completes or a shorter-budget protocol is chosen.

## Next Actions

1. Wait for AdaLoRA seed43/44 completion, then run final lm-eval and compute n=3.
2. Wait for PiSSA-niter-16 seed42 completion, but watch for severe overfit. If final performance is bad, use best-val checkpoint behavior as a separate diagnostic.
3. Evaluate plain `pissa` seed42 after completion to separate PiSSA initialization from the niter variant.
4. Do not use AdaLoRA rank stats until the handle collector is fixed.
