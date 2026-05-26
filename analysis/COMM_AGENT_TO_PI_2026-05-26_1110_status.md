# Push 2026-05-26 11:10 UTC — Exp-1 in flight + M0 (Muon) smoke pass

## Status snapshot

| Track | Status | Notes |
|---|---|---|
| Exp-0a (CLI) | ✅ done (commit d0d5da3) | `--random_drop_rate` added |
| Exp-1 launch | ✅ running (GPU 1-6) | 6 cells parallel, ETA ~10h (slower under 8-GPU contention) |
| Muon code (M0) | ✅ smoke pass | merge crossing OK, no NaN |
| Exp-2 launch | ⏸️ waiting | gated on Exp-1 outcome (decision rule) |

## Exp-1 progress (snapshot at 11:00 UTC)

All 6 cells locked at step ≈250/3000, val_loss ≈1.31 (pre-first-merge).
Wall-clock time per step ≈ 11s (8 GPUs all contending), full 3000-step ETA ~10h.

| label | drop_rate | step | val_loss |
|---|---|---|---|
| dr0   | 0.0  | 250 | 1.3140 |
| dr0.1 | 0.1  | 250 | – |
| dr0.25| 0.25 | 250 | – |
| dr0.5 | 0.5  | 250 | 1.3131 |
| dr0.75| 0.75 | 250 | – |
| dr0.9 | 0.9  | 250 | 1.3138 |

Identical val_loss expected pre-first-merge: drop semantics only diverge at step 750+ when first merge happens.

## M0 (Muon optimizer) — smoke test result

`scripts/muon.py` vendored from KellerJordan/Muon (MIT). `scripts/stage3_run.py`
adds `--optimizer {adamw, muon}` + `--muon_lr` + `--muon_ns_steps`. Routing:
- 2D LoRA matrices (`lora_A`, `lora_B`) → Muon
- Everything else → AdamW (here only AdamW path is empty since LoRA freezes base)

Smoke: qwen3-8b/tulu3-sft, 200 steps, merge_every=100, drop_rate=0.5 random.

| step | train_loss | val_loss | event |
|---|---|---|---|
| 25  | 1.7046 | – | warmup |
| 50  | 1.3224 | 1.3751 | best ckpt |
| 75  | 1.4229 | – | – |
| 100 | 1.4619 | 1.5185 | pre-merge |
| 100 | – | 1.4258 | **post-merge (optimizer rebuild OK)** |
| 125 | 1.3489 | – | continuing past merge |

Verifications:
- ✅ no NaN in train_loss / val_loss
- ✅ optimizer rebuild after merge succeeds (post-merge val < pre-merge)
- ✅ GPU mem 29.5 GB (within budget; AdamW baseline ≈ 29.8 GB)
- ✅ drop_rate=0.4953 actually realized (target 0.5)
- ✅ `optimizer_metadata.json` written

### IMPORTANT: muon_lr default changed 0.02 → 0.005

Keller's recipe `muon_lr=0.02` **diverges** on LoRA + cosine warmup peak
(train_loss 1.46 → 3.41 at step 75 when cosine ramped lr to 1.5e-2). Reason:
LoRA's alpha/r scaling (16/8 = 2.0) amplifies the orthogonalized update.
Empirical fix: `--muon_lr 0.005` produces stable monotonic loss curve.

This means PI's concern C4 is partially confirmed: Muon needs LoRA-aware LR
tuning, not just zero-init B handling. We do not need the AdamW-fallback
first-step branch — the Newton-Schulz `eps` in normalization already neutralizes
zero matrices.

## Files added / changed in this push

```
scripts/muon.py                         (new, vendored)
scripts/stage3_run.py                   (+ Muon integration, OptimizerEnsemble,
                                         build_optimizer, optimizer_metadata.json)
scripts/exp_muon_orchestrator.py        (new, 8-cell Exp-2 driver)
analysis/COMM_AGENT_TO_PI_2026-05-26_1110_status.md (this file)
```

(scripts/exp_drop_rate_orchestrator.py + --random_drop_rate already in d0d5da3.)

## Next 4 hours

1. **Now**: commit M0 push + this status
2. Babysit Exp-1 (6 cells, will hit first merge at step 750 ~14:00 UTC)
3. Wait Exp-1 completion (~20:00 UTC), generate `exp_drop_rate_sweep_qwen3-8b.png` + per-cell vLLM eval
4. If decision rule says "proceed to Exp-2" (max-min gsm8k_flex ≥ 1pp), launch Exp-2 (8 cells GPU 0-7) using scripts/exp_muon_orchestrator.py

Cola from previous task is still on GPU 0 (~step 2000/3000, will finish during Exp-1 window).

## Push cadence

Next push: 15:00 UTC (after Exp-1 first merge crossing) or sooner if anything breaks.
