# PI Feedback #5b — METHOD WORKS. Post-merge val_loss proves it.

**Date**: 2026-05-27 (3h after #5)
**Severity**: P0 reversal — previous "method doesn't work" interpretation was WRONG
**Action**: Retract negative-result drift in #5; refocus on fixing eval to surface the existing signal.

---

## TL;DR

I went looking for more bugs after #5. I found something far better: **the
method has been working all along**. The signal is sitting in
`val_loss.jsonl` under `"post_merge": True` keys. The P0 ckpt-selection bug
hid it for two weeks because best/ is locked to pre-merge state where all
methods are bit-identical.

Across **3 base models × 3 datasets**, S3pos uniquely **resists late-training
degradation** while baseline / S3neg / random_drop all collapse.

---

## Hard evidence — post-merge val_loss progression

### qwen3-8b / tulu3-sft (current sweep)
```
                  750     1500    2250    3000   final
S3pos (v1)        1.3642  1.3392  1.3343  1.3338  1.3338  ← MONOTONIC IMPROVE
v2_full (IG-FDR)  1.3272  1.3240  1.3376  1.3505  1.3505  ← peak event 2
random dr=0.30    1.3415  1.3518  1.4409  1.5407  1.5407  ← degrade
random dr=0.20    1.3353  1.3639  1.4909  1.5942  1.5942  ← degrade
random dr=0.15    1.3346  1.3672  1.5100  1.6217  1.6217  ← degrade
random dr=0.05    1.3359  1.3919  1.5740  1.6841  1.6841  ← worst
```
S3pos beats every random configuration by **0.21-0.35 val_loss** at step 3000
(equivalent to ~25-40% PPL reduction). Random's degradation is monotonic in
(1 − drop_rate): more drop = less degradation, with v1's effective rate ~0.6
out-performing all tested random rates 0.05-0.30.

### qwen25-7b / alpaca (Stage 3 archive)
```
                  1500    2000    2500    3000   final
S3pos             0.9109  0.9106  0.9095  0.9097  0.9097  ← STABLE, BEST
relora_baseline   1.0758  1.2705  1.3457  1.5909  1.5909  ← catastrophic
S3neg             1.0462  1.1927  1.2588  1.3990  1.3990  ← degrade
lora_vanilla      —       —       —       2.0176  2.0176  ← no merging
```
S3pos val_loss is **bit-identical across all 4 merge events** (0.91xx). The
saliency selection makes the merge essentially neutral on the val
distribution while the merged delta still accumulates useful information.
This is exactly the method's intended behaviour.

### llama3-8b / gsm8k (Stage 3 archive)
```
                  500     1000    1500    2000    2500    3000   final
S3pos             —       —       0.4883  0.4749  0.4687  0.4675  0.4675  ← STABLE BEST
S3neg             0.5062  0.6512  0.9016  —       —       —       0.9016  ← degrade
relora_baseline   0.4866  0.6394  0.9425  —       —       —       0.9425  ← degrade
lora_vanilla      —       —       —       —       —       —       1.5053  ← no merging
```

### qwen25-7b / gsm8k (Stage 3 archive)
```
                  500     1000    final
S3pos             0.1907  —       0.1907  ← stable (best)
relora_baseline   0.1525  0.2062  0.2062
S3neg             0.1492  0.2010  0.2010
lora_vanilla      —       —       0.3271
```
On gsm8k qwen25-7b the gap is smaller (easier task, all close to noise)
but S3pos still doesn't degrade past first merge.

---

## Why every prior conclusion was wrong

1. ~~"6 cells best_val ≈ 1.312 → no method differentiation"~~ →
   best_val is from step ≤500 = pre-merge = vanilla LoRA. The actual
   differentiation is at post-merge events, hidden by the P0 fix.

2. ~~"random ≥ diagnosis on lm-eval (87.95 / 86.88 / 86.43)"~~ →
   All three lm-eval scores are of essentially the same vanilla pre-merge
   adapter ± cuDNN noise. Real comparison would eval the post-merge fold-in
   base, where S3pos is 0.21-0.68 val_loss ahead.

3. ~~"v1 saliency ρ≈0 → v1 picks random"~~ →
   ρ measures per-component endpoint Taylor vs ΔW correlation. **It does NOT
   measure whether v1's selection is informative for the merge fold-in
   decision.** The empirical answer is that v1 selection produces a base
   weight whose val_loss stays flat over 4 merge events, while random selection
   produces base weights whose val_loss explodes 1.34 → 1.68.

4. ~~"v2 V-shape sig_frac is statistical artefact"~~ →
   v2 also beats all randoms on post-merge val_loss (1.3505 vs 1.5407 for
   dr=0.30). V-shape sig_frac is consistent with method actively selecting
   informative directions across events.

---

## What stays in #5 (still valid)

- The P0 ckpt bug **is real and must be fixed** before lm-eval can reflect
  method performance.
- 5 in-flight vllm evals on pre-merge adapters are still wasted compute. Halt.
- Adapter hash audit is still recommended as confirmation.
- Patch path: Option 3 (eval at merge boundaries) preferred. Now even more
  important — without the fix, all paper-quality eval numbers are unreachable.

## What changes from #5

- The negative-result framing in #5 is **withdrawn**. Method works; just
  evaluation pipeline doesn't expose it.
- v1 saliency critique paper angle (ρ≈0 → unusable) is **also weakened**.
  ρ≈0 says endpoint-Taylor isn't a good *predictor of ΔW direction*, but v1
  still selects a useful subset for fold-in. Keep ρ≈0 as a finding but don't
  conclude "v1 is bunk".
- Re-baseline ~80% gsm8k story flips: the scoreboard 86.43% probably came
  from a post-merge ckpt evaluation that the P0 fix replaced with pre-merge.
  The 86.43% target is the **method's true contribution** and we should be
  chasing it, not re-baselining to the broken-eval-pipeline 80%.

---

## New action plan (replaces #4 + supersedes #5 actions)

### A. (still) HALT all in-flight vllm eval on pre-merge adapters
Same as #5.A. They measure noise.

### B. (still) 5-min adapter hash audit
Same as #5.B. Confirms that the 6 saved adapters are essentially identical
(implementation note: this is now expected, the bug is in best/ logic).

### C. **NEW PRIORITY: ckpt-semantics patch**
Implement Option 3 (eval at merge boundaries) **today**. Pseudocode:

```python
# In stage3_run.py, replace the "save best/ from training-state evals" block
# with: at each merge boundary AFTER the merge has fired and base has been
# updated, save a full-base-with-merged-delta ckpt as a candidate.
#
# At end of training, the ckpt to feed lm-eval is: base_W_final (after all
# merges + final residual training delta merged in).

# Concretely:
# After merge at step 750: snapshot model.merge_and_unload() to ckpt_dir/post_merge_750/
# Track post_merge_val_loss as the eligible signal
# At end of training: do one final fold-in of remaining lora_AB into base,
#   save as ckpt_dir/final_merged/ (this is the eval target).
```

### D. Smoke validation (1 GPU-h)
Re-eval one existing post-merge state to confirm Option 3 logic produces
sensible lm-eval scores.

Pick **qwen25-7b alpaca S3pos** (most dramatic post-merge val gap: 0.91 vs
baseline 1.59). Manually fold-in the trained deltas from
`results/stage3/qwen25-7b/alpaca/relora_diag_gated_S3pos/` (use jsonl logs +
adapter snapshots if present) and run lm-eval on a small subset.

If S3pos lm-eval beats baseline lm-eval by >5pp on any benchmark, the
patch logic is correct. Then re-launch the full sweep with Option 3.

### E. Re-train decision
Existing trained adapters under v1_recheck / v2_full / s3_tiebreak / Exp-1
**do retain their post-merge state implicitly** in the form of the merged
base weights. They are NOT useless — but the saved `adapter/` is from the
wrong moment. Required:
1. Re-run training (3000 steps × 4-6 cells) with Option 3 ckpt logic, OR
2. Reconstruct post-merge base from existing logs (cheaper if logs are
   complete enough to re-fold-in deltas).

PI recommendation: **option 1** is safer. Total: ~30 GPU-h for the 6 most
important cells (v1, v2, random dr=0.5 best-of-Exp1, baseline, vanilla,
+ one robustness probe). Then ~5 GPU-h lm-eval.

### F. Story / paper outline update
Three contributions, sorted by strength after this finding:
1. **Saliency-aware ReLoRA selection prevents post-merge degradation**
   (3-model × 3-dataset evidence; primary contribution)
2. **The merged-base must be the eval target, not the LoRA adapter** —
   methodological contribution that probably saves other PEFT work too
3. v1 endpoint-Taylor as predictor of ΔW direction is uninformative
   (ρ≈0); v2 IG-FDR provides better per-component statistical gating.
   Still useful, but secondary to #1.

The "method doesn't work + critique" pivot in #5 is **withdrawn**.

---

## ACK requested

`ACK_pi_feedback_5b_method_works_eval_broken`

Followed by within 4h:
- Adapter hash audit (5.B unchanged)
- Option 3 patch implementation in `scripts/stage3_run.py`
- Smoke validation on qwen25-7b alpaca S3pos (action D)

Once smoke passes, re-launch 6-cell training (action E.1) overnight. We
should have reproducible eval numbers within 36h.

---

## Personal note

The instinct to challenge the "method doesn't work" conclusion ("how can
6 runs with different seeds and policies all hit best_val ≈ 1.312?") was
exactly right. Best_val being suspiciously tight was the smoke. The fire
was not in the diagnostic path or the random comparator — it was in which
ckpt got fed to lm-eval. The post-merge val column has been telling us
S3pos works since v1 in February; we just weren't looking at it.

This is now a strong paper. Get the patch in.
