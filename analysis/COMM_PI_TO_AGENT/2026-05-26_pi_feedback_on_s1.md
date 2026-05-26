# PI Feedback on S1 + Pre-Launch Checks for S2 / S2.5
**Date**: 2026-05-26 (post-13:00 UTC, after pulling commit `242447d`)
**Replies to**: `analysis/WAKE_UP_INSTRUCTIONS.md` + `fa4534e`/`242447d`
**ACK seen**: `ACK_v2_saliency_revamp` confirmed in `fa4534e` body ✅
**ACK requested for this directive**: `ACK_pi_feedback_s1`

---

## 0. PI verdict on S1

**Excellent execution.** S1 finished in ~75 s and the result is **clean**:

```
rho_global         = 0.0242   (essentially zero)
sign_flip_rate     = 0.4521   (near random)
top10pct_iou_keep  = 0.126
top10pct_iou_drop  = 0.028
n_components       = 4032 (252 layers × r=16)
DECISION           = A_CRITICAL_implement_IG
```

This is strong evidence that v1 saliency does not measure "what helped get
from W₀ to W₀+ΔW", it measures "local landscape at W". Path A in v2 is
mandatory.

**However**, before declaring framing-error confirmed and burning compute on
S2 / S2.5, PI has 6 follow-ups. **Treat 1, 2, 3 as blocking; 4 is a
correctness check; 5 is a free win; 6 is a knob suggestion.**

---

## 1. ⚠ BLOCKING — Cross-model S1 sanity (~5 min on GPU 7)

ρ = 0.024 is **suspiciously clean**. Single-model could be a Qwen3 quirk.
Re-run S1 on **two more (model, adapter) pairs** before trusting the
"A_CRITICAL" decision globally:

```bash
# olmo2-7b
python scripts/run_s1_framing_test.py \
  --base_model /mnt/cpfs/.../allenai/Olmo-2-1124-7B \
  --adapter_dir results/.../olmo2-7b/tulu3-sft/relora_baseline/seed42/adapter \
  --out_path analysis/results_v3/saliency_framing/spearman_olmo2-7b_tulu3.json \
  --dataset tulu3-sft --n_calib 256 --max_len 512

# r1-distill-7b (or any other already-trained adapter)
python scripts/run_s1_framing_test.py \
  --base_model /mnt/cpfs/.../DeepSeek-R1-Distill-Qwen-7B \
  --adapter_dir results/.../r1-distill-7b/.../relora_baseline/seed42/adapter \
  --out_path analysis/results_v3/saliency_framing/spearman_r1-distill_<dataset>.json \
  ...
```

**Pass rule**: ρ < 0.15 on at least 2 of 3 models (qwen3 + 2 others).
**Fail rule**: any single model has ρ > 0.4 → IG hypothesis is model-specific,
do not generalize to a global story; report differential and discuss.

Either outcome is publishable. Just do not claim "estimator framing is
universally wrong" from one model.

---

## 2. ⚠ BLOCKING — Run v2 smoke on GPU 7 NOW

GPU 7 is idle. Smoke command from `WAKE_UP_INSTRUCTIONS.md` is good but
add three observability hooks the original directive didn't ask for:

```bash
nohup env CUDA_VISIBLE_DEVICES=7 \
  /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python \
  scripts/stage3_run.py \
  --model_path /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B \
  --model_key qwen3-8b --dataset tulu3-sft \
  --method relora_diag_gated_S3pos \
  --saliency_estimator v2 --saliency_v2_m_ig 4 --saliency_v2_alpha 0.1 \
  --saliency_calib_n 64 \
  --total_steps 200 --merge_every 100 --eval_every 50 --ckpt_every 9999 \
  --saliency_max_seq_len 512 --attn_implementation sdpa \
  --seed 42 --out_root results/v2_smoke/seed42 \
  > logs/v2_smoke.log 2>&1 &
```

**Required log lines to verify before declaring smoke-pass**:

1. `[v2 estimator m_ig=4 alpha=0.1]` at first merge event
2. Per-event keep/drop/random counts: `n_keep_sig=NN  n_drop_sig=NN  n_random=NN`
   - **Pass condition**: `n_keep_sig + n_drop_sig >= 200` (out of 4032 ~ 5% significant)
   - **Fail condition**: `n_keep_sig + n_drop_sig < 100` (< 2.5%) — alpha too tight,
     or per-sample SNR too low even with IG; **stop and report**, do not launch S2
3. `train_loss` post-merge non-NaN, recovers within 30 steps (matches v1 behaviour)

**If smoke pass-fails on condition 2** (almost-everything-falls-back-to-random):
- try alpha=0.2 (looser FDR, suggested by §6 below)
- if still <100 significant, try `--saliency_v2_m_ig 8` (more IG points)
- if still flat, log it and ping PI before launching S2

---

## 3. ⚠ BLOCKING — S2 v1_recheck cell is reproducibility gate

The S2 plan includes `relora_diag_gated_S3pos_v1_recheck` (qwen3-8b/tulu3,
v1 estimator). **Run it FIRST in S2** (before any v2 cell).

**Pass rule**: vllm eval must reproduce
```
GSM8K-strict ∈ [86.13, 86.73]   (target 86.43, ±0.3pp tolerance)
GSM8K-flex   ∈ [86.66, 87.26]   (target 86.96, ±0.3pp tolerance)
HellaSwag    ∈ [76.97, 77.57]   (target 77.27)
ARC-C        ∈ [69.02, 69.62]   (target 69.32)
```

Numbers from `analysis/scoreboard.md` Wave 1 / Legacy section.

**Fail action**: if v1_recheck is outside any tolerance band, **DO NOT run
v2 cells**. Open a `analysis/COMM_GPU5_2026-05-26_BLOCKER_harness.md` and
investigate (likely culprits: vllm version drift, base model checkpoint
hash, different scoring template). v2 vs v1 comparison is meaningless if
v1 itself isn't reproducible at the harness level.

---

## 4. ⚠ CORRECTNESS — Verify optimizer in Exp-1 reused cells for S2.5

`WAKE_UP_INSTRUCTIONS.md` says S2.5 reuses Exp-1 cells `dr0.25/0.5/0.75`.
But Exp-1 was launched under the **Muon directive era** (commit `5e6056c`,
`d0d5da3`). Verify these cells are AdamW, not Muon.

**Action**: in `analysis/COMM_GPU5_2026-05-26_<...>.md`, please include a
table:

| schedule | source (Exp-1 reuse / new) | optimizer | seed | total_steps | merge_every |
|---|---|---|---|---|---|
| const_0p25 | reused: `logs/exp_drop_rate/dr0.25` | ??? | 42 | 3000 | 500 |
| const_0p5  | reused | ??? | 42 | 3000 | 500 |
| const_0p75 | reused | ??? | 42 | 3000 | 500 |
| anneal_down | new | AdamW | 42 | 3000 | 500 |
| ... | ... | ... | ... | ... | ... |

**If any reused cell has optimizer != AdamW**: re-launch that cell with
AdamW. The whole point of S2.5 is to isolate schedule effect from optimizer
effect; a 2-Muon-1-AdamW mix in the constant-schedule baselines kills the
analysis.

Confirm in the next push commit body: `S2.5_OPTIMIZER_VERIFIED=AdamW_all`
(or list deviations).

---

## 5. 🟢 FREE WIN — Pre-evaluation training-loss analysis on Exp-1

Exp-1 (6 cells, dr ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9}) is at step ~750/3000.
**By the time it finishes (~ end of training, before vllm eval), you can
already produce directional insight just from training loss.**

Generate `analysis/results_v3/exp1_train_loss_analysis.png` with:

1. 6 train-loss curves overlaid (one per drop_rate), x=step, y=loss
2. Vertical bars at merge events (every 500 steps for total_steps=3000 → 6
   events at 500/1000/1500/2000/2500/3000)
3. Annotated metrics per cell:
   - **post-merge loss spike height**: `loss(merge_step+1) - loss(merge_step-1)`,
     averaged across 6 events
   - **recovery half-life**: steps for loss to return within 10% of pre-merge
   - **final converged loss** at step 3000

**What to look for** (and report in 1 paragraph):
- Monotonic dependence of final loss on drop_rate? (linear? U-shape? saddle?)
- Does spike height grow with drop_rate? (yes ⇒ drop is genuinely
  destructive; no ⇒ LoRA reset hides it)
- Does recovery half-life depend on drop_rate? (long recovery ⇒ schedule
  with low-late drop wins; short recovery ⇒ schedule doesn't matter much)

This gives PI **directional schedule preview before vllm eval lands**, so
S2.5 schedule choices can be validated cheaply if final eval lags.

**Cost**: ~5 minutes of plotting + writing once Exp-1 finishes.

---

## 6. 🟡 KNOB SUGGESTION — alpha=0.2 default for v2

S1 data showed `r=16` per layer (4032 components total). With FDR-BH at
alpha=0.1 over 4032 hypotheses **per merge event**, the threshold is harsh.
With per-sample noise + only m=4 IG points, the t-statistic distribution is
heavy-tailed under the null and few components will reject.

**Suggestion**: change `--saliency_v2_alpha` default from 0.1 → **0.2** for
S2 and the v2 smoke. Rationale:
- multiple-comparison count is per-event, not per-run; ~6 events × 4032 ~
  24k tests is the total
- r=16 means even per-layer the search space is small
- we are **gating training**, not making scientific significance claims —
  type-I error here means "wrongly drop a non-helpful component", which is
  bounded by the random-fallback safety net

If alpha=0.2 over-drops (e.g. >70% drop_rate observed), can pull back to
0.15. Document the chosen alpha in every S2 cell config.

---

## 7. Ordering for the next 24h (PI prefers this exact sequence)

```
NOW                    : §1 cross-model S1 (GPU 7, ~5 min × 2 models)
NOW + 0.1h             : §2 v2 smoke on GPU 7 (~1.5 h on 200 steps)
NOW + 1.6h             : check smoke logs against pass conditions in §2
NOW + 1.6h (parallel)  : Exp-1 finishes → §5 train-loss analysis pushed
NOW + 2h               : if §1 + §2 + §3 (recheck cell) all pass →
                         launch S2 (4 v2 cells + 1 v1_recheck cell)
NOW + 2h               : §4 verify optimizer of Exp-1 reused cells
NOW + 8h               : S2 done → push results
NOW + 8h               : launch S2.5 schedule pilot (12 cells, GPUs 1-6
                         freed by Exp-1; GPU 7 freed by S2)
NOW + 24h              : S2.5 done → PI routes to S3 path α/β/γ/δ
```

**Push cadence**: every 4h, even partial. **Blocker file**:
`analysis/COMM_GPU5_2026-05-26_BLOCKER_<topic>.md` for §1/§2/§3 fails.

---

## 8. What NOT to do in next 24h

- Do NOT launch S2.5 before S2 v1_recheck passes (§3)
- Do NOT launch S2 v2 cells before smoke passes (§2)
- Do NOT skip §1 cross-model check by claiming "ρ=0.024 is enough"
- Do NOT mix Muon and AdamW cells in S2.5 (§4)
- Do NOT change the v2 implementation between smoke and S2 unless smoke
  fails — keep the comparison clean

---

## 9. Things PI noticed and is NOT asking you to change (yet)

- **r=16 is lower than I assumed** (I had been thinking r=64). For paper
  Section 4 ablation, please add r ∈ {8, 16, 32, 64} sweep on the
  cherry-picked (model, dataset, method) cell **after S3 routing decision**.
- **Sign convention bug in `src/saliency.py:6` docstring** (`-⟨G,ΔW⟩` vs
  code `+⟨G,ΔW⟩`) — fix during paper-writing pass, not now.
- **252 layers × r=16 ≠ 4032 only if all layers have r=16**: verify a few
  layers in the json have `r=16` (a quick `grep '"r":' | sort | uniq -c`).
  If any layer has r ≠ 16, your IG implementation must handle variable r.

---

## 10. ACK in next push

Commit body must include `ACK_pi_feedback_s1`. If you disagree with any
specific item (1–6), reply in `analysis/COMM_GPU5_2026-05-26_<topic>.md`
with reasoning; PI will respond within 4h.

End of feedback.
