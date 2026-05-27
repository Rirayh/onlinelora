# PI Feedback #4 — Re-baseline Approved + Eval Auth + Pre-Merge-Convergence Finding
**Date**: 2026-05-27 (post-05:30 UTC, after pulling commits `54c9b07` + `afa7880` + `5f53503`)
**Replies to**: Exp-1 eval routed to E + v1_recheck BLOCKER + v2_full DONE + tie-break ALL DONE
**ACK strings confirmed**: `S5_SCHEDULE_SANITY=PASS` ✅ + `S3_ROUTE=E_ambiguous_tiebreak` ✅
**ACK requested for this directive**: `ACK_pi_feedback_4_rebaseline_approved`

---

## 0. PI verdict on overnight progress

**Outstanding execution.** In ~14 hours since `acd3441`:
- §5 schedule sanity PASS (12 events ±5%) ✅
- Exp-1 6 cells eval DONE, routed E correctly ✅
- v1_recheck train+eval DONE (with proper BLOCKER memo) ✅
- v2_full train DONE, 4 events captured with full instrumentation ✅
- v1 Bernoulli rejection >10σ — independent paper finding ✅
- S3 tie-break 4 cells train DONE ✅
- Cross-reference logic on the 6pp scoreboard gap — exemplary scientific
  reasoning, not just "results are off, please advise" ✅

This directive resolves the BLOCKER, authorizes the eval phase, and adds
**one new finding** that PI noticed in `5f53503` that needs investigation
before any S3 path-α/β/γ decision.

---

## 1. 🟢 RE-BASELINE APPROVED (Hybrid path)

PI accepts the re-baseline argument. Reasoning chain:

1. v1_recheck (sound code, sound data) = 80.36/81.43/77.61/66.38
2. Exp-1 dr=0.0 (vanilla LoRA, no merge at all) = 79.15/80.14/77.68/66.21
3. **Two completely independent code paths produce the same ~80% number**
4. Scoreboard 86.43/86.96 is a **single-source data point** with no
   reproducibility from current pipeline
5. Bayesian: probability(scoreboard is right, current pipeline is wrong) ≪
   probability(scoreboard had unknown contamination)

### Action items (autonomous)

#### 1.1 New baseline = v1_recheck values (`80.36 / 81.43 / 77.61 / 66.38`)

Update `analysis/scoreboard.md` in-place with a clearly labeled
**"Re-baseline 2026-05-27 (post-P0-fix, deprecated scoreboard 6pp gap)"**
section. Do **not** delete the old scoreboard rows — annotate them with:
```
DEPRECATED 2026-05-27: produced by pre-P0 pipeline; reproducibility
fails by 6pp on GSM8K with current code. See COMM_PI_v1_recheck_BLOCKER.md.
```

#### 1.2 Forensics (parallel, non-blocking, 1 GPU × ≤8h)

Open `analysis/COMM_GPU5_2026-05-27_forensics.md` and run these checks
on **one GPU only**, in this order, stopping when any positive finding
emerges:

**Check 1 (cheap, 5 min)**: lm_eval JSON args diff
```bash
# Find a scoreboard-era qwen3-8b/tulu3 lm_eval results JSON
find results -path "*scoreboard*" -o -path "*pre_p0*" -name "results_*.json" | head -3
# OR check git log for the commit that wrote the scoreboard 86.43 number
git log --all --diff-filter=A -- analysis/scoreboard.md | head -5
git log --all --pretty=format:"%h %ad %s" --date=short \
  -G"86.43" -- analysis/scoreboard.md | head -10
```

For each pre-P0 results JSON, diff against current Exp-1 dr=0.0 results
JSON for these fields:
- `config.model_args` (vllm version, dtype, max_model_len)
- `config.task_version` (`gsm8k.0`, `gsm8k.1`, etc.)
- `config.gen_kwargs` (max_new_tokens, do_sample, temperature)
- `config.fewshot_split`, `config.num_fewshot`
- prompt template (look at `samples_*.jsonl` first 5 prompts)

**Output**: `analysis/results_v3/forensics_lm_eval_diff.tsv` with one row
per differing field.

**Check 2 (medium, 1h)**: replay scoreboard config on current adapter

If Check 1 found `gen_kwargs` or `task_version` differences, replay v1_recheck
eval with the OLD config:

```bash
# Hypothetical: if scoreboard used max_new_tokens=512 but we use 256
lm_eval --model vllm --model_args ... \
  --tasks gsm8k --gen_kwargs "max_new_tokens=512,do_sample=False" \
  --output_path results/forensics/v1_recheck_old_geneval/
```

If old gen_kwargs reproduces 86.43 on v1_recheck adapter → **scoreboard is
right, our current eval config is wrong**. Update default eval config to
match scoreboard era. Re-baseline reverts.

If old gen_kwargs still produces 80.36 → adapter is the difference, not
eval. Move to Check 3.

**Check 3 (expensive, 4h)**: pre-P0 commit replay

Last resort: checkout `92280b28d` (pre-P0 commit), train fresh
qwen3-8b/tulu3 relora_baseline cell, eval with current pipeline.

If pre-P0 code+current eval gives 86.43 → P0 fix removed something useful,
need root-cause diagnosis. **Critical finding** — escalate immediately.

If pre-P0 code+current eval gives 80% → confirms re-baseline 100%, P0 fix
was correct, scoreboard contamination origin remains unknown but is not
in current code path.

#### 1.3 Forensics SLA

- Check 1 must be in next 4h push
- Check 2 only if Check 1 finds candidate differences
- Check 3 only on PI green-light (do not auto-launch the 4-hour replay)

---

## 2. 🟢 v2 §1 STRICT CRITERION ACCEPTED AS HEALTHY

PI accepts your argument that the strict `event2 ≥ event1` criterion was
too tight. The full v2_full trajectory:

```
sig_frac:  0.406 → 0.329 → 0.274 → 0.420   (V-shape, event 4 recovers)
spread/|q50|:  22× → 41× → 17.4× → 18.8×  (all > 5× threshold)
q95 magnitude: 9.4e-5 → 1.2e-4 → 4.4e-4 → 3.9e-4  (4× growth)
post-merge val: 1.327 → 1.324 → 1.338 → 1.350
```

V-shape with event 4 recovery + sustained spread/|q50| signal + q95
4× growth = **estimator is healthy**. PI #2 §1 criterion superseded.

### Action: AUTHORIZE v2_full vllm eval IMMEDIATELY

GPUs 0-7 are all idle per `5f53503`. Launch vllm eval on best ckpt
(step 500) NOW:

```bash
# v2_full (1 GPU)
nohup env CUDA_VISIBLE_DEVICES=0 \
  bash scripts/lm_eval_merged_4tasks.sh \
  results/s2/qwen3-8b/tulu3-sft/v2_full/seed42 \
  > logs/eval_v2_full.log 2>&1 &
```

Expected runtime: ~30-45 min on H100.

---

## 3. 🟢 AUTHORIZE 4× tie-break vllm eval IN PARALLEL

Tie-break 4 cells (dr ∈ {0.05, 0.15, 0.2, 0.3}) train DONE, best ckpt at
step 250-500. Eval in parallel on GPUs 1-4:

```bash
for dr in 0.05 0.15 0.2 0.3; do
  CUDA_VISIBLE_DEVICES=$((i++)) bash scripts/lm_eval_merged_4tasks.sh \
    results/s3_tiebreak/qwen3-8b/tulu3-sft/dr${dr}/seed42 \
    > logs/eval_tiebreak_dr${dr}.log 2>&1 &
done
```

5 parallel jobs (1 v2_full + 4 tie-break) → all done in ~45 min.

---

## 4. 🚨 NEW FINDING — best ckpt at step 250-500 across all cells

PI noticed in `5f53503` that **all 4 tie-break cells + v1_recheck +
v2_full have best_val_loss at step 250-500**, which is:

- **before the first merge event (step 750)** for v2_full (merge_every=750)
- right around the first merge event (step 500, merge_every=500) for
  Exp-1 / v1_recheck (need to verify merge_every for those cells)

**Implication if true**: post-merge training is pure overfitting. Merge
operations are not contributing to validation improvement. This challenges
the entire ReLoRA framework on this dataset+model.

### Action: SANITY CHECK across all completed cells

Before any S3 path decision, run this analysis (PI estimates 30 min):

```bash
# extract best_step + best_val from every completed cell
for cell_dir in results/{exp_drop_rate,s3_tiebreak,s2}/qwen3-8b/tulu3-sft/*/seed42; do
  echo "$cell_dir"
  grep -E "best_val|best_step|merge_every|total_steps" \
    $cell_dir/{summary.json,config.json,trainer_state.json} 2>/dev/null
done
```

Output to `analysis/results_v3/best_ckpt_step_audit.tsv` with columns:
`cell, total_steps, merge_every, best_step, best_val, n_merge_events_passed_before_best`

### Interpretation gate

| pattern | implication |
|---|---|
| **all cells** best_step < merge_every (no merge passed before best) | merge framework not helping; paper section becomes "ReLoRA collapses to early-stopped vanilla LoRA on tulu3-sft" — major negative finding |
| **most cells** best_step at merge boundaries | merges occasionally help, schedule-dependent |
| **mixed** based on dr | dr is the real story (some dr push best later) |

If **all cells** pattern: this is a paper-altering finding. PI wants to
hear it before launching S2.5 12-cell schedule sweep — the entire premise
of schedule-over-merge-events would be moot.

### Cross-check eval should be best vs final ckpt

When you eval v2_full + 4 tie-break + Exp-1 cells, also produce **best
ckpt eval AND final ckpt eval** for at least 2 cells (e.g. v2_full and
dr=0.5). If best-ckpt-eval ≫ final-ckpt-eval, confirms overfitting story.

```bash
# Same lm_eval call but pointing at adapter dir vs best dir
lm_eval --model vllm --model_args ...,peft=$cell_dir/adapter ...   # final
lm_eval --model vllm --model_args ...,peft=$cell_dir/best    ...   # best
```

---

## 5. 🟡 v1 MONOTONIC DRIFT — note as paper finding, no follow-up compute

v1_recheck event-by-event drop_rate `0.561 → 0.611 → 0.598 → 0.670` with
each event >10σ off Bernoulli null is a clean publishable finding.

### Action: write a 1-page memo, no new experiments

`analysis/COMM_GPU5_2026-05-27_v1_drift_finding.md`:
- table of 4 events with z-score vs H0=Bernoulli(0.5)
- 1-paragraph mechanistic explanation: why does sign(grad·A) drift to
  predominantly positive at endpoint W as ΔW accumulates?
  - hypothesis: as W approaches a (local) minimum, ⟨G, ΔW⟩ → 0+, but
    components that overshot the minimum produce ⟨G, ΔWᵢ⟩ > 0
    (i.e. "drop-recommended" by v1)
  - so v1 increasingly identifies "overshoot" components, not
    "harmful" components — different physical quantity
- mock-up bar chart for paper Section 4

**No new compute**. Multi-seed reproducibility check defers to
path-α/β/γ phase.

---

## 6. 🟡 IoU ANALYSIS — weak version OK

PI accepts that strict v1↔v2 IoU cannot be done (v1_recheck didn't log
component IDs). Two acceptable substitutes:

### Option A (cheap, prefer this)

200-step v1 mini-cell with new logging on GPU 5 (idle), compare to
v2_full's first 200 steps' selection:

```bash
# 200 steps = before any merge events for typical merge_every=500/750
# but force merge_every=100 so we get 2 merge events in 200 steps
nohup env CUDA_VISIBLE_DEVICES=5 \
  python scripts/stage3_run.py \
  --method relora_diag_gated_S3pos \
  --saliency_estimator v1 \
  --total_steps 200 --merge_every 100 \
  ...
  --out_root results/v1_mini/qwen3-8b/tulu3-sft/seed42 \
  > logs/v1_mini.log 2>&1 &
```

Then compute IoU between v1_mini events 0,1 and v2_full events 0,1
per-layer-type. ~1 GPU-hour cost.

### Option B (free, weaker)

v2_full per-event keep-count distribution per layer-type as a
descriptive figure. No comparison, just shows which layer-types v2 drops
heaviest.

**Pick A unless GPU pressure**, write IoU per (event, layer_type) per
PI #3 §3 spec.

---

## 7. ⚠️ TIE-BREAK SINGLE-SEED CAVEAT — to be flagged in any conclusion

Exp-1 dr=0.5 (80.74) vs dr=0.1 (81.43) is 0.69pp gap; tie-break 4 cells
also single-seed=42. **All conclusions from tie-break alone must be
flagged "single-seed, ≥1.5pp required for non-noise claim"**.

### Action: in tie-break eval summary doc

Include a **noise floor estimate** paragraph:

> "Tie-break results are single seed=42. Based on the Exp-1 spread
> across dr={0.0, 0.1, 0.25, 0.5, 0.75} excluding the dr=0.9 outlier
> (which is 3pp below the rest), GSM8K-flex variance ≈ ±0.7pp. Any
> tie-break result within 1.5pp of another is not statistically
> distinguishable at single-seed. Multi-seed CI is required for path-γ
> final figures and is deferred to S3 path-γ stage."

If the 4 tie-break + dr=0.0/0.1 are all within 1.5pp of each other →
**there is no clean peak**, and the schedule pilot becomes the only way
to find signal (different mean drop rates over training, not different
constant drop rates).

---

## 8. UPDATED 24h ORDERING

```
NOW              : ACK this directive in next push commit body
NOW              : §2 + §3 launch 5 parallel vllm evals (GPUs 0-4)
                   → done ~06:15 UTC
NOW (parallel)   : §4 best_ckpt_step_audit.tsv from existing cells (cheap)
NOW (parallel)   : §1.2 Check 1 forensics (lm_eval JSON diff, 5 min)
NOW (parallel)   : §6 Option A v1_mini launch on GPU 5 (1h)
~06:15 UTC       : 5 evals done → analysis push:
                   - exp1_eval_vs_droprate_v2.png (with v2 + tie-break overlaid)
                   - best vs final ckpt eval comparison for 2 cells (§4)
                   - v1↔v2 IoU per layer-type (§6 Option A)
~06:30 UTC       : §1.2 Check 1 forensics findings
~07:00 UTC       : v1 monotonic drift memo (§5)
~07:00 UTC       : PI inspects results, decides:
                   - if best_ckpt_step_audit shows "all cells before
                     first merge" → CRITICAL pause, debug merge framework
                   - if v2 eval > 81.43 → path-α saliency-saved, full sweep
                   - if v2 eval ≤ 81.43 + tie-break flat → path-γ schedule
                   - if v2 eval ≤ 81.43 + tie-break has peak → path-γ
                     with const at peak as anchor
```

**Push cadence**: this directive doesn't change cadence. Next push due
~09:30 UTC, ideally after eval landing.

---

## 9. Things PI is NOT asking for

- Multi-seed yet — still deferred until S3 path commit
- Path-δ (Muon) — frozen
- Other models / datasets — qwen3-8b/tulu3 only
- alpha sweep on v2 — locked at 0.2

---

## 10. Open scientific question (no action required)

The combination of two independent findings is interesting:
- v1 drop_rate drifts up monotonically with merge events (§5)
- best ckpt at step 250-500 for all cells, before/at first merge (§4)

If both hold, the paper narrative might shift to:
> "First-order saliency produces non-random selection on tulu3-sft, but
> the selection drifts toward 'overshoot' components rather than
> 'harmful' components. Combined with empirical evidence that merges
> after step 500 do not improve validation, this suggests the merge
> framework on this dataset is operating in an overfitting regime
> where saliency is selecting against the very components that would
> have helped on a longer-horizon distribution."

This is a **path-γ-with-mechanistic-twist** narrative. Don't write the
paper section yet — but worth keeping in mind when interpreting eval
results.

---

## 11. ACK in next push

Commit body must include `ACK_pi_feedback_4_rebaseline_approved`. Eval
results pushed in same window. Disagreements via
`analysis/COMM_GPU5_2026-05-27_<topic>.md` reply file.

End of feedback.
