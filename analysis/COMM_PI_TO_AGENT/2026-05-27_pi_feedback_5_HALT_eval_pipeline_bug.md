# PI Feedback #5 — STOP. Eval pipeline is fundamentally broken.

**Date**: 2026-05-27
**Severity**: P0 / SHOWSTOPPER
**Action requested**: HALT all eval immediately. Re-design before any further compute.
**Supersedes**: 2026-05-27_pi_feedback_rebaseline_eval_auth.md (the 5-cell eval auth in #4 is RETRACTED)

---

## TL;DR

The "best" adapter that we feed to lm-eval is a **pre-merge LoRA checkpoint
saved before any drop policy has fired**. Because drop only happens at merge
events (`step ∈ {750, 1500, 2250, 3000}`), and `best_step ∈ {250, 500}` for
**all 6 runs we have on disk**, every adapter we have evaluated so far is
effectively the **same vanilla-LoRA pre-merge state plus cuDNN reduction noise**.

This explains every "weird" finding of the past two weeks. It is not "method
doesn't work". It is "we have not been measuring the method at all".

---

## Hard evidence

### 1. All 6 trained runs share identical (seed, model, dataset, schedule)
```
config.yaml across {dr0.05, dr0.15, dr0.2, dr0.3, v1_recheck, v2_full}:
  model_path  : /mnt/cpfs/.../Qwen3-8B
  dataset     : tulu3-sft
  total_steps : 3000
  merge_every : 750
  eval_every  : 250
  seed        : 42
```

### 2. summary.json best_step distribution
| run | first_eval | best_val | best_step | merge_every |
|---|---|---|---|---|
| dr=0.05      | 1.31408 | 1.31215 | 500 | 750 |
| dr=0.15      | 1.31536 | 1.31255 | 500 | 750 |
| dr=0.20      | 1.31462 | 1.31339 | 500 | 750 |
| dr=0.30      | **1.31316** | **1.31316** | **250** | 750 |
| v1_recheck   | 1.31591 | 1.31317 | 500 | 750 |
| v2_full      | 1.31186 | 1.31141 | 500 | 750 |

`dr=0.30` has `first_eval == best_val` to 13 decimal places.
**All `best_step ∈ {250, 500}` — i.e. before first merge at step 750.**

### 3. Drop policy is ONLY active at merge events
`scripts/stage3_run.py:1106-1132` (random_drop) and `:1133+` (gated):
```python
if step in merge_steps:
    # build keep_mask via gate_sign / saliency_estimator / random_drop_rate
    ...
```
None of `gate_sign`, `saliency_estimator`, `target_drop_rate` participates in
the optimizer step or forward pass between merges. Therefore for `step ∈ [0, 749]`,
the training loop is **identical bit-for-bit (modulo cuDNN reduction order)
across all 6 runs**.

### 4. The "best" we save is exactly that pre-merge state
`scripts/stage3_run.py:1062-1078`:
```python
if step % args.eval_every == 0:
    vl = evaluate_lm(model, val_loader, device, ...)
    if vl < best_val_loss:
        best_val_loss = vl
        model.save_pretrained(str(best_dir))
```
`scripts/stage3_run.py:1351`:
```
# NOTE: do NOT update best ckpt from post-merge val_loss.
```
`scripts/stage3_run.py:1413-1428` (the **P0 FIX**):
```python
if do_relora and _best_dir_final.exists() and ...:
    shutil.copytree(str(_best_dir_final), str(adapter_dir))
    log.info("adapter saved (copied from best/ to avoid post-merge B=0)")
```

### 5. Therefore the lm-eval numbers we have been quoting...
```
              gsm8k   arc_c   hellaswag
S3pos         87.95%  66.13%  76.09%
S3neg         86.88%  67.15%  77.82%
random_drop   86.43%  67.24%  77.14%
```
...are scores of (essentially) **the same vanilla LoRA pre-merge adapter run
3 times**. The ~1.5pp spread between S3pos / S3neg / random is well within
GPU non-determinism + lm-eval batch-order noise for an LLM at this scale.

There is **no signal of method here**. None.

### 6. Why every prior "weird" observation falls into place
| Symptom | True cause |
|---|---|
| 6 cells best_val ≈ 1.312 (±0.002) | Same pre-merge ckpt evaluated 6 times |
| random ≥ diagnosis on eval | Both ckpts identical; gap = noise floor |
| v1 monotonic drift 0.56→0.67 in dropped fraction | Real, but only affects post-merge state — invisible to best/adapter |
| v2 V-shape sig_frac 0.406→0.329→0.274→0.420 | Real, but invisible for the same reason |
| v1_recheck reproducibility gate fails -6pp vs scoreboard | We are evaluating a ckpt that has never seen any merge; scoreboard target was probably on a post-merge or final state |
| best_step universally < first_merge | The smoking gun: best/ never sees method active |

---

## Immediate actions (PI directive, no ack required)

### A. HALT: kill any in-flight vllm eval on GPUs 0-4
The 5-cell eval authorized in feedback #4 is retracted. It would burn ~5
GPU-h producing 5 noise-floor measurements of the same vanilla LoRA ckpt.

If eval has already started, kill it. If not, do not start it.

### B. PROVE the bug with a 5-minute audit
```bash
# Bit-level adapter comparison across 6 runs
python - <<'PY'
import hashlib, json, pathlib
runs = {
    "dr0.05":   "results/s3_tiebreak/qwen3-8b/tulu3-sft/dr0.05/seed42",
    "dr0.15":   "results/s3_tiebreak/qwen3-8b/tulu3-sft/dr0.15/seed42",
    "dr0.20":   "results/s3_tiebreak/qwen3-8b/tulu3-sft/dr0.2/seed42",
    "dr0.30":   "results/s3_tiebreak/qwen3-8b/tulu3-sft/dr0.3/seed42",
    "v1":       "results/s2_v1_recheck/qwen3-8b/tulu3-sft/relora_diag_gated_S3pos_v1_recheck/seed42",
    "v2_full":  "results/s2/qwen3-8b/tulu3-sft/v2_full/seed42",
}
for name, d in runs.items():
    p = pathlib.Path(d) / "adapter" / "adapter_model.safetensors"
    if not p.exists():
        p = pathlib.Path(d) / "ckpt" / "best" / "adapter_model.safetensors"
    if not p.exists():
        print(name, "MISSING"); continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f"{name:10s}  {h}  {p}")
PY
```

Expected outcome: hashes are either identical, or differ only in low-order
bits attributable to cuDNN. If even one pair is identical, the bug is
proven directly. Save the output to `analysis/audits/2026-05-27_adapter_hash_audit.txt`
and commit before any other step.

Also: dump per-tensor max-abs diff between v1 / v2_full / dr0.05 best
adapters. If max-abs-diff < 1e-3 across all tensors → confirmed identical
modulo CUDA noise.

### C. Decide the right "best" semantics

The **only meaningful state for ReLoRA** is the **post-merge accumulated base
weight** (i.e. `W_base + sum_k merged_delta_k`). lora_B=0 after merge by
design — that is the entire point of ReLoRA. The current code papers over
this by saving a pre-merge LoRA snapshot, which **defeats the entire merge
mechanism for evaluation purposes**.

Three options for next round of training+eval:

**Option 1 — eval the merged base directly (correct, expensive)**
After each merge event k (and at end of training), save
`W_base + sum_{j≤k} merged_delta_j` as a full base ckpt. Track best_val on
these post-merge states. Eval feeds the merged base directly to lm-eval
(no LoRA at eval time). This is the "real" ReLoRA semantic.

**Option 2 — fold lora_AB into base before save**
At each `eval_every` boundary (post-merge included), fold current `lora_A @ lora_B`
into a copy of the base, save that copy as the eligible best ckpt. Avoids
lora_B=0 issue while keeping LoRA training cheap.

**Option 3 — eval at merge boundaries + final, no "best" tracking**
Eval at step ∈ {750, 1500, 2250, 3000} only. Take max of those four scores
as the cell's score. Removes the entire "best/" P0-fix hack. Simplest;
matches what ReLoRA paper actually does.

PI recommendation: **Option 3 first** (cheapest, removes P0 hack), then
**Option 2** if Option 3 still has issues. Option 1 is correct but costs
4× more disk per cell.

### D. Re-train decision
- v1_recheck, v2_full, 4× tie-break adapters on disk are **all useless for eval**.
- They still contain valid `dropped_components.jsonl` / `saliency_*.jsonl`
  → keep them for estimator analysis (V-shape, sig_frac, ρ).
- For Exp-1 and S3 path decision: must re-train under fixed eval semantics
  (Option 3 by default).

### E. Forensics: check the original PI scoreboard (gsm8k 86.43%)
The scoreboard gsm8k_strict 86.43% / gsm8k_flex 86.96% / hellaswag 77.27%
/ arc_c 69.32% — what was the eval ckpt provenance? Was it pre-merge or
post-merge? If post-merge (Option 1 or Option 2 in spirit), then re-baseline
~80% is wrong; the scoreboard targets are correct and we have to re-train
under the correct ckpt semantics to chase them.

This forensics question is now blocking. Please trace which commit / orchestrator
produced the scoreboard JSON.

---

## Updated milestones

| When | Owner | Output |
|---|---|---|
| now (immediate) | agent | kill any running eval; commit "HALT" status |
| now+30min | agent | adapter hash audit (action B) → `analysis/audits/2026-05-27_adapter_hash_audit.txt` |
| now+30min | agent | trace scoreboard ckpt provenance (action E) |
| now+2h | agent | code patch implementing Option 3 ckpt semantics |
| now+2h | PI | review patch before any retrain |
| now+10h | agent | one (1) retrain with new semantics — `random_drop @ dr=0.5` as smoke |
| now+11h | agent | smoke eval; ckpts must show non-noise variance vs vanilla |

Only after the smoke shows >2pp signal-to-noise on at least one
benchmark do we re-launch the full sweep.

---

## What stays valid from prior work
- v1 saliency ρ≈0 cross-model: independent of eval pipeline → still valid
  paper finding.
- v2 IG-FDR estimator V-shape and spread/|q50| > 17×: independent of eval
  pipeline → still valid statistical finding.
- v1 monotonic drift 0.56→0.67 across events: independent of eval pipeline
  → still valid paper finding (Bernoulli null rejected at >10σ per event).
- §5 schedule sanity 12 events ±5%: independent → valid.
- Effective rank curves, condition number trajectories: depend on training
  state at each rank_stat_every, not on best ckpt → valid.

What is **invalidated**:
- Every lm-eval scoreboard number that came from a `relora_*` run with the
  P0-fix copy-from-best path. This includes all S3pos/S3neg/random eval
  rows, all Exp-1 sweep eval rows, v1_recheck eval row.

---

## ACK requested

`ACK_pi_feedback_5_eval_pipeline_HALT`

Followed by within 60 minutes:
- `analysis/audits/2026-05-27_adapter_hash_audit.txt` committed
- ckpt provenance trace for scoreboard 86.43% / 86.96% / 77.27% / 69.32%
- statement of which Option (1 / 2 / 3) you propose for the patch

Do **not** start any retrain or eval before PI sign-off on the patch.

---

## How this happened (post-mortem)

The P0 fix in L1413-1428 was added to work around a real bug (final-state
lora_B=0 → empty adapter → eval = base score). The fix prevented the
0-score symptom but introduced a silent semantic change: the saved adapter
is no longer the trained method's final state, it is a snapshot from before
the method was active. We then spent two weeks A/B testing snapshots that
were essentially the same ckpt.

Lesson: any "P0 fix" that bypasses a numerical artifact by selecting a
different state must be challenged — what state is now being evaluated? Is
that state representative of the method? In hindsight, the fix should have
been "fold lora_AB into base then save" (Option 2) not "save the pre-merge
LoRA snapshot".

---

## Personal note from PI
This is a hard finding to swallow but it is also a clean one. We now know
exactly why every signal we measured was flat. Get the patch in, run one
smoke, and we will know within 12 hours whether the underlying method has
real signal or not. If not, the negative-result paper plus the v1-saliency-is-bunk
critique still stands as a publishable contribution.
