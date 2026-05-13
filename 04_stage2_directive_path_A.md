# Stage 2 Directive — Path A (binding)

> **READ THIS BEFORE TOUCHING THE REPO.** This file overrides the relevant subsections of `03_handover_for_gpu_agent.md` (§3.8 decision, §4.3 default gating criterion, §4.5 success rule, §5 ablation table). All other rules in the handover remain in force.
>
> **Owner**: human PI. **Status**: APPROVED. **Date**: 2026-05-12.

---

## 0. TL;DR

Stage 1 ran clean. Verdict was AMBIGUOUS under the conjunctive (FO **AND** Fisher) gate. The PI has decided to **proceed under Path A**: Stage 2 advances with **first-order val saliency (`S3_fo_val`) as the primary gating signal**; **Fisher (`S5_fisher_val`) is demoted to ablation**. Narrative pivots from "val-Hessian-gated" to "**val-diagnostic-gated**" (first-order primary, second-order ablation).

You do **not** need to ask further questions. Proceed to Stage 2 Phase A immediately on free GPUs. Keep STATUS.md append-only; record the decision entry below verbatim before launching anything.

---

## 1. Stage 1 evidence supporting Path A

From `results/stage1/summary/decision.json` and `correlation_aggregate.json` (already produced by the cloud agent):

| Signal | Mean Δ\|ρ\| (val − train) | CI95 | Sign test | AUC (latest ckpt) | Verdict |
|---|---|---|---|---|---|
| **First-order (S3 − S2)** | **+0.246** | 0.120 .. 0.373 | 11 / 15 positive | mrpc **0.776**, rte **0.759**, sst2 0.587 | **STRONG** |
| Fisher (S5 − S4) | +0.046 | 0.008 .. 0.088 | 10 / 15 positive | (not headline) | weak / threshold-failing |

Interpretation:
- The val/train predictive-validity gap is **large and robust in first-order**: the headline figure `plots/stage1/fig3_train_vs_val_paired.png` carries the paper.
- Fisher tracks the same direction (positive Δρ, 10/15 sign test) but is variance-dominated on large-data / mild-overfit regimes (SST-2). With 256 samples and r=8, the per-component squared-grad estimate has too much noise to clear the 0.10 threshold.
- The harmful-detection AUC ≥ 0.65 is achieved by S3 on two of three tasks — meaning **S3 can actually rank which components are harmful**, which is exactly the predicate Stage 2's gate requires.

Conclusion: the val-side advantage is real, it lives in the first-order channel, and Stage 2's diagnostic gate should be **first-order val**. We do not abandon Fisher; we keep it as an ablation column to show the gate is not signal-source-specific.

---

## 2. STATUS.md append-only entry — paste verbatim before launching Stage 2

```
## 2026-05-12 — Stage 1 → Stage 2 decision: PATH A (approved by PI)

Stage 1 verdict was AMBIGUOUS under the v1 §3.8 conjunctive (FO AND Fisher) gate.
PI authorized Path A: proceed to Stage 2 with first-order val saliency as the
primary gating signal; Fisher demoted to ablation.

Binding changes (overrides 03_handover §3.8 / §4.3 / §4.5 / §5):
1. Default gate signal = S3_fo_val_signed (sign-preserving first-order val).
   Gate criterion = "drop / do-not-merge component i if s_i >= 0", where
       s_i = <grad_A^val L [i,:], A [i,:]>
   This is the directional form of the FO score: positive means merging would
   push val loss UP, so we drop; negative means merging would push val loss
   DOWN, so we keep+merge. (Sign convention: matches handover §3.4 / saliency.py.)
2. Stage 2 `relora_diag_gated` arm uses the rule above (was: S5_fisher_val > 0).
3. New Stage 2 ablation arm `relora_diag_gated_fisher` = same pipeline but gate
   on S5_fisher_val (variance-controlled at a higher max_samples). This is a
   methodology check, not a baseline.
4. v2 framing word-replace: "val-Hessian-gated" → "val-diagnostic-gated"
   throughout method section. First-order is the primary signal; Fisher is the
   robustness ablation. Update the §0 sentence accordingly when writing up.
5. Stage 2 GO criterion (was §4.5): relora_diag_gated must beat relora_baseline
   on val loss AND show effective-rank curve that is not strictly decreasing,
   on AT LEAST ONE of the three model sizes. Phase A (11M) alone is sufficient
   for a GO/NO-GO call if the gap is ≥ 5% relative on val loss.
6. Hard rules from handover §9 remain unchanged. In particular: rule 9 (EPI is
   concurrent, not a baseline) still holds; EPI design ideas remain inspiration
   only.

Launching Stage 2 Phase A on next free GPUs. See 04_stage2_directive_path_A.md.
```

---

## 3. What changes in Stage 2 (delta vs 03_handover §4)

### 3.1 Default gating function — `scripts/stage2_run.py`

CLI now exposes a `--gate_signal` flag. Default is `S3_fo_val_signed`.

```python
# scripts/stage2_run.py (skeleton excerpt — keep your existing structure)
import argparse

GATE_CHOICES = {
    "S3_fo_val_signed":  "first-order val, signed (PRIMARY — Path A default)",
    "S5_fisher_val":     "Fisher val (ABLATION — variance-controlled)",
    "none":              "no gating — i.e. vanilla ReLoRA (this is relora_baseline)",
}

p = argparse.ArgumentParser()
p.add_argument("--size", choices=["11M", "33M", "66M"], required=True)
p.add_argument("--method", choices=[
    "full_rank",
    "relora_baseline",
    "relora_diag_gated",          # uses --gate_signal
    "relora_diag_gated_fisher",   # forces gate_signal = S5_fisher_val
    "relora_signed",              # uses S3_fo_val_signed for revert (see §4.3 of 03_handover, line 628)
], required=True)
p.add_argument("--gate_signal", choices=list(GATE_CHOICES.keys()),
               default="S3_fo_val_signed",
               help="Saliency function used to gate component merging at each ReLoRA event.")
p.add_argument("--fisher_max_samples", type=int, default=512,
               help="When gate_signal=S5_fisher_val, use at least this many val samples to control variance.")
p.add_argument("--saliency_batches", type=int, default=16,
               help="Batches of val diagnostic loader used to estimate first-order saliency at each merge.")
# ...
args = p.parse_args()

# Resolve effective gate signal per method
if args.method == "relora_baseline" or args.method == "full_rank":
    effective_gate = "none"
elif args.method == "relora_diag_gated_fisher":
    effective_gate = "S5_fisher_val"
elif args.method == "relora_signed":
    effective_gate = "S3_fo_val_signed"   # signed-revert variant
else:  # relora_diag_gated
    effective_gate = args.gate_signal
```

### 3.2 Per-merge gating predicate

At each ReLoRA merge event (every 5000 steps, per handover §4.2):

```python
def keep_component_for_merge(s_i: float, gate_signal: str) -> bool:
    """Return True iff rank-1 component i should be merged into base.

    Sign convention (handover §3.4): s_i^FO_signed = <grad_A^val L[i,:], A[i,:]>.
    s_i > 0  =>  merging pushes val loss UP  => DROP (return False)
    s_i < 0  =>  merging pushes val loss DOWN => KEEP+MERGE (return True)
    For Fisher (squared, always >= 0), use a threshold; positive variance-floor.
    """
    if gate_signal == "S3_fo_val_signed":
        return s_i < 0.0
    elif gate_signal == "S5_fisher_val":
        # Fisher is always >= 0; "informative" means above noise floor.
        # Use per-layer median as threshold (within-layer normalization, EPI §1.7 hint).
        # See utility `fisher_layer_threshold` below.
        return s_i > fisher_layer_threshold
    elif gate_signal == "none":
        return True   # vanilla ReLoRA: merge everything
    else:
        raise ValueError(gate_signal)
```

**Per-layer threshold for Fisher** (variance-controlled ablation arm only):
- Compute `S5_fisher_val` for all rank-1 components in a layer.
- Threshold = median of within-layer scores.
- This implements EPI's "within-layer normalization" heuristic (v2 §1.7 lesson 3); it is also independent evidence that the gate works even when the signal is noisier.

### 3.3 Logging at merge events

Each merge event must write one record into `results/stage2/<size>/<run>/saliency_at_merge.jsonl`:

```json
{
  "step": 5000,
  "merge_event": 1,
  "gate_signal": "S3_fo_val_signed",
  "components_total": 192,
  "components_kept": 71,
  "components_dropped": 121,
  "drop_rate": 0.630,
  "per_layer_keep_counts": {"layer.0.q": 4, "layer.0.v": 3, "...": "..."},
  "score_quantiles": [-0.182, -0.041, -0.008, 0.012, 0.117]
}
```

This file is the ablation evidence: if `relora_diag_gated` and `relora_diag_gated_fisher` drop materially different component sets but **both** improve over `relora_baseline`, the gate works — and it's not just a property of the FO signal.

### 3.4 Decision rule update (overrides §4.5)

**Method works (Stage 2 → Stage 3 GO)** if **all** of:
1. `relora_baseline` reproduces the Weiss failure on ≥ 1 size (effective rank non-monotone OR val loss worse than `full_rank`).
2. `relora_diag_gated` beats `relora_baseline` on val loss by ≥ **5% relative** on ≥ 1 size.
3. Either: (a) `relora_diag_gated_fisher` also beats `relora_baseline` (≥ 2% relative) — strong evidence the gate is signal-agnostic; or (b) clearly does not — in which case write up FO as the headline gate and ship.

**Method does not work (NO-GO)** if:
- `relora_diag_gated` ties `relora_baseline` on val loss on all three sizes (within 1% relative),
- OR effective-rank curves are visually indistinguishable.

Phase A (11M alone) is sufficient to make the call if the val-loss gap is ≥ 5% relative.

---

## 4. Parallel launch templates (80GB A100 node)

> Host has 8× A100-80G. Run experiments in parallel; do not single-stream. Conda binary at `/mnt/cpfs/junlongke/miniconda3/bin/conda`. **Reuse `espo` env; do NOT mutate it.** All Python paths route through `/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python`.

### 4.1 Phase A: 11M × 4 methods on 4 GPUs (parallel)

```bash
export PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
export HF_HOME=/mnt/cpfs/junlongke/hf_cache
cd /mnt/cpfs/junlongke/onlinelora/lora_obd
mkdir -p logs results/stage2/11M

# Check GPU map (do this every time)
nvidia-smi --query-gpu=index,memory.free --format=csv

# Launch 4 methods in parallel on free GPUs (adjust GPU IDs to what's free)
CUDA_VISIBLE_DEVICES=0 $PY scripts/stage2_run.py --size 11M --method full_rank                          > logs/s2_11M_full.log         2>&1 &
echo $! >> .stage2_pids
CUDA_VISIBLE_DEVICES=1 $PY scripts/stage2_run.py --size 11M --method relora_baseline                    > logs/s2_11M_relo.log         2>&1 &
echo $! >> .stage2_pids
CUDA_VISIBLE_DEVICES=3 $PY scripts/stage2_run.py --size 11M --method relora_diag_gated                  > logs/s2_11M_diag.log         2>&1 &
echo $! >> .stage2_pids
CUDA_VISIBLE_DEVICES=4 $PY scripts/stage2_run.py --size 11M --method relora_diag_gated_fisher --fisher_max_samples 512  > logs/s2_11M_diag_fisher.log  2>&1 &
echo $! >> .stage2_pids
wait

# Optional 5th method on a 5th free GPU (11M only — signed-revert variant):
# CUDA_VISIBLE_DEVICES=5 $PY scripts/stage2_run.py --size 11M --method relora_signed > logs/s2_11M_signed.log 2>&1 &
```

Read GPU map first; the snapshot in STATUS.md said GPUs 0,1,3,4,5,6,7 are free, GPU 2 belongs to another user — **leave GPU 2 alone**.

### 4.2 Phase B: 33M and 66M × 3 methods (drop `full_rank` for 66M if compute pressure)

Fan out across 6 GPUs:

```bash
# 33M
CUDA_VISIBLE_DEVICES=0 $PY scripts/stage2_run.py --size 33M --method full_rank          > logs/s2_33M_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/stage2_run.py --size 33M --method relora_baseline    > logs/s2_33M_relo.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY scripts/stage2_run.py --size 33M --method relora_diag_gated  > logs/s2_33M_diag.log 2>&1 &
# 66M
CUDA_VISIBLE_DEVICES=4 $PY scripts/stage2_run.py --size 66M --method full_rank          > logs/s2_66M_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY scripts/stage2_run.py --size 66M --method relora_baseline    > logs/s2_66M_relo.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY scripts/stage2_run.py --size 66M --method relora_diag_gated  > logs/s2_66M_diag.log 2>&1 &
wait
```

`relora_diag_gated_fisher` runs **only on 11M** for Phase A. Don't burn 66M compute on the ablation arm unless Phase A says the Fisher gate also works.

### 4.3 Mid-run polling

Every ~30 minutes, the cloud agent should:
1. `for pid in $(cat .stage2_pids); do kill -0 $pid 2>/dev/null && echo alive || echo done; done`
2. `tail -50 logs/s2_*.log | grep -E "step|val_loss|effective_rank|merge_event"`
3. Append a one-line status to `STATUS.md`. Append-only, never overwrite.

If any run dies, **don't auto-relaunch**. Inspect the log, write a STATUS entry, and let the PI see it. Burning a 36h job is cheap; burning a 36h job with a silent bug is not.

---

## 5. Plot/report changes vs handover §4.7

Add one figure, change one column label. Everything else as planned.

| Figure | Status |
|---|---|
| `fig5_effective_rank_curves.png` | unchanged — headline figure |
| `fig6_condition_number_curves.png` | unchanged |
| `fig7_paloma_perplexity.png` | unchanged |
| `fig8_saliency_dist_at_merges.png` | **add a 5th violin per merge event for `relora_diag_gated_fisher`** so the reader can see FO and Fisher gates pick different sets |
| `fig9_gate_signal_ablation.png` (NEW) | bar chart at end-of-training val loss: `full_rank` / `relora_baseline` / `relora_diag_gated` / `relora_diag_gated_fisher`. Same task. This is the figure that answers "does the gate work because of FO specifically, or because of val-side curvature in general?" |

Report file `results/stage2/report.md` should lead with:
1. Effective-rank reproduction (Weiss replication) — `fig5`.
2. Path-A gate fixes it — `fig5` + `fig7`.
3. Gate ablation — `fig9`. (Either "Fisher also works" → method is signal-agnostic; or "only FO works" → tighten claim and own the FO framing.)

---

## 6. What must NOT change

These remain non-negotiable from `03_handover_for_gpu_agent.md` §9:

1. NEVER train on diagnostic or test_holdout.
2. NEVER auto-skip a stage.
3. NEVER merge stable updates by default — Δ_stable stays as a separate adapter slot until the report is written; the Stage 2 `relora_diag_gated` merges are part of the ReLoRA recipe, not a global merge.
4. ALWAYS log seed = 42.
5. ALWAYS save per-checkpoint state_dicts.
6. ALWAYS report effect sizes with bootstrapped CIs.
7. DO NOT silently change saliency formulas. The S3_fo_val_signed formula and sign convention are now **canonical**. Any change must be discussed first.
8. DO NOT use official GLUE val for saliency/pruning (still test_holdout, sealed).
9. DO NOT add EPI (arXiv:2604.14010) as baseline. Concurrent work. Borrow heuristics only (within-layer normalization, continuous-k-low gating — both already folded into §3.2 / §3.4 above).
10. **(new)** DO NOT silently switch the default gate signal back to Fisher in any Stage 2 script. If the cloud agent decides a different signal should be the default, it must write a STATUS entry and stop for PI confirmation.

---

## 7. Cloud-agent self-check before launching Stage 2

Tick all four before running `wait`:

- [ ] `nvidia-smi` shows ≥ 4 free GPUs (we need 4 for Phase A; GPU 2 stays off-limits).
- [ ] `$CONDA env list` returns `espo` and `$PY --version` runs cleanly.
- [ ] `git status` shows the Stage 1 results committed (don't lose them).
- [ ] `STATUS.md` has the Path A decision entry from §2 above appended verbatim.

If any item fails, **stop and write a STATUS line; do not improvise**.

---

## 8. Quick reference — sign conventions

Pulled from `src/saliency.py` and handover §3.4, so the gate predicate is unambiguous.

```text
Definition (handover §3.4):  s_i^FO = -<G, b_i a_i^T>,  where G = grad_{ΔW} L^val.
Equivalent form actually computed (saliency.py):  per_comp_i = <grad_A^val L [i,:], A [i,:]>.

Relation:  s_i^FO = -per_comp_i  (the inner-product form absorbs scaling*B^T and flips one sign).

S3_fo_val_signed (as stored in components.jsonl) = per_comp_i  (NOT s_i^FO).

Gate predicate for Path A:
    drop component i  iff  S3_fo_val_signed[i] >= 0
    (because S3_fo_val_signed >= 0  <=>  s_i^FO <= 0  <=>  merging pushes val loss UP)
```

If this sign convention conflicts with what `stage2_run.py` ends up doing, **trust the unit test in `tests/test_saliency.py::test_first_order_identity` and the empirical AUC sign in `results/stage1/*/auc_signed.json`**, not the doc. Stage 1's AUC was reported as `S3_fo_val_signed_neg_auc_harmful` — i.e. **`-S3_fo_val_signed` is the harmful-detection score**, which means positive `S3_fo_val_signed` correlates with "merging is harmful", which is exactly the predicate above.

---

## 9. Pointer back to the rest of the package

- v1 method spec: `01_research_v1.md`
- v2 white-space + theory + EPI concurrent note: `02_research_v2_baselines_theory.md`
- Cloud-agent execution plan (Stages 0–3, env probe, hard rules): `03_handover_for_gpu_agent.md`
- **You are here**: `04_stage2_directive_path_A.md`
