# PI Directive — Saliency Estimator Revamp + Drop-Schedule Pilot (v2)
**Date**: 2026-05-26
**Replaces**: `2026-05-26_pi_muon_decoupling.md` (Muon experiment is now S3 fallback, not first action)
**ACK string requested in next agent push**: `ACK_v2_saliency_revamp`

---

## 0. Why this directive replaces the Muon-first plan

The Muon directive jumped to "remove AdamW reg confound" without first
verifying that **the saliency estimator itself is working**. After re-reading
`src/saliency.py` and `scripts/stage3_run.py`, three concrete framing errors
were found that are **strictly upstream** of any optimizer-level confound. If
those are unfixed, Muon will not save the saliency story even if AdamW is the
problem; we'd be blaming the optimizer for an estimator bug.

Order of operations is therefore:

```
S1  Diagnostic (1 cell)  — Spearman G(W₀) vs G(W) to confirm framing error
S2  Saliency revamp (1)  — ABC-fixed saliency on qwen3-8b/tulu3
S2.5 Schedule pilot (≥10) — drop-rate schedules including counter-intuitive
S3  Route by S2/S2.5     — saliency-saved | schedule-saved | Muon-fallback
```

`STOP / DEFER`: multi-seed, other-model sweep, OOD eval, Wave 1 stragglers.
None of these matter if S1/S2 reveal the estimator is broken.

---

## 1. Background — What is the current saliency, what is wrong with it

### 1.1 Current formula (verified against code)

`src/saliency.py:64-97`:

```
h.A.grad      = ∂L/∂A = scaling · Bᵀ · G        (G = ∂L/∂W in current state)
per_comp_i    = ⟨∂L/∂A[i,:], A[i,:]⟩
              = ⟨G, scaling · bᵢ aᵢᵀ⟩
              = ⟨G, ΔWᵢ⟩
```

Note: file-level docstring says `s_i^FO = -⟨G, ΔWᵢ⟩`; **code computes
`+⟨G, ΔWᵢ⟩`**. This is a sign-convention bug in the docstring (does not
affect symmetric S3pos/S3neg decisions, but please align both before paper
submission).

Decision (`scripts/stage3_run.py:464-516`):
- `S3pos_drops`: `keep_mask = (s < 0)` → keep components whose ΔWᵢ is helpful
- `S3neg_drops`: `keep_mask = (s > 0)` → opposite (sanity arm)
- `random`: Bernoulli p=0.5

### 1.2 Three framing errors (priorities A > B > C > D)

#### **A. Wrong evaluation point** — IG framing

`s_i = ⟨G, ΔWᵢ⟩` is computed at the **end-of-segment** weight
W = W₀ + ΔW. The Taylor expansion answers:

> "If I subtract ΔWᵢ from the current W, how does loss change?"

But what we *should* be asking is:

> "Over the trajectory from W₀ to W₀+ΔW, how much did this ΔWᵢ contribute?"

The correct quantity is **path integral** / **Integrated Gradient**:

```
s_i^IG = ∫₀¹ ⟨G(W₀ + t·ΔW), ΔWᵢ⟩ dt
       ≈ (1/m) · Σ_{t∈linspace(0,1,m)} ⟨G(W₀ + t·ΔW), ΔWᵢ⟩
```

Our `m=1` end-point approximation collapses **as the model approaches
convergence** (G → 0 ⇒ all s_i → 0), which is exactly what we observed
after P0 fix. The "signal floor" we attributed to noise is the IG-truncation
artefact.

#### **B. Sign-only decision discards SNR information**

`m = (s < 0)` is binary on the sign of mean(grad·A) computed from
**16 examples** (`diag_batches=8`, `bs=2`). Components with mean ≈ 0
have sign dominated by sampling noise; they get force-binarised into
keep/drop based on a coin flip.

The principled estimator is **t-statistic gating with FDR control**:

```
t_i = mean_i / (std_i / √n)
keep = (t_i < -threshold)
drop = (t_i > +threshold)
mid  = ~(keep | drop)         # below significance → fall back to random p=0.5
```

This naturally produces a **dynamic drop_rate** (high when SNR is high,
falls back to random when signal is weak), which dovetails with §3 below.

#### **C. Average-gradient erases per-sample direction signal**

`mean(grad·A)` allows sample-level cancellation. The decoupled form is:

```
fisher_i      = E_x[ ⟨grad_x A, A⟩² ]              # magnitude (always ≥ 0)
sign_vote_i   = E_x[ sign(⟨grad_x A, A⟩) ]         # direction agreement ∈ [-1,1]
score_i       = sign_vote_i · √fisher_i            # only large + consistent → significant
```

`fisher_saliency` is already implemented in `src/saliency.py:101` but **not
wired into `stage3_run.py`**.

#### **D. Second-order curvature** (deferred; only if A+B+C don't restore signal)

OBD/OBS-style `Δloss ≈ ½ s·H·s` with K-FAC approximation of H. Skip unless
S2 still lies on the random-drop floor.

#### **E/F/G — accepted as inherent limitations** (not fixed in this directive)

- E: drop_rate sweep is the entire point of S2.5 below
- F: val ≠ deployment distribution — already has `--saliency_calib_set` flag
- G: signed FO can flip across distributions — irreducible

### 1.3 The "is it diagnosing the difference?" question

PI verified: **mathematically yes** — `s_i = ⟨G, ΔWᵢ⟩` operates on the
component delta `bᵢaᵢᵀ`. **Operationally not really** — G is evaluated at
the wrong point (current W, not over the trajectory). **A** fixes this.

---

## 2. S1 — Diagnostic: confirm framing error empirically (1 cell, ~2h)

**Goal**: prove (or refute) that endpoint saliency disagrees with start-point
saliency. If they disagree, A is the dominant problem. If they agree,
demote A and proceed.

**Protocol** (qwen3-8b/tulu3, single seed=42):

1. Take an existing trained checkpoint (e.g. `qwen3-8b/tulu3/relora_baseline`
   at end of training, just before any merge).
2. Compute `s^end_i = ⟨G(W₀+ΔW), ΔWᵢ⟩` (current code).
3. Compute `s^start_i = ⟨G(W₀), ΔWᵢ⟩` by:
   - cloning B, A
   - setting B := 0 (so ΔW collapses to 0, model = base)
   - running `first_order_saliency(...)` with calib loader
   - the per-component product `(grad_A_with_B_zeroed) · A_original` is the
     start-point quantity (math check below)
   - restore B to original after
4. Compute Spearman ρ between `s^end_i` and `s^start_i` flattened across
   layers. Also report:
   - top-10% IoU (Jaccard of "highly helpful" components)
   - sign-flip rate (fraction of components where sign(s^end) ≠ sign(s^start))

**Math check for start-point saliency with B=0**:
```
With B=0 + A unchanged: ΔW = 0, model output = base model output, G = ∂L/∂W|_{W₀}.
∂L/∂A[i,:] at this point = scaling · Bᵀ · G = 0  (B is 0)
```
That collapse means we cannot use the standard `grad_A · A` form at B=0.
**Workaround**: compute the start-point saliency as

```
s^start_i = ⟨G(W₀), scaling · b_orig_i · a_orig_iᵀ⟩
```

i.e. compute G at W₀ (forward+backward with B set to 0, but use **original**
b_i, a_i in the inner product). Implementation:

```python
# pseudocode
B_orig = {h.name: h.B.detach().clone() for h in handles}
for h in handles: h.B.data.zero_()

# now backward at W₀
loader_results = []
for batch in calib_loader:
    out = model(**batch); out.loss.backward()
    # Capture ∂L/∂(B@A) implicitly via grad on a hooked handle.
    # Simplest: hook the linear's base_layer.weight to grab G directly.
    ...

# After collecting G per layer, compute
s_start[h.name] = (G_layer @ A_orig.T * scaling * B_orig).sum(...)   # element-wise
```

A simpler-but-equivalent route: **temporarily scale B by ε ≈ 1e-3 instead of
zero**, run normal `first_order_saliency`, then divide by ε. This avoids
hooks. Document whichever you implement.

**Deliverable**:
- `analysis/results_v3/saliency_framing/spearman_qwen3-8b_tulu3.json`
- a 2-panel scatter PNG: x=s^end, y=s^start (one panel per layer-type),
  with Spearman ρ and IoU annotated
- a one-paragraph text summary: is ρ < 0.3? if yes, A is critical.

**Decision rule**: if Spearman ρ ≥ 0.5 across layer-types → A is **not** the
dominant problem; demote A in S2 (skip IG, only do B+C). If ρ < 0.3 → A
is critical, must implement IG.

---

## 3. S2 — ABC-fixed saliency (1 cell, ~6h)

**Goal**: re-run qwen3-8b/tulu3 with the new saliency estimator. If S3pos
opens a clear gap (≥1pp) over random_drop, the estimator was the culprit
and we sweep widely. If no gap, the estimator wasn't the bottleneck.

### 3.1 Implementation

New file: `src/saliency_v2.py`. Do **not** mutate `src/saliency.py` (we still
need to reproduce the old numbers for ablation). Add a `--saliency_estimator
{v1,v2}` flag to `scripts/stage3_run.py`, default `v1` (no behaviour change
for unrelated runs).

`saliency_v2.py` must implement:

1. **A — IG with m=4 interpolation points**
   ```python
   def integrated_gradient_saliency(model, handles, loader, m=4, ...):
       deltas = {h.name: (h.B.detach().clone(), h.A.detach().clone())
                 for h in handles}
       per_sample_records = []   # for B (t-stat) and C (sign vote)
       for t in torch.linspace(0.0, 1.0, m+1)[1:]:   # skip t=0 (degenerate)
           # interpolate ΔW by scaling B by t (A held fixed)
           for h in handles:
               h.B.data.copy_(deltas[h.name][0] * t)
           # collect per-sample (not per-batch) saliencies
           records_t = first_order_saliency_per_sample(
               model, handles, loader, signed=True, ...)
           per_sample_records.extend(records_t)
       # restore
       for h in handles:
           h.B.data.copy_(deltas[h.name][0])
       return per_sample_records  # downstream applies B+C
   ```

2. **B — t-statistic gating**
   ```python
   def t_stat_decision(per_sample_records, alpha=0.1):
       # records: shape (n_samples_total, n_layers, r), already signed
       S = stack(per_sample_records)
       mean = S.mean(0); std = S.std(0).clamp_min(1e-8)
       n = S.shape[0]
       t = mean / (std / sqrt(n))
       # BH-FDR correction across all components in this merge event
       p_two_sided = 2 * (1 - student_t.cdf(|t|, df=n-1))
       reject = benjamini_hochberg(p_two_sided, q=alpha)
       keep = reject & (mean < 0)        # significant + helpful
       drop = reject & (mean > 0)        # significant + harmful
       mid  = ~reject                    # not significant → random p=0.5
       random_assign(mid, p=0.5)
       return keep_mask
   ```

3. **C — Fisher × sign-vote**
   ```python
   def fisher_signvote_score(per_sample_records):
       fisher    = (S ** 2).mean(0)                     # ≥ 0
       sign_vote = S.sign().mean(0)                     # ∈ [-1, 1]
       return sign_vote * sqrt(fisher)
   ```

4. **Increase calib budget**: `diag_batches=64, bs=4 → 256 examples` (16×
   over current). At m=4 IG points × 256 examples × 28 layers, this is
   ~30s extra per merge event on H100; acceptable.

### 3.2 Run config

`qwen3-8b/tulu3 × {lora_vanilla, relora_baseline, relora_random_drop,
relora_diag_gated_S3pos_v2, relora_diag_gated_S3pos_v1_recheck}` =
**5 cells**, single seed=42.

The `_v1_recheck` cell **must reproduce** the existing scoreboard number
(86.43% strict / 86.96% flex). If it doesn't, debug the harness before
trusting v2 numbers.

### 3.3 Pass/fail rules

- **Saliency-saved**: S3pos_v2 beats `max(random, baseline)` by ≥1.0pp on
  GSM8K-flex → A+B+C was the problem; promote v2 and run full sweep in S3.
- **Saliency-still-dead**: S3pos_v2 within 1pp of random_drop → estimator is
  not the bottleneck; route to S2.5 schedule pilot and Muon fallback.

---

## 4. S2.5 — Drop-schedule pilot (≥10 schedules, ~12h, parallelisable)

**Goal**: drop-rate is non-stationary across merge events; test whether
**any** schedule (including counter-intuitive ones) beats fixed 0.5.

**Critical**: use **random selection** (not saliency) so this experiment
isolates the schedule effect. Saliency × schedule joint sweep is S3 territory.

Single config: qwen3-8b/tulu3, seed=42, 6 merge events (default
`merge_every=500`). Each schedule defines `drop_rate(event_t)` for
t ∈ {0, 1, 2, 3, 4, 5}.

### 4.1 Mandatory schedules (≥12 cells)

| # | name | per-event drop_rate (events 0..5) | rationale |
|---|---|---|---|
| 1 | `const_0p5` (baseline) | 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 | reference |
| 2 | `const_0p25` | 0.25 × 6 | low fixed |
| 3 | `const_0p75` | 0.75 × 6 | high fixed |
| 4 | `anneal_down` | 0.75, 0.65, 0.55, 0.45, 0.35, 0.25 | simulated annealing (explore→exploit) |
| 5 | `anneal_up` | 0.25, 0.35, 0.45, 0.55, 0.65, 0.75 | curriculum dropout (Morerio 2017) mirror |
| 6 | `triangle_up_down` | 0.25, 0.45, 0.65, 0.65, 0.45, 0.25 | mid-peak (matches our SNR-peak hypothesis) |
| 7 | `triangle_down_up` ⚠ | 0.75, 0.55, 0.35, 0.35, 0.55, 0.75 | **counter-intuitive**: low in middle |
| 8 | `early_burst` ⚠ | 0.9, 0.5, 0.5, 0.5, 0.5, 0.5 | one big shock early, then stable |
| 9 | `late_burst` ⚠ | 0.5, 0.5, 0.5, 0.5, 0.5, 0.9 | "final reset" before end |
| 10 | `bookend_burst` ⚠ | 0.9, 0.3, 0.3, 0.3, 0.3, 0.9 | shocks at both ends |
| 11 | `extreme_alternate` ⚠⚠ | 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 | all-or-nothing alternating |
| 12 | `random_schedule` | per-event ~ U[0.1, 0.9], **fixed seed for reproducibility** | sanity baseline: is any non-constant schedule better than constant? |

⚠ = counter-intuitive, included on purpose (we may be surprised). ⚠⚠ =
extreme stress-test, may fail catastrophically — that's fine, we want to
know.

### 4.2 Optional bonus schedules (run if compute allows)

| # | name | per-event drop_rate | rationale |
|---|---|---|---|
| 13 | `cosine_anneal_down` | 0.5·(1+cos(πt/5)) → maps to [0.0, 1.0] re-scaled to [0.25, 0.75] | smooth anneal |
| 14 | `step_high_then_low` | 0.75, 0.75, 0.25, 0.25, 0.25, 0.25 | hard transition mid-training |
| 15 | `step_low_then_high` | 0.25, 0.25, 0.25, 0.75, 0.75, 0.75 | reverse |
| 16 | `single_skip` | 0.5, 0.5, 0.0, 0.5, 0.5, 0.5 | one merge event with NO drop (skip event 2) |
| 17 | `geometric_decay` | 0.9, 0.6, 0.4, 0.27, 0.18, 0.12 | aggressive front-load |

### 4.3 Implementation

Add `--drop_schedule` arg to `scripts/stage3_run.py` accepting either:
- a name from a registry (`const_0p5`, `anneal_down`, ...)
- a comma-list `0.9,0.5,0.5,0.5,0.5,0.5`
- `random_schedule:seed=N` for reproducible random

Wire through to `build_keep_mask`: pass `target_drop_rate=schedule[event_idx]`.

### 4.4 Deliverable

- `analysis/results_v3/schedule_pilot/results_table.tsv` with columns:
  `schedule, gsm8k_strict, gsm8k_flex, hellaswag, arc_challenge,
   final_train_loss, drop_rate_per_event_actual` (the actual realised drop
  rate, since random schedule is stochastic per event).
- `analysis/results_v3/schedule_pilot/per_event_loss_curves.png`: training
  loss across all 12 schedules, one line per schedule, vertical bars at
  merge events (annotated with that event's drop rate).
- 4-paragraph text summary in `analysis/COMM_GPU5_2026-05-26_schedule_pilot.md`:
  best schedule, gap to const_0p5, any counter-intuitive winners?,
  recommendation.

### 4.5 Pass/fail rules

- **Schedule-saved**: any schedule beats `const_0p5` by ≥1.0pp on
  GSM8K-flex → schedule dimension matters; promote to S3 saliency × schedule.
- **Schedule-flat**: all within 1pp → schedule doesn't matter at fixed drop=
  0.5 mean; the variance was ours to shrink. Skip schedule axis in S3.

---

## 5. S3 — Routing decision

After S2 + S2.5, route based on the 2×2 outcome matrix:

```
                 schedule helped (≥1pp)         schedule flat (<1pp)
saliency saved    PATH α: full sweep over        PATH β: full sweep saliency
(≥1pp gap)        models with v2 saliency       only, fixed const_0p5
                  + best schedule, find          (closest to current setup)
                  if α > β

saliency dead    PATH γ: drop-rate paper —       PATH δ: Muon decoupling
                 reframe story around schedule   experiment from prior
                 / drop-rate, not saliency       directive (Exp-1 + Exp-2
                                                 with Muon)
```

**Do not** run S3 until both S2 and S2.5 are complete and PI has
acknowledged the route choice in `COMM_PI_TO_AGENT/`.

### Path-α / β config (saliency-saved)

Models: qwen3-8b, qwen35-9b, olmo2-7b, llama3-8b (drop r1-distill, mistral —
inconclusive in earlier runs).
Datasets: tulu3-sft, metamathqa-10k.
Methods: lora_vanilla, relora_baseline, relora_random_drop,
relora_diag_gated_S3pos_v2, dora, cola.
Seed: 42 only (single-seed is accepted limitation; CI by bootstrap on
GSM8K eval examples).

### Path-γ config (drop-rate paper)

Same models × datasets, only `random_drop_with_best_schedule` vs
`random_drop_const_0p5` vs `lora_vanilla`. Story: "stochastic
schedule-aware drop is the operative mechanism, no information-theoretic
selection needed."

### Path-δ config (Muon fallback)

Use the original 2026-05-26_pi_muon_decoupling.md plan
(Exp-1 drop_rate sweep + Exp-2 2×2×2). PI provides Muon impl directive
at that point.

---

## 6. Ordering & deliverable cadence

```
Day 0 (now)    : ACK this directive in next agent commit (string above)
Day 0–1        : S1 (Spearman framing test) — 1 cell
Day 1          : S2 implementation (saliency_v2.py + stage3_run wiring)
Day 1–2        : S2 run (5 cells qwen3-8b/tulu3)
Day 2–3        : S2.5 schedule pilot (12 mandatory + up to 5 bonus cells)
Day 3          : Decision push from PI → S3 route
```

Deliverable cadence (unchanged): commit + push every 4h, even if only
partial logs. Use `analysis/COMM_GPU5_<date>_<topic>.md` for replies.

---

## 7. What NOT to do

- **No multi-seed runs** until route is chosen. Single seed=42 is fine for
  S1/S2/S2.5 because we are looking for ≥1pp effects.
- **No new models/datasets** in S1/S2/S2.5. We are debugging the method,
  not surveying.
- **No OOD bench expansion**. GSM8K-flex + HellaSwag + ARC-C is enough
  signal for routing.
- **No Muon implementation yet**. Muon is path-δ only.
- **Do not modify `src/saliency.py`**. Add `src/saliency_v2.py` so v1 stays
  reproducible.
- **Do not silently skip schedules** in §4.1. If a schedule (e.g.
  `extreme_alternate`) blows up training, log it as a failure case and
  report; don't omit.

---

## 8. Code locations to touch

| file | change |
|---|---|
| `src/saliency_v2.py` | NEW — IG + per-sample + t-stat + Fisher×signvote |
| `scripts/stage3_run.py` | add `--saliency_estimator {v1,v2}`, `--drop_schedule`, `--saliency_calib_n` already present |
| `scripts/stage3_run.py:464` | `build_keep_mask` accepts `target_drop_rate` from schedule (already supported, just route) |
| `scripts/run_schedule_pilot.sh` | NEW — driver for §4.1 + §4.2 |
| `scripts/run_s1_framing_test.py` | NEW — Spearman G(W₀) vs G(W) |
| `analysis/results_v3/saliency_framing/` | NEW dir |
| `analysis/results_v3/schedule_pilot/` | NEW dir |

---

## 9. Communication

- ACK in commit body of next push: `ACK_v2_saliency_revamp`
- Reply file: `analysis/COMM_GPU5_2026-05-26_<S1|S2|S2.5>_results.md`
  per stage
- Blocker file: `analysis/COMM_GPU5_2026-05-26_BLOCKER.md` if anything in
  this directive is unimplementable; PI will respond within 4h

---

## 10. Open scientific questions for the agent to keep in mind

1. If S1 finds Spearman ρ ≈ 1.0 (start ≈ end), then path A is *not* the
   problem and the IG implementation in S2 still adds compute but no value
   — **report this honestly** rather than retro-fitting the result.
2. In S2.5, if `extreme_alternate` (0,1,0,1,0,1) wins, that is a major
   negative finding for selection-based methods (random restart > anything
   informational) and needs to be in the paper.
3. In S2.5, if `triangle_down_up` (low in middle) wins, it would refute
   the "mid-training has highest saliency SNR" hypothesis. Look at the
   training loss curve to see what was happening mid-training in the
   winning schedule.
4. The 1pp threshold for "saved" is single-seed; the real number from
   bootstrap CI may be larger. Compute and report 95% bootstrap CI on
   GSM8K accuracy in all S2/S2.5 cells.

End of directive.
