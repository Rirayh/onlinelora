# PI Feedback #2 — Post-S1/Cross-Model + v2-Smoke Inspection
**Date**: 2026-05-26 (post-13:30 UTC, after pulling commits `bf5d452` + `679f990`)
**Replies to**: `bf5d452` (§1+§2+§4+§6 ack) + `679f990` (§3+§5 ack)
**ACK strings confirmed**: `ACK_pi_feedback_s1` + `S2.5_OPTIMIZER_VERIFIED=AdamW_all` ✅
**ACK requested for this directive**: `ACK_pi_feedback_s2_v2smoke`

---

## 0. PI verdict on progress

**Excellent pace.** Both feedback rounds were ack'd within hours, all 6
items addressed, 2 commits with substantive deliverables. S1 cross-model
result (3/3 pass) elevates the framing finding from "qwen3 quirk" to
**publishable mechanistic insight**. Train-loss monotonic data on Exp-1
@ step 1050 is already a clean figure for the paper.

That said, **5 follow-ups before S2 cells launch**. Items 1, 3 are
blocking; 2, 4, 5 are correctness/efficiency wins.

---

## 1. ⚠ BLOCKING — v2 smoke needs proper diagnostic breakdown + re-run at alpha=0.2

§2 was marked passed in commit `bf5d452` referencing smoke from commit
`242447d`. Two issues:

### Issue A — alpha drift

Smoke ran at alpha=0.1 (the default at the time). §6 then changed default
to 0.2. **Current code path has never been smoke-tested**. Re-run smoke at
alpha=0.2 before launching any S2 v2 cell.

### Issue B — `dropped` total without decomposition is not informative

Reported numbers:
```
event 1: dropped = 997 / 4032  (~24.7%)
event 2: dropped = 1863 / 4032 (~46.2%)
```

This is **post-decision** count. PI cannot tell from these numbers whether:
- (a) IG saliency genuinely strengthened between events (good — ΔW
  accumulates so per-component s_i grows out of noise)
- (b) t-stat thresholding became more permissive due to changing
  variance (neutral)
- (c) Random fallback dominated event 2 (bad — IG estimator degrades
  over training, which would tank the whole approach)

**Required for re-smoke**: log per-event triple in this exact format:
```
[v2 estimator m_ig=4 alpha=0.2] merge_event=<i>
  n_keep_sig=<a>  n_drop_sig=<b>  n_random_assigned_keep=<c>  n_random_assigned_drop=<d>
  → final keep=<a+c>  final drop=<b+d>  drop_rate=<(b+d)/4032>
  fisher_signvote_score: q05=<…>  q50=<…>  q95=<…>
```

### Pass criteria for v2 re-smoke

| condition | threshold | reason |
|---|---|---|
| `(n_keep_sig + n_drop_sig) / 4032` ≥ 0.10 | event 1 AND event 2 | at least 10% components rejected by t-stat (i.e. IG signal exists above noise) |
| `(n_keep_sig + n_drop_sig)` not decreasing event-over-event | event 2 ≥ event 1 | rules out scenario (c) — estimator degrading |
| post-merge val_loss non-NaN, recovers | both events | sanity |
| score q95 / q05 ratio ≥ 5× | both events | dynamic range — flat distributions ⇒ no information |

**If event 1 has < 10% sig rejection at alpha=0.2**: ping PI with the
breakdown. Likely action will be alpha=0.3 + m_ig=8.

**If event 2 < event 1 (degrading)**: this is a major finding by itself
— IG works at low ΔW magnitude but breaks at high ΔW. Document and ping;
do not silently work around it.

**Cost**: 1 GPU × ~90 min (200 steps + log instrumentation).

---

## 2. ⚠ CORRECTNESS — Add v1 baseline event-decomposition to v1_recheck

v1_recheck is currently running on GPU 0 (PID 2575654). Right now it only
reports final eval scores. **Cheap addition**: dump v1's per-event drop
breakdown to the same format as §1.

Expected v1 behaviour (sign-only, pure 50/50):
```
[v1 estimator] merge_event=<i>
  n_dropped=<n>  drop_rate=<n/4032>
  expected ~50%, std ~0.8% from 4032 Bernoullis
```

If v1's drop trajectory is also non-50% or also rises event-over-event,
that contaminates the v2 comparison story. We need the **side-by-side
table**:

| event | v1 drop_rate | v2 drop_rate (at alpha=0.2) | v1 vs v2 selection IoU |
|---|---|---|---|
| 1 | ~0.50 (expected) | ~0.25 (observed at alpha=0.1) | ??? |
| 2 | ~0.50 | ~0.46 | ??? |
| 3 | ~0.50 | ??? | ??? |
| ... | ... | ... | ... |

**IoU(v1_drop_set, v2_drop_set)** at each event is the headline number
for the paper's saliency section: do v1 and v2 agree on *which*
components are bad, or just on *how many*?

Implementation: in `build_keep_mask` for v1, also compute and return the
component-id set of dropped components, log it. Same for v2. Then a
postprocessing script computes IoU.

**Cost**: 1-day compute already sunk; this is just logging + a 50-line
analysis script.

---

## 3. ⚠ BLOCKING (efficiency) — Pre-launch v2 full-run cell on freed GPU

`v1_recheck` uses GPU 0; Exp-1 uses GPUs 1-6 (all 6 dr cells); GPU 7 is
free.

**Action**: launch `relora_diag_gated_S3pos_v2_full` on GPU 7 NOW with the
full S2 config (total_steps=3000, merge_every=750, alpha=0.2,
saliency_calib_n=64, m_ig=4). This is the headline v2 cell of S2. Running
it in parallel with v1_recheck saves ~100 min serial wall clock.

**Risk**: if v1_recheck fails the gate (§3 of previous directive), the v2
cell wastes ~3 GPU-hours. PI accepts this risk because:
- v1_recheck failure is unlikely (we have not changed v1 code)
- v2 logs are independently useful (per-event diagnostic from §1) even
  if v1_recheck blocks the cross-comparison

**Config**:
```bash
nohup env CUDA_VISIBLE_DEVICES=7 \
  /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python \
  scripts/stage3_run.py \
  --model_path /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B \
  --model_key qwen3-8b --dataset tulu3-sft \
  --method relora_diag_gated_S3pos \
  --saliency_estimator v2 --saliency_v2_m_ig 4 --saliency_v2_alpha 0.2 \
  --saliency_calib_n 64 \
  --total_steps 3000 --merge_every 750 --eval_every 250 --ckpt_every 9999 \
  --saliency_max_seq_len 512 --attn_implementation sdpa \
  --seed 42 --out_root results/s2/qwen3-8b/tulu3-sft/v2_full/seed42 \
  > logs/s2_v2_full.log 2>&1 &
```

**Naming**: `s2/qwen3-8b/tulu3-sft/v2_full/seed42` so it slots cleanly
into S2 results dir.

---

## 4. 🟢 FREE WIN — Exp-1 vllm eval first-priority figure

Exp-1 finishes ~20:00 UTC. The moment vllm eval completes for all 6 dr
cells, produce **a single PNG**:

`analysis/results_v3/exp1_eval_vs_droprate.png`

Layout:
- 2×2 grid
- top-left: `gsm8k_flex` vs drop_rate (6 points + connecting line)
- top-right: `gsm8k_strict` vs drop_rate
- bottom-left: `hellaswag` vs drop_rate
- bottom-right: `arc_challenge` vs drop_rate
- Each panel annotated with: dr value at peak, gap (peak − dr=0), shape
  classification (`monotonic↑` / `monotonic↓` / `U-shape` / `flat`)

**This single figure routes the paper.** PI will respond within 1h of
seeing it with S3 path α/β/γ/δ assignment.

In the same push, please include a 1-paragraph interpretation in
`analysis/COMM_GPU5_2026-05-26_<time>_exp1_eval_summary.md`:
- Best dr cell + accuracy
- Shape classification of each metric
- Cross-metric agreement: do all 4 metrics peak at the same dr?
- Train-loss minimum vs eval-best mismatch (we expect train-loss best
  at dr=0, eval-best at dr>0 — confirm or refute)

---

## 5. 🟡 SANITY — 2-cell pilot-of-pilot for schedule registry before S2.5

Before launching all 12 S2.5 schedule cells, **verify the schedule
indexing**. 30-minute investment to catch bugs that would invalidate
12 cells.

Run 200-step smoke with these two schedules (random selection, default
config else):

```bash
# anneal_down 0.75→0.25
... --method relora_random_drop --drop_schedule anneal_down \
    --total_steps 200 --merge_every 50 ...
# anneal_up 0.25→0.75
... --method relora_random_drop --drop_schedule anneal_up \
    --total_steps 200 --merge_every 50 ...
```

(merge_every=50 with total_steps=200 gives 4 merge events — enough to
verify the schedule curve is being read in correct order.)

**Required log output** at each merge event:
```
[schedule=anneal_down event_idx=<i>/4 target_drop_rate=<r>]
```

**Verification before greenlighting full S2.5**:
| schedule | expected event 1 dr | expected event 4 dr |
|---|---|---|
| anneal_down | 0.75 | 0.25 |
| anneal_up | 0.25 | 0.75 |

If first merge event of `anneal_down` reports drop_rate ≈ 0.5 (mid of
schedule) instead of 0.75, we have an off-by-one in event indexing — fix
before S2.5 launch.

**Pass condition**: realised drop fractions match expected within ±5%
(stochastic — Bernoulli over 4032 has σ ≈ 0.8%).

---

## 6. Updated 24h ordering

```
NOW (~13:35 UTC) : §1 v2 re-smoke at alpha=0.2 with full breakdown (GPU TBD)
NOW              : §3 v2_full launch on GPU 7 (3000 steps, parallel)
NOW + 0.5h       : §5 schedule registry sanity (200 steps × 2 schedules)
NOW + 1.5h       : v1_recheck completes, §2 dump v1 per-event drop_rate
NOW + 1.5h       : v1_recheck eval against tolerance band → S2 gate decision
~17:00 UTC       : v2 re-smoke complete → PI inspects breakdown
~20:00 UTC       : Exp-1 6 cells finish training → vllm eval kickoff
~21:00 UTC       : §4 exp1_eval_vs_droprate.png pushed
~21:00 UTC       : v2_full cell completes (~7h × ~0.5h/200steps × 3000=??)
                   actually 3000 steps × ~3.5min/step is too slow, recompute:
                   if Exp-1 reaches 1050 steps in ~13.5h since launch
                   (started ~midnight UTC), then 3000 step ETA = ~38h.
                   Adjust: v2_full will not complete in 24h. Plan
                   accordingly: vllm eval v2_full at next-day check.
NOW + 24h        : PI routes S3 based on Exp-1 eval shape (§4) + v2 partial
                   data (intermediate checkpoint at step 1500)
```

**Push cadence reminder**: every 4h. Last push 13:30 UTC, next 17:30 UTC.

---

## 7. Things PI is NOT asking for now

- Multi-seed runs — still deferred until S3 routing
- Other-model S2 / S2.5 cells — qwen3-8b/tulu3 only for now
- OOD eval expansion beyond the 4 standard benchmarks
- Bootstrap CI on individual eval cells — can be added in path α/β when
  we report final tables

---

## 8. Open scientific question for the agent (not blocking)

The Exp-1 train-loss data is striking enough that PI wants you to think
about this for the paper writeup:

> "Drop_rate produces a monotonic train-loss → drop is genuinely
> destructive at training time. But if eval also peaks at non-zero
> drop_rate (U-shape), that's the cleanest possible 'drop is implicit
> regularization' figure. We then need ONE more experiment: explicit
> dropout/weight-decay sweep at dr=0 to compare regularization
> mechanisms. **Note this for path-γ (drop-rate paper) but do not run
> yet** — only relevant if §4 figure shows U-shape."

This is a flag, not a directive. Just keep the experiment design in mind
when writing your next status doc.

---

## 9. ACK in next push

Commit body must include `ACK_pi_feedback_s2_v2smoke`. Disagreements
welcome via `analysis/COMM_GPU5_<...>.md` reply file.

End of feedback.
