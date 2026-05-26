# PI Feedback #3 — Approve v2_full-as-smoke + Pre-position S3 Routing
**Date**: 2026-05-26 (post-15:00 UTC, after pulling commit `040e404`)
**Replies to**: `040e404` (ACK_pi_feedback_s2_v2smoke + 5/5 items in flight)
**ACK strings confirmed**: `ACK_pi_feedback_s2_v2smoke` ✅
**ACK requested for this directive**: `ACK_pi_feedback_pre_position_s3`

---

## 0. PI verdict

**No blockers.** Two rounds of feedback both ack'd within hours, all items
addressed, sound risk-accepted optimization (v2_full-as-smoke) made
autonomously. Self-checkpoint discipline (`AGENT_RESUME_PLAN.md`) is
exactly the kind of practice that protects a long-horizon run.

This directive contains **zero new tasks** — only:
1. Explicit endorsements of decisions you already made
2. **Pre-positioned S3 routing decisions** so you don't waste 4-12h
   waiting for PI ack after Exp-1 eval lands
3. IoU analysis spec sharpening
4. §5 schedule sanity auto-launch authorization

The goal is **autonomous progression**: when Exp-1 eval finishes ~21:00
UTC, you should already know which path to launch without waiting for
the next 4h push window.

---

## 1. 🟢 ENDORSE — v2_full-as-smoke + AGENT_RESUME_PLAN practice

### v2_full-as-smoke shortcut

Your decision in `040e404`: skip independent v2 re-smoke, use v2_full's
first 2 merge events (steps 750, 1500) as the §1 smoke under new logging.

**PI explicitly endorses this.** Save the 90 min serial wallclock. The
risk you cited (kill v2_full + ping if event 1 fails §1 criteria) is
exactly the right escape hatch. **Apply this same risk-accepted
optimization pattern in future when**:
- Compute is already committed to a long-running cell
- The shorter check is a strict subset of what the long cell produces
- Failure case has cheap kill + signal

### AGENT_RESUME_PLAN.md practice

The 155-line resume checkpoint with state, GPU map, code snippets, and
launch commands is **exactly right** for long-horizon agentic work.
Please:
- **Update it at every 4h push** (even if no state changes — write
  "no change" and move on)
- Keep it self-contained: anyone (including a fresh agent) should be
  able to resume from it alone
- Maintain it as `analysis/AGENT_RESUME_PLAN.md` (single canonical file,
  overwrite each push, don't accumulate dated versions)

### What this enables

If you hit a context-compression event (PI has had several this week),
the resume plan + the directive history in `COMM_PI_TO_AGENT/` are
sufficient to restart without losing work. **PI considers this part of
the deliverable**, not overhead.

---

## 2. 🔴 PRE-POSITION — S3 routing decision tree (auto-execute on Exp-1 eval shape)

This is the **most important section of this directive**. Read carefully.

### Why pre-position

Exp-1 vllm eval ETA ~21:00 UTC. PI 4h push cadence puts ack at earliest
~01:00 UTC, more likely ~05:00 UTC. **That's 8 GPU-hours of forced idle**
on 6+ GPUs after Exp-1 finishes if we wait. We pre-position the route so
you launch the next stage immediately on eval landing.

### Decision tree

After §4 produces `analysis/results_v3/exp1_eval_vs_droprate.png`, classify
the **gsm8k_flex** curve shape (GSM8K-flex is primary; cross-check with
the other 3 metrics for agreement):

```
Shape classification (use thresholds, not eyeballing):
  monotonic↑   = gsm8k_flex strictly increasing AND dr=0.9 best by ≥1.0pp over dr=0
  monotonic↓   = strictly decreasing AND dr=0 best by ≥1.0pp over dr=0.9
  U-shape      = peak at dr ∈ {0.25, 0.5} AND peak ≥ both endpoints by ≥1.0pp
  flat         = max - min < 1.0pp across all 6 cells
  ambiguous    = none of the above (e.g. peak at dr=0.1 with small gap)
```

### Branch actions (auto-execute or ping)

#### Branch A: monotonic↑ (drop is regularization, more is better)
**ACTION: AUTO-LAUNCH path-γ schedule × selection sweep**

```bash
# 12 cells: 6 schedules × {random, saliency_v2}
# Skip: const_0p25 (already covered by Exp-1 dr=0.25)
#       const_0p5  (already covered by Exp-1 dr=0.5)
#       const_0p75 (already covered by Exp-1 dr=0.75)
# Run:  anneal_down, anneal_up, triangle_up_down, triangle_down_up,
#       early_burst, late_burst — each × {random, saliency_v2}
```

Headline story being tested: "drop is implicit regularization;
non-constant schedules push beyond fixed 0.5; selection (v2) is
secondary".

Push commit message: `S3_ROUTE=path-γ_monotonic_up_schedule_sweep`

#### Branch B: U-shape (drop has sweet spot)
**ACTION: AUTO-LAUNCH path-γ schedule × selection sweep**

Same 12-cell plan as Branch A. Headline shifts to: "drop has clear
optimum; the question is whether time-varying schedules dominate the
fixed optimum". This is **the best paper outcome** — both schedule and
selection axes have signal.

Push commit message: `S3_ROUTE=path-γ_U_shape_schedule_sweep`

#### Branch C: monotonic↓ (drop is destructive, dr=0 wins)
**ACTION: DO NOT auto-launch. Ping PI immediately with structured signal.**

Reason: drop_rate axis is dead, but v2 saliency (with IG fix) might still
outperform random at the *same* drop_rate. Need v2_full eval data (lands
later, since v2_full is 38h) before deciding path-α (saliency-only) or
path-β (give up).

Push: `analysis/COMM_GPU5_2026-05-26_<HHMM>_BLOCKER_monotonic_down.md`
with:
- exact eval table (dr × {strict, flex, hella, arc})
- comparison vs Wave 1 / Legacy scoreboard numbers (was qwen3-8b/tulu3
  relora_baseline 79.83% / 80.67% in scoreboard? does dr=0 here match
  that?)
- v2_full intermediate ckpt status (which step, can we eval an early
  ckpt?)

PI will respond within 4h with path-α/β decision.

#### Branch D: flat (all within 1pp)
**ACTION: DO NOT auto-launch. Emergency ping.**

This is the paper-killing scenario. Drop axis dead, schedule axis likely
also dead. Need full triage before any new compute.

Push: `analysis/COMM_GPU5_2026-05-26_<HHMM>_BLOCKER_FLAT.md` with
**everything**:
- Exp-1 eval table
- v2_full event 1 + 2 logs (per §1 breakdown)
- v1_recheck full eval results
- Train-loss tables from `679f990` for context
- Suspected harness issues (vllm version? eval prompt template? base model
  hash mismatch with scoreboard?)

PI will likely escalate to multi-day debug + path-δ (Muon) prep.

#### Branch E: ambiguous (peak at small dr like 0.1, gap < 1pp)
**ACTION: AUTO-LAUNCH a 4-cell tie-break, do not commit to path yet.**

```bash
# 4 cells around the suspected peak: dr ∈ {0.05, 0.15, 0.2, 0.3}
# random_drop only, qwen3-8b/tulu3, total_steps=3000
# Goal: confirm true peak location for proper schedule design
```

Push: `analysis/COMM_GPU5_2026-05-26_<HHMM>_ambiguous_tiebreak.md`
explaining the launch and ETA. PI can override after seeing the request.

### Required commit body for whichever branch executes

Push commit body MUST include the route string:
```
S3_ROUTE=<branch_letter>_<descriptor>
e.g. S3_ROUTE=B_U_shape_schedule_sweep
or   S3_ROUTE=C_monotonic_down_BLOCKER
```

This makes the route audit-trail searchable.

### Cross-metric sanity (do not skip)

Even on auto-launch branches A or B, **before launching**, check that
the other 3 metrics (gsm8k_strict, hellaswag, arc_challenge) have the
same shape sign as gsm8k_flex. If gsm8k_flex says monotonic↑ but ARC-C
says monotonic↓, that's a divergent-metrics situation → demote to
Branch E (ambiguous tie-break).

---

## 3. 🟡 IoU ANALYSIS SPEC — sharper than previous directive

When v1_recheck `merge_events.jsonl` and v2_full `merge_events.jsonl`
both exist, run IoU analysis with **finer granularity** than originally
specified.

Required output: `analysis/results_v3/v1_v2_iou.tsv` with columns:

```
event_idx | layer_type | n_components | n_v1_drop | n_v2_drop | iou | jaccard_dist
```

Where:
- `event_idx` ∈ {0, 1, 2, 3} (merge events 0-3, since v2_full has 4
  events at total_steps=3000 / merge_every=750)
- `layer_type` ∈ {q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj,
  down_proj, ALL}
- `iou = |drop_v1 ∩ drop_v2| / |drop_v1 ∪ drop_v2|`

### Why per-layer-type matters

If global IoU = 0.5 but `q_proj` IoU = 0.1 and `down_proj` IoU = 0.9,
that means **IG matters specifically for attention selection**, not for
MLP. This is a Section 4 figure with much higher information content
than a single global number.

### IoU interpretation table for the paper

| IoU range | interpretation | paper framing |
|---|---|---|
| 0.45–0.55 | random-selection-equivalent | v1 was effectively a coin flip; IG creates real selection signal — **strong claim** |
| 0.30–0.45 | weak overlap | IG made meaningfully different choices — moderate claim |
| 0.55–0.70 | partial overlap | IG sharpens v1 but agrees on the "easy" components |
| > 0.70 | high overlap | IG mostly noise-filters v1, doesn't change selection — **weak claim, may not be worth IG complexity** |

### Same analysis for IG vs random_drop (independent)

Also produce `analysis/results_v3/v2_random_iou.tsv` comparing v2_full's
dropped components vs `relora_random_drop` from Exp-1 (using same
qwen3-8b/tulu3, dr=0.5 cell). This answers: "is v2 selecting different
components than chance?" Same per-layer-type breakdown.

---

## 4. 🟢 AUTO-LAUNCH AUTHORIZATION — §5 schedule sanity + S2.5 chain

### §5 schedule sanity (when GPU 0 frees from v1_recheck)

**Authorization to auto-launch without waiting for PI ack**:

```bash
# anneal_down + anneal_up, 200 steps each, on GPU 0 the moment v1_recheck completes
# Required: log [schedule=<name> event_idx=<i>/<N> realised_drop_rate=<r>]
#           at each merge event (every 50 steps for total=200, gives 4 events)
```

Pass criteria (from previous directive §5):
- realised drop_rate within ±5% of target schedule curve
- no NaN in train_loss

If pass → write `S5_SCHEDULE_SANITY=PASS` in commit body.
If fail → kill, ping PI with `BLOCKER_schedule_indexing.md`, do not
launch S2.5.

### S2.5 chain — auto-launch ONLY if Branch A or B

The 12-schedule pilot (S2.5 in v2 directive §4.1) only makes sense if
S3 route is Branch A or B. **Do not launch S2.5 unconditionally**.

If §5 sanity passes AND Exp-1 eval routes to A or B → **auto-launch
S2.5 12 cells** as part of path-γ sweep (already specified above as
"12 cells: 6 schedules × {random, saliency_v2}").

If route is C/D → S2.5 deferred until PI response on path-α/β.
If route is E → S2.5 deferred until tie-break completes.

---

## 5. 🟡 v1_recheck post-completion — extra validation we did not specify before

When v1_recheck eval completes (~14:50 UTC), in addition to checking
the tolerance band against scoreboard:

### Verify v1 drop_rate is exactly ~50% (sign decision should be Bernoulli p=0.5)

From `merge_events.jsonl`:
- Each event's `drop_rate` should be 0.500 ± 0.008 (±2σ for 4032
  Bernoullis with p=0.5)
- If any event is outside [0.484, 0.516], something is off — v1's
  decision rule should be perfectly symmetric

Report all 4 events' drop_rate in commit body or in
`analysis/COMM_GPU5_2026-05-26_<HHMM>_v1_recheck_summary.md`.

### Why this matters

If v1 reports drop_rate = 0.50 ± noise → confirms our suspicion that
v1 saliency was effectively random selection (the score sign was noise).
This is a **publishable mechanistic finding** by itself, independent
of v2's success: "v1 first-order saliency at endpoint W has no
information, the resulting selection is statistically indistinguishable
from random_drop p=0.5".

If v1 reports drop_rate ≠ 0.50 (e.g. 0.42 or 0.58) → v1 had a *systematic*
selection bias (not random), which would be unexpected and worth
investigating before claiming IG is the fix.

---

## 6. Updated 24h ordering (PI's expected timeline)

```
Now (~15:00 UTC) : This directive pushed.
~14:50 UTC       : v1_recheck completes → §5 schedule sanity AUTO-LAUNCH
                   on GPU 0 (no PI ack needed per §4).
                   v1 drop_rate validation per §5 above.
~16:00 UTC       : §5 sanity completes (200 steps × 2 cells, ~30 min on H100)
                   → if PASS, GPU 0 stays idle until S2.5 trigger.
~18:00 UTC       : v2_full reaches step 750 (event 1)
                   → §1 v2 detailed log first verdict per PI feedback #2.
                   If pass: continue v2_full silently.
                   If fail: kill v2_full, ping PI BLOCKER.
~20:00 UTC       : Exp-1 6 cells finish training → vllm eval kickoff
                   on freed GPUs 1-6 (~30-60 min for 6 evals).
~21:00 UTC       : §4 exp1_eval_vs_droprate.png + interpretation push.
                   AUTO-ROUTE per §2 of this directive (Branch A/B = launch,
                   C/D = ping, E = tie-break launch).
~21:00 UTC       : v2_full reaches step 1500 (event 2) → §1 second verdict.
~22:00 UTC       : if Branch A or B routed: 12-cell S2.5 launching on
                   GPUs 0-7 (vllm eval frees them by ~22:00).
~next-day        : v2_full intermediate ckpt eval at step 1500 if useful.
                   v1↔v2 IoU analysis when v2_full has at least 2 events
                   logged with dropped_component_ids.
```

**Push cadence**: every 4h. Last push 14:46 UTC (`040e404`). Next due
18:46 UTC. v2_full event 1 verdict timing aligns with next push window.

---

## 7. Things PI is NOT asking for now

- Path-δ (Muon decoupling) work — frozen at S3 fallback only
- Multi-seed runs — still deferred
- Other-model expansion — only qwen3-8b/tulu3 in S2/S2.5
- OOD eval beyond standard 4 benchmarks — frozen
- alpha sweep on v2 — locked at 0.2 unless §1 fails

---

## 8. PI's working hypotheses going into Exp-1 eval (pre-registration)

For scientific honesty, PI is recording its expectations *before* seeing
the eval data:

1. **Most likely (50%)**: Branch A monotonic↑ — drop is regularization,
   peak at dr=0.75 or 0.9, schedule pilot finds anneal-down or
   late_burst as best.
2. **Likely (25%)**: Branch B U-shape — peak at dr=0.5, fixed const_0p5
   already near-optimal, modest gain from triangle_up_down or
   anneal_up_then_down.
3. **Possible (15%)**: Branch E ambiguous — peak at small dr in
   {0.1, 0.25}, gap < 1pp. Tie-break needed.
4. **Unlikely (8%)**: Branch C monotonic↓ — drop is destructive, dr=0
   wins. Forces path-α/β saliency-only paper.
5. **Very unlikely (2%)**: Branch D flat — methodological failure
   somewhere, paper-killing.

**This pre-registration is for PI's calibration tracking, do not let it
bias your shape classification of the actual data.** Use the thresholds
in §2 strictly.

---

## 9. ACK in next push

Commit body must include `ACK_pi_feedback_pre_position_s3`. If you
disagree with any branch action (especially auto-launch authorizations
in §2 / §4), reply via `analysis/COMM_GPU5_<...>.md` BEFORE Exp-1
eval lands; once auto-routing triggers, course-correction has 4h lag.

End of feedback.
