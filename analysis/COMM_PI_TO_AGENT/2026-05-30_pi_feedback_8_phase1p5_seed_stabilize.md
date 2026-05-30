# PI Feedback #8 — Phase 1.5 seed stabilization + paired-t for Phase 1

**Date**: 2026-05-30
**Prior**: feedback #7 (Phase 1.5 launched), agent runtime + eval results 2026-05-30
**ACK token**: `ACK_pi_feedback_8_phase1p5_seed_stabilize`
**Tag in commit**: cite this filename

---

## 1. Why this directive (read first)

Phase 1 (n=3) is clean: v1_S3pos beats random_dr=0.5 on 5/6 metrics
(gsm8k +3.18pp, arc-c +1.65, mmlu +0.66, ifeval +2.09, hellaswag −1.08).
Phase 1 verdict needs only paired-t + a write-up.

Phase 1.5 (n=1, seed 42) is **noisy and ambiguous**:

| schedule | gsm8k | ifeval |
|---|---:|---:|
| v1_S3pos (s42)         | 79.61 | 40.67 |
| random_anneal_up       | 73.39 | 27.54 |
| random_anneal_down     | **77.94** | **45.47** |
| random_triangle_up_down| 74.98 | 38.63 |
| random_triangle_down_up| running | — |

`random_anneal_up` is **6.22pp below v1** on gsm8k → kills the
"v1's monotonic-up shape is the cause" hypothesis. ✅ good for our story.

But `random_anneal_down` (decreasing drop-rate) hits 77.94 gsm8k and 45.47
ifeval at seed 42 — only 1.67pp behind v1 on gsm8k and **ahead of v1 on
ifeval by 4.80pp**. Per feedback #7 decision rule this falls in
`[0, 2.0) → SALIENCY_WEAKLY_ADDS_VALUE`.

**This is single-seed.** Phase 1 ifeval seed-spread for the same cell is
{30.31, 31.42, 25.14} = 6pp range. Without n=3 we cannot tell if
anneal_down is genuinely competitive or a 1-seed lucky draw.

**Phase 1.5 must be brought to n=3 on at least the cell that threatens
the story before any verdict gets written.**

---

## 2. Tasks (priority-ordered)

### §A — Phase 1.5 seed stabilization (HIGHEST PRIORITY)

**A1.** Run `random_anneal_down` at **seeds 43 and 44**, full 3000 steps,
same config as seed42 cell, with `--save_merged_final`. Then lm-eval
both on the same task suite (gsm8k_strict, gsm8k_flex, hellaswag, arc_c,
mmlu, ifeval).

   - Out root: `results/phase1p5_schedule_ablation/qwen3-8b/tulu3-sft/random_anneal_down/seed{43,44}/`
   - Reuse seed42 cell, do not retrain it.

**A2.** Wait for the in-flight `random_triangle_down_up/s42` to finish.
Then check its gsm8k. If it **≥ anneal_down s42 (77.94)**, treat it as
the real story-threat instead and run **its** seeds 43, 44 too. If it is
< anneal_down, no further triangle_down_up training needed.

**A3.** v1_S3pos already has seeds 42,43,44 in Phase 1 (`results/phase1_robustness/.../v1_S3pos/seed{42,43,44}/`). **Do NOT retrain v1.** Just reuse those three runs as the v1 column for Phase 1.5 n=3 comparison too. Same for random_dr0.5 (= constant 0.5 schedule).

**A4.** When all the above evals exist, recompute the Phase 1.5 decision:

```
delta_v1_vs_best_random_schedule_n3 =
    mean(v1_gsm8k, n=3) - max_over_schedules(mean(random_*_gsm8k, n=3))

If max-schedule cell only has n=1 still → mark VERDICT_PENDING_N3 and DO NOT
write the decision.md as final.
```

Apply paired-t (v1 seeds 42/43/44 vs winning-random-schedule seeds 42/43/44)
with α=0.10 as in #6/#7.

### §B — Phase 1 paired-t + decision write-up (medium priority, do after §A1 launches)

**B1.** Compute paired-t on Phase 1 (n=3, qwen3-8b/tulu3-sft):

   - v1_S3pos vs random_dr0.5 → on each of {gsm8k_strict, hellaswag, arc-c, mmlu, ifeval}
   - v1_S3pos vs relora_baseline → same

**B2.** Run `scripts/phase1_decision_analysis.py`. Push the resulting
`analysis/results_phase1/phase1_decision.md` (or wherever it lands) with
seed-level rows + paired-t p-values + verdict per metric.

   - Decision rule (from #6): gsm8k Δ(v1, random_dr=0.5) ≥ 1.5pp + paired-t
     p<0.10 → PASS. Anything else → soft-claim.

**B3.** Do **NOT** trigger Phase 2 (cross-model olmo2-7b / llama3-8b)
yet. Hold Phase 2 until after §A4 + §B1 jointly verdict, because
Phase 1.5 outcome can change what the paper claim looks like.

### §C — Phase D continues, no change

PhaseD is at ~5175/10000 (50%) and healthy. Let it finish. Eval it on
the same suite after `merged_final` lands. No additional action needed.

### §D — Hygiene (low priority but worth doing)

**D1.** When a Phase 1.5 cell has duplicate result JSONs from a retry
(you mentioned `random_anneal_up` has dups), delete the older ones and
keep only the latest. Document deletion in next runtime_progress.md so we
keep an audit trail.

**D2.** Replace the temporary autodrain loop with a simple drain mode in
`phase1D_eval_orchestrator.py` (a flag `--drain` that loops the existing
launch logic until the pending list is empty), so this isn't carried as
a "running shell loop in tmux" tech debt into Phase 2/PhaseD-eval. This
is non-blocking; do it when no eval batch is queued.

---

## 3. Concrete commands (suggested, adapt to your env)

```bash
# §A1: anneal_down seeds 43, 44 train
python scripts/phase1p5_train_orchestrator.py \
    --cells random_anneal_down \
    --seeds 43 44 \
    --save_merged_final \
    --gpus 5,6   # whichever 2 are free

# §A1: their evals (after merged_final exists)
python scripts/phase1D_eval_orchestrator.py --phase1p5 \
    --gpus 5,6,7   # or whatever is free
# Filter to only the new cells via existing has_result short-circuit.

# §A2: triangle_down_up gsm8k check
python -c "
import json, pathlib
p = pathlib.Path('results/phase1p5_schedule_ablation/qwen3-8b/tulu3-sft/random_triangle_down_up/seed42')
result = max(p.glob('lm_eval_*.json'), key=lambda x: x.stat().st_mtime)
data = json.loads(result.read_text())
print(data['results']['gsm8k']['exact_match,strict-match'])
"

# §B1: paired-t for Phase 1 (3 seeds × 3 cells)
python scripts/phase1_decision_analysis.py    # already on disk per #6 ack
```

---

## 4. Output expected (push these as new files)

1. `analysis/COMM_AGENT_TO_PI/2026-05-30_agent_ack_pi8_phase1p5_seed_stabilize.md` — ACK with launch ETA + GPU plan.
2. `analysis/results_phase1/phase1_decision.md` — Phase 1 paired-t + verdict (per §B2).
3. `analysis/COMM_AGENT_TO_PI/<date>_phase1p5_n3_results.md` — Phase 1.5 with anneal_down (and possibly triangle_down_up) at n=3, plus updated decision (or `VERDICT_PENDING_N3`).
4. Updated `analysis/COMM_AGENT_TO_PI/<date>_agent_runtime_progress.md` per usual cadence.

Do **not** commit `results/`, `logs/`, `merged_final/` weights, or the
duplicate result JSONs you delete in §D1.

---

## 5. Decision tree after this batch

```
After §A4 (Phase 1.5 n=3) + §B1 (Phase 1 paired-t):

(1) Phase 1 v1 vs random_dr0.5 paired-t passes (p<0.10, Δ>=1.5pp)
    AND
    Phase 1.5 v1 vs best-random-schedule (n=3) Δ_gsm8k ≥ 2.0pp
    → SALIENCY_ADDS_VALUE confirmed → launch Phase 2 (cross-model) +
      start drafting paper §1-§3 in parallel.

(2) Phase 1 passes paired-t but Phase 1.5 Δ ∈ [0, 2.0)
    → SALIENCY_WEAKLY_ADDS_VALUE → still launch Phase 2 but soften
      paper claim to "saliency selection consistently beats matched-rate
      random and is competitive with the strongest schedule heuristic".
      In this case the paper has a non-trivial schedule-design discussion
      section — anneal_down is a finding, not a competitor we ignore.

(3) Phase 1.5 Δ < 0 (some random schedule beats v1 at n=3)
    → STORY_FLIP → paper becomes "drop-rate scheduling drives ReLoRA
      recovery; saliency is a failed alternative selection signal".

(4) Phase 1 paired-t fails (Δ < 1.5pp or p ≥ 0.10)
    → method-vs-random fails → consider negative-result write-up or
      pivot back to OPLoRA-style continual-learning setting.

PI will pick branch and write feedback #9 once §A4 + §B1 land.
```

---

## 6. Open question (no action required, just for your awareness)

`random_anneal_down` posting the highest ifeval (45.47) at single seed
is a separate finding worth noting. ifeval has high seed variance in
Phase 1 (relora_baseline {30.31, 31.42, 25.14}, 6pp spread), so the
seed-43/44 reruns will tell us whether anneal_down's ifeval signal is
real. If it holds at n=3, we may want a small additional ablation —
"high-then-low drop rate as instruction-following stabilizer" — but
that's strictly future work, not blocking.

— PI
