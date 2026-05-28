# PI Feedback #6 — Robustness sweep + story pivot

**Date**: 2026-05-28
**Severity**: Story-decisive. n=1 results are insufficient for ICLR 2027.
**Supersedes**: open questions in `2026-05-28_agent_pi5b_final_results.md`

## TL;DR

Excellent execution on #5/#5b. The 6-cell results are publishable in
direction but the **+2.50pp v1 vs random_dr=0.5 gap on gsm8k is the
sole load-bearing number**, and it's well within typical lm-eval
single-seed noise (1-2pp). We cannot ship the paper on n=1.

This directive: **(a) story pivot** — drop the v1-vs-v2 axis; (b) **robustness
sweep** (3 model × 3 cell × 3 seed) to lock the +2.5pp signal; (c) **add
mmlu / ifeval / math500**; (d) **vanilla over-train control** to characterize
the gsm8k cost honestly.

---

## A. Answers to your 4 open questions

1. **Merge cost (8pp gsm8k vanilla vs v1)**: yes, expected for ReLoRA. The
   ReLoRA paper itself reports task-accuracy regressions vs vanilla LoRA on
   reasoning tasks. We do not need to close this gap; we need to **position
   our contribution honestly**.
2. **Hellaswag improvement of relora variants**: known regularization effect
   from periodic resets. Useful supporting evidence, not standalone novelty.
3. **v1 vs v2 (IG-FDR)**: shelve. Both legitimate. v1 wins gsm8k, v2 wins
   hellaswag. Picking a winner is over-claiming.
4. **Next step**: this directive (robustness sweep + new benchmarks +
   vanilla long-train control). NO single-config publication run yet.

## B. Story pivot (mandatory before any new training)

**Old framing** (from #5b): "Saliency-aware ReLoRA prevents post-merge
val degradation." → val_loss is not the headline metric.

**New framing**:
> Saliency-aware ReLoRA recovers a substantial fraction of vanilla LoRA's
> task-accuracy on math reasoning that the standard ReLoRA baseline
> destroys, while preserving ReLoRA's regularization gains on commonsense
> reasoning. Concretely on qwen3-8b/tulu3-sft, our v1 variant closes
> **53% of the gsm8k gap** between ReLoRA baseline (70.28%) and vanilla
> LoRA (87.64%) — i.e. v1 reaches 79.53%, recovering 9.25pp of the 17.36pp
> total gap.

Required headline numbers (n>=3 seeds, error bars):
- v1 vs relora_baseline gsm8k delta (the +9.25pp number — primary)
- v1 vs random_dr=0.5 gsm8k delta (the +2.5pp number — establishes
  "selection > random at same drop rate")
- v1 vs lora_vanilla hellaswag delta (the +1.9pp number — preserves the
  ReLoRA hellaswag advantage)

If any of these three goes inside the seed-noise band after n=3, story
adjusts. We need to know now, not at submission.

## C. Robustness sweep (the experiment that must run next)

### C.1 Phase 1 (priority, ~90 GPU-h, 2-3 days on 4-8 GPUs)

**qwen3-8b/tulu3-sft × 3 cells × 3 seeds = 9 cells**

cells:
- `v1_S3pos`        (the win condition)
- `random_dr0.5`    (the "selection > random" comparator at matched rate)
- `relora_baseline` (the "method > baseline" comparator)

seeds: `42, 43, 44`

config: identical to s2_pi5b_v3 (total_steps=3000, merge_every=750,
        --save_merged_final, lr/optim unchanged)

eval: gsm8k_strict + gsm8k_flex + hellaswag + arc_challenge + **mmlu (5-shot)**
      + **ifeval** (instruction-following, native lm-eval task) on each cell.

Output: `analysis/results_v3/phase1_robustness/{cell}/seed{42,43,44}/scores.json`
        + a consolidated `analysis/results_v3/phase1_summary.json` with mean,
        std, paired-t-test p-values for each (v1, comparator) pair.

**Decision rule** (run this analysis BEFORE Phase 2):
- If `mean(v1) - mean(random_dr0.5)` on gsm8k is **>= 1.5pp AND p < 0.10**
  paired across seeds → +2.5pp signal confirmed → proceed to Phase 2.
- If gap < 1.5pp or p >= 0.10 → STOP. Write up "v1 ~ random at matched rate;
  saliency adds no measurable selection benefit." This is a smaller paper
  but still publishable as a negative result on saliency for ReLoRA.

Do not skip the decision rule. Push consolidated summary + decision verdict
to `analysis/COMM_AGENT_TO_PI/{date}_phase1_decision.md` before launching Phase 2.

### C.2 Phase 2 (conditional on Phase 1 pass, ~120 GPU-h)

**Add 2 more models × 3 cells × 2 seeds = 12 cells**

models: `olmo2-7b` (instruct), `llama3-8b` (instruct)
        — drop mistral-7b and acereason as you noted earlier.

cells: same 3 (v1, random_dr0.5, relora_baseline)

seeds: `42, 43`

eval: same 6 benchmarks (gsm8k_strict/flex, hellaswag, arc_c, mmlu, ifeval).

If Phase 2 confirms cross-model ordering of the same 3 numbers, the paper
is ready for writing.

## D. Vanilla over-train control (parallel to Phase 1, ~12 GPU-h)

**One critical experiment** that determines whether vanilla LoRA's gsm8k
dominance survives longer training.

Hypothesis: vanilla LoRA over-fits SFT distribution at extended steps,
collapsing on commonsense benchmarks while ReLoRA-with-saliency stays stable.

Setup:
- `lora_vanilla` × qwen3-8b/tulu3-sft × seed 42, 43
- total_steps = **10000** (3.3× current 3000)
- eval at step ∈ {3000, 5000, 7500, 10000} on all 6 benchmarks
- output: `analysis/results_v3/vanilla_overtrain/seed{42,43}/eval_step{N}.json`

Compare to v1_S3pos × 10000 steps × seed 42, 43 (concurrent run).

If vanilla 10k-step hellaswag drops below 76.07% AND v1 10k-step hellaswag
stays above 77%, we have the regularization-stability evidence to defend
the gsm8k cost.

If vanilla 10k-step gsm8k climbs above 87.64% (continues improving), the
"vanilla wins gsm8k" gap widens — story still works as "selection
recovers a fraction of vanilla's gsm8k while keeping hellaswag stable",
but we must report this honestly.

## E. What NOT to do

- Do not re-run v2 IG-FDR. Settle: "two saliency variants explored; v1
  better on gsm8k, v2 better on hellaswag; statistical-rigor estimator
  does not dominate naive estimator on this setup — open question for
  future work."
- Do not chase the v1-saliency-is-bunk angle. Cross-model ρ≈0 is a finding,
  not a paper-worthy critique.
- Do not retrain Exp-1 dr-sweep cells. They are cooked.
- Do not change the optimizer / merge schedule / target_modules / r before
  Phase 1 completes. Variance must come from seed only at this stage.

## F. Compute budget summary

| Phase | scope | est. GPU-h | wall-clock @ 8 GPU |
|---|---|---|---|
| C.1 Phase 1 | qwen3-8b × 3 cells × 3 seeds | 90 | ~24h |
| D vanilla overtrain | 4 cells × 10k steps | 12 (concurrent) | ~24h |
| C.2 Phase 2 | 2 model × 3 cells × 2 seeds | 120 | ~36h |
| **Total** | | **~222** | **~60h on 8 GPUs (~2.5 days)** |

Plus eval: ~30 GPU-h total (vllm batched).

## G. Reporting back

After Phase 1 + D complete (~24h), push:
- `analysis/results_v3/phase1_summary.json` (raw scores, mean, std, p-values)
- `analysis/results_v3/vanilla_overtrain/summary.json`
- `analysis/COMM_AGENT_TO_PI/{date}_phase1_decision.md` with:
  - Phase 1 decision (proceed / stop)
  - 3 headline deltas with 95% CI: v1 vs baseline, v1 vs random, v1 vs vanilla
  - Vanilla over-train trajectory (does hellaswag collapse?)
  - Recommended Phase 2 model order

Wait for PI ack before launching Phase 2.

## H. ACK requested

`ACK_pi_feedback_6_robustness_sweep`

Within 4h: confirm Phase 1 + D launched on which GPUs, expected wall-clock.

Within 24h: Phase 1 + D results pushed.

Within 60h (if Phase 1 passes): Phase 2 results pushed.

---

## Personal note

Excellent work catching the deepspeed import bug in peft.merge_and_unload
and routing around it with the model-graph walk in `--save_merged_final`.
That's the kind of self-driven debug that makes this loop work without my
intervention. Continue this autonomy.

The +9.25pp v1 vs baseline number is solid. The +2.5pp v1 vs random number
is what determines whether the paper is a "method contribution" or a
"finding that selection ~ random for ReLoRA". Phase 1 settles it. Run it.
