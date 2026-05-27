# ACK_pi_feedback_5_eval_pipeline_HALT

**Date**: 2026-05-27 06:55 UTC
**Agent reply to**: `2026-05-27_pi_feedback_5_HALT_eval_pipeline_bug.md`

## Status of HALT (Action A)

✅ No eval was running at the time PI #5 landed. All 8 GPUs were free as of
06:35 UTC. No kill required.

## Action B: adapter hash audit — DONE

See `analysis/audits/2026-05-27_adapter_hash_audit.txt`.

**Headline**: All 6 adapters have distinct sha256 hashes (NOT bit-identical),
but max-abs-diff = 1.5e-2 to 2.3e-2 and L2 norm ratio diff/ref = 22-45%.
These diffs are NOT cuDNN noise — they come from `best_step` varying
(250 vs 500), saliency-calib RNG drift (calib_n=64 vs 256), and code-drift
(3 different commit hashes across runs).

**PI's deeper claim is fully confirmed**: every saved adapter IS the
pre-merge state (best_step ∈ {250, 500} < first merge at step 750), so
drop policy NEVER affects what we evaluate. The pairwise adapter diffs
exist but encode nothing about the method.

## Action E: scoreboard provenance — DONE

PI scoreboard targets traced via repo grep:

```
analysis/oplora/jsons/qwen35-2b__tulu3-sft__lora_vanilla.json    → 0.8643
analysis/oplora/jsons/qwen35-0p8b__tulu3-sft__relora_diag_gated_S3pos.json → 0.8640 / 0.7732
analysis/oplora/jsons/qwen3-1p7b__tulu3-sft__relora_diag_gated_S3pos.json  → 0.8642 / 0.6932
```

No `qwen3-8b__tulu3-sft__*.json` exists. Scoreboard came from DIFFERENT
base models (qwen35-2b / qwen35-0p8b / qwen3-1p7b), not qwen3-8b.

→ The 80% gsm8k floor we observe IS correct for qwen3-8b. There is no
"lost gap" to chase; the scoreboard was a cross-model transcription.

## Patch proposal: Option 3 (preferred per PI)

Goal: remove the `best/` + `copy-from-best` P0 hack; eval at merge boundaries
+ final; take max over merge-event scores.

### Concrete changes to `scripts/stage3_run.py`

1. **DELETE L1413-L1433** (the `if save_adapter` copy-from-best block).
   Adapter dir is no longer the eval entry point.

2. **DELETE L1069-L1078** (the `if vl < best_val_loss → save best/`
   branch). Pre-merge "best" tracking is misleading; we keep `best_val_loss`
   as a metric in summary.json but don't ckpt on it.

3. **ADD** at each merge event (after `merge_and_reset_lora`):
   - The merge mutates `base_linear.weight` in-place (L517 of stage3_run.py).
   - Snapshot the merged base (full PEFT model with merged weights folded
     in) at the END of training only (the final merge at step 3000).
   - Save via `model.merge_and_unload()` then `save_pretrained()` to a
     directory named `merged_final/`. ~16GB per cell.
   - Avoid saving at every merge boundary (would be 4×16GB per cell).

4. **ADD** in-process eval at every merge boundary:
   - After `merge_and_reset_lora`, call lm-eval API on a 200-sample subset
     of gsm8k (strict-match) inline. ~2min per merge event.
   - Append to `merge_eval_scores.jsonl` in `out_root`.
   - Best merge event = argmax over these scores.

5. **CHANGE** `--save_adapter` semantics: replace with `--save_merged_final`
   (default True). When True, save full merged model (for offline lm-eval).
   When False, only the in-process scores are produced.

6. **summary.json gains**:
   - `merge_eval_scores`: list of `{event_idx, step, gsm8k_strict_200}`
   - `best_merge_event`: argmax
   - `final_merged_dir`: path to merged_final/ if saved
   - keep `best_val_loss` / `best_step` as a stability sanity metric, but
     mark with comment "pre-merge only; not method indicator".

### Smoke plan after patch (PI sign-off needed)

1. Single retrain on qwen3-8b/tulu3-sft, GPU 0:
   - `--method relora_random_drop --random_drop_rate 0.5`
   - `--total_steps 3000 --merge_every 750 --seed 42`
   - `--eval_at_merge_boundaries --save_merged_final`
   - ~9.5h training + 4×2min in-process evals = ~10h total
2. Offline lm-eval on `merged_final/` (full gsm8k, hellaswag, arc_c).
3. Acceptance:
   - In-process scores at events 1/2/3/4 must show >2pp spread → confirms
     method effect is now visible.
   - Offline eval on `merged_final/` should match (within 0.5pp) the
     event-4 in-process score.
4. If <2pp spread: method has no signal, write up as negative result.

### Disk / cost budget

- merged_final/ per cell: ~16GB
- merge_eval_scores.jsonl: <10KB
- 200-sample gsm8k subset eval cost: ~2min × 4 events = 8min/cell
  (vs 9.5h training, 1.4% overhead)

### Open questions / sign-off needed before patch

1. Confirm Option 3 (above) vs Option 2 (fold AB into base before each
   `save_pretrained`)? Option 2 keeps adapter ckpts at ~166MB each but
   requires teaching offline lm-eval to load a "merged adapter" config.
   I prefer Option 3 (one final merged base, in-process scores at
   boundaries) because it cleanly separates "training-time signal" from
   "deployment-time score".

2. Should I delete ALL existing `lm_eval/` JSON results (Exp-1, S2, S3
   tie-break, v1_recheck)? They're meaningless under new semantics. Or
   keep with a `INVALID_PRE_MERGE_BUG.flag` marker per-dir?

3. Confirm: do you want me to run the smoke retrain BEFORE patching
   `exp_drop_rate_orchestrator.py` / S2.5 / S3 orchestrators? My read of
   the directive: yes, smoke first; only re-launch sweep if smoke shows
   >2pp signal.

4. v2_full / v1_recheck / tie-break statistical findings (saliency
   estimator behavior) remain VALID per PI — they don't depend on eval
   pipeline. I'll keep `dropped_components.jsonl` / `saliency_at_merge.jsonl`
   and continue any analysis that lives entirely in those (e.g.
   sig_frac trajectory plot, v1 Bernoulli rejection visualization).

Awaiting sign-off on (1), (2), (3) before applying the patch.

## Summary checklist

- [x] HALT: confirmed (no eval running)
- [x] Action B: adapter hash audit, committed
- [x] Action E: scoreboard provenance traced
- [x] Patch proposal: this doc, Option 3
- [ ] PI sign-off on patch direction
- [ ] Patch applied to stage3_run.py
- [ ] Smoke retrain launched
- [ ] Smoke eval shows >2pp signal
- [ ] Sweep re-launched
