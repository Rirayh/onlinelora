# Agent Resume Plan — 2026-05-27 07:00 UTC

## CRITICAL: PI #5 P0 SHOWSTOPPER landed

PI feedback #5 (commit `bc752cc`) says: ALL prior lm-eval was on pre-merge LoRA snapshots. Drop policy never affected the saved state because best_step ∈ {250, 500} < first merge at step 750. PI #4 (commit `913c0db`) is RETRACTED.

**Required by PI**:
- ACK `ACK_pi_feedback_5_eval_pipeline_HALT`
- Action B: adapter hash audit → `analysis/audits/2026-05-27_adapter_hash_audit.txt`  ✅ DONE
- Action E: scoreboard ckpt provenance trace → ✅ DONE (cross-model error: 86.43 came from qwen35-2b/qwen35-0p8b/qwen3-1p7b, NOT qwen3-8b)
- Decide Option 1/2/3 → proposed Option 3
- DO NOT retrain or eval before PI sign-off on patch

## Files written this session (need commit + push)

1. `analysis/audits/2026-05-27_adapter_hash_audit.txt` — hash audit + provenance trace
2. `analysis/COMM_AGENT_TO_PI/2026-05-27_agent_ack_pi5_eval_halt.md` — ACK + patch proposal Option 3

## Key technical findings

**Hash audit**: 6 adapters distinct hashes, max_abs_diff 1.5e-2 to 2.3e-2, L2 ratio diff/ref 22-45%. NOT cuDNN noise. Diffs come from best_step (250 vs 500), saliency_calib_n (64 vs 256), commit_hash drift (547f8c9, bf5d452, 87da7d4).

**PI hypothesis confirmed**: pre-merge state, drop policy never affects saved adapter.

**Scoreboard provenance**: Found in `analysis/oplora/jsons/qwen35-2b__tulu3-sft__lora_vanilla.json` (0.8643), `qwen35-0p8b__tulu3-sft__relora_diag_gated_S3pos.json` (0.8640/0.7732), `qwen3-1p7b__tulu3-sft__relora_diag_gated_S3pos.json` (0.8642). NO qwen3-8b prior eval exists. Scoreboard = cross-model transcription error.

## Patch proposal (Option 3) waiting for PI sign-off

`scripts/stage3_run.py` changes:
1. DELETE L1413-L1433 (copy-from-best block)
2. DELETE L1069-L1078 (best_val ckpt branch); keep best_val_loss as metric only
3. ADD merge_and_unload() + save_pretrained() of full merged model at FINAL merge → `merged_final/` (~16GB per cell)
4. ADD in-process lm-eval (gsm8k 200-sample strict-match) at each merge boundary → `merge_eval_scores.jsonl`
5. CHANGE --save_adapter → --save_merged_final
6. summary.json gains: merge_eval_scores, best_merge_event, final_merged_dir

## Smoke plan after PI sign-off

Single retrain: --method relora_random_drop --random_drop_rate 0.5 --total_steps 3000 --merge_every 750 --seed 42 --eval_at_merge_boundaries --save_merged_final on GPU 0. ~10h. Acceptance: >2pp spread in event scores.

## What stays valid from prior work
- v1 ρ≈0 cross-model (independent of eval)
- v2 IG-FDR V-shape sig_frac trajectory (independent)
- v1 Bernoulli rejection >10σ per event (independent)
- §5 schedule sanity 12 events ±5% (independent)
- effective rank, condition number trajectories (independent)

What is INVALIDATED: every lm_eval JSON result from `relora_*` runs with the P0-fix copy-from-best path.

## Current GPU state (06:35 UTC)
All 8 GPUs FREE. Tie-break done (commit 5f53503). v2_full done. v1_recheck done. s5 sanity done.

## Push planned: NOW (after writing this state)

Commit body should include:
- ACK_pi_feedback_5_eval_pipeline_HALT
- Action B + E results summary
- Option 3 patch proposal status (proposed, awaiting sign-off)

## Open questions for PI (in agent ACK doc)
1. Option 3 vs Option 2 confirmation
2. Delete vs flag existing lm_eval JSON results
3. Smoke first before sweep re-launch
4. Keep dropped_components/saliency_at_merge analyses (independent of eval pipeline)

## env paths
- training: /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
- vllm eval: /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python
- model: /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B

## Prior ACK chain
- bf5d452: ACK_pi_feedback_s1
- 040e404: ACK_pi_feedback_s2_v2smoke
- 87da7d4: ACK_pi_feedback_pre_position_s3
- 54c9b07: S3_ROUTE=E_ambiguous_tiebreak
- afa7880: v1_recheck DONE + v2_full DONE + s5 sanity PASS
- 5f53503: S3 tie-break ALL 4 CELLS DONE
- (NEXT) ACK_pi_feedback_5_eval_pipeline_HALT
