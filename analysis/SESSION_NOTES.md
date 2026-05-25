# SESSION STATE — auto-saved (08:35 UTC 2026-05-25)

## Last 4 commits (all pushed to origin/main)
- f9f88ca Commit A: DIAGNOSIS_v1 + scripts/find_p0_tainted_evals.py
- 57038fa Commit B: Task 2 keep_B_after_merge + relora_diag_gated_S3pos_keepB method
- 490b35c Commit C: Task 3 OOD saliency calib (--saliency_calib_set) + scripts/p0_reeval_orchestrator.py
- dbf8790 Commit D: scripts/exp_v1_orchestrator.py (Task 4 priority slice 7 methods)

## RUNNING PROCESSES (started 07:37 UTC, ~1hr in as of 08:35)
### exp_v1_orchestrator (PID 2336363) — GPU 0,1,2 — max_parallel=3
3 concurrent stage3_run.py training jobs at step ~300/3000 (qwen3-8b/tulu3-sft, ~3-4hr each):
  - GPU 0: lora_vanilla        (step 300, train_loss=1.32, elapsed 3363s)
  - GPU 1: relora_baseline     (step 300, train_loss=1.32, elapsed 3356s)
  - GPU 2: relora_random_drop  (step 300, train_loss=1.32, elapsed 3361s)
Queued (4 remain, will start as above finish):
  relora_S3pos, relora_S3pos_keepB, relora_S3pos_keepB_calibgsm8k, cola
ETA each: ~3.0hr (at 11s/step extrapolation = 8.4hr from start = ~16:00 UTC)
Output: results/exp_v1/qwen3-8b/tulu3-sft/<label>/seed42/

### p0_reeval_orchestrator (PID 2336570) — GPU 3,4,5,6,7 — max_parallel=5
67 cells. Status as of 08:35:
  Completed (5): all llama3-8b (metamathqa-10k+tulu3-sft cola/baseline/S3pos/random_drop)
  Running (5): mostly llama3-8b/mistral-7b. mistral-7b PI cells just started 08:27/08:29.
  Per-cell time: ~30min average (merge ~1min CPU + vllm eval ~25-30min)
  PI tally: mistral-7b done=0, qwen25-7b done=0, qwen3-8b done=0  ← needs 5 PI by next 4hr push

## PI USER DIRECTIVE (urgent — RUN NOW)
1. Prioritize 20 PI target cells (mistral-7b/qwen25-7b/qwen3-8b)
2. After 5 PI done, START exp_v1 — but exp_v1 ALREADY running.
3. Push partial results EVERY 4 HOURS. Format: "P0 re-eval: N/67 done; exp_v1: M/7 done"
4. If GPU full by stage3_v2 daemon, KILL it. (No daemon — orchestrators only.)
5. Hard goal next push (4hr): ≥5 PI P0 reevals + ≥3 exp_v1 cells

## P0 orchestrator processes alphabetically. Need to verify PI cells get done in time.
Current rate: 5 cells in 1hr. 67 / 5 = ~13.4 hr total. PI mistral starting now → all 20 PI done in ~4hr.
At 4hr mark (~11:37 UTC):
  P0 reeval expected: ~25 done (16-20 PI cells should be done)
  exp_v1 expected: 0 cells (3000 steps ~3hr, plus eval queue ~30min). First 3 may complete by 11:30.

## Next 4-hr push schedule (cron-style)
- 11:37 UTC (1st checkpoint)
- 15:37 UTC
- 19:37 UTC
- 23:37 UTC

## Push procedure (when waking up at next checkpoint)
1. Tally completed: 
   `grep "DONE.*ok=True" logs/p0_reeval/orchestrator.log | wc -l`
   `ls results/exp_v1/qwen3-8b/tulu3-sft/*/seed42/summary.json | wc -l`
2. Verify exp_v1 cells have lm_eval_v3/ subdirs (need to launch evals manually after train done — exp_v1_orchestrator does NOT auto-eval)
3. **STILL TODO**: Add eval-after-train hook into exp_v1_orchestrator (or manually trigger merge+vllm for completed exp_v1 cells)
4. Commit format: "P0 re-eval: N/67 done; exp_v1: M/7 trained, K/7 evaluated"
5. NEVER include results/ artifacts in commits (they're gitignored or noise; just commit code/scripts/analysis)

## Files modified/created in this session (all pushed)
- analysis/DIAGNOSIS_v1.md
- analysis/p0_tainted_manifest.json
- scripts/find_p0_tainted_evals.py
- scripts/p0_reeval_orchestrator.py
- scripts/exp_v1_orchestrator.py
- scripts/stage3_run.py (Task 2 keep_B + Task 3 saliency_calib_set)

## Critical missing piece: exp_v1 cells need lm_eval_v3 after training
exp_v1_orchestrator only launches train. After summary.json appears for an exp_v1 cell,
I need to:
  a) merge adapter (CPU): scripts/merge_adapter.py --base ... --adapter checkpoints/best --out merged/
  b) vllm eval: lm_eval --model vllm pretrained=merged/ ... --output_path lm_eval/
This must happen for each of 7 exp_v1 cells. Plan: extend exp_v1_orchestrator with post-train hook,
OR just feed exp_v1 cells into p0_reeval_orchestrator's job list once they have summary.json.

## Working tree noise (not for commits)
- 20 D-status results/stage3_v2/.../lm_eval/results_*.json (orchestrator renamed dirs, git tracks old paths)
- Various .jsonl train artifacts in stage3_v2 (qwen3-14b mostly)
- DO NOT add these to any commit. Only commit scripts/, analysis/, code changes.
