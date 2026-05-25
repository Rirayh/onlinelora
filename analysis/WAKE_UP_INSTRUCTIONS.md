# WAKE-UP INSTRUCTIONS (read this first when context is reset)

## Background
PI emergency directive 2026-05-25 ~07:30 UTC: P0 contamination + method bug + saliency calib.
4 commits already pushed to origin/main:
- f9f88ca DIAGNOSIS_v1
- 57038fa keep_B_after_merge (Task 2)
- 490b35c OOD calib + p0_reeval_orchestrator (Task 3 + Task 1)
- dbf8790 exp_v1_orchestrator (Task 4 priority slice)

## Two orchestrators are RUNNING (started 07:37 UTC)
1. PID 2336363: exp_v1_orchestrator (GPU 0,1,2) → 7 qwen3-8b/tulu3-sft trains × seed42
2. PID 2336570: p0_reeval_orchestrator (GPU 3,4,5,6,7) → 67 P0 reevals via vllm-on-merged

Verify alive:
   ps -p 2336363 2336570 -o pid,etime,stat

If dead, restart:
   cd /mnt/cpfs/junlongke/onlinelora/lora_obd
   nohup /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python scripts/exp_v1_orchestrator.py --gpus 0,1,2 --max_parallel 3 --poll 30 > logs/exp_v1/orchestrator.stdout.log 2>&1 &
   nohup /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python scripts/p0_reeval_orchestrator.py --gpus 3,4,5,6,7 --max_parallel 5 --poll 30 > logs/p0_reeval/orchestrator.stdout.log 2>&1 &

## Status snapshot commands
P0 done count:    grep "DONE.*ok=True" logs/p0_reeval/orchestrator.log | wc -l
PI done count:    grep -E "DONE.*(mistral-7b|qwen25-7b|qwen3-8b).*ok=True" logs/p0_reeval/orchestrator.log | wc -l
exp_v1 trained:   ls results/exp_v1/qwen3-8b/tulu3-sft/*/seed42/summary.json 2>/dev/null | wc -l
exp_v1 evaluated: ls results/exp_v1/qwen3-8b/tulu3-sft/*/seed42/lm_eval/*/*.json 2>/dev/null | wc -l

## CRITICAL TODO: exp_v1 needs post-train eval pipeline
exp_v1_orchestrator only does train. After each cell's summary.json appears,
must MANUALLY queue merge+vllm eval. Either:
  (a) Extend exp_v1_orchestrator with post-train eval hook (preferred)
  (b) Run scripts/p0_reeval_orchestrator.py against exp_v1 cells (would need to
      add them to p0_tainted_manifest.json or a new manifest)

Suggested fix: write scripts/exp_v1_eval.py that:
  for each /results/exp_v1/qwen3-8b/tulu3-sft/<label>/seed42/summary.json:
    if not lm_eval/*.json: queue merge+vllm eval

## Push every 4 hours (PI mandate)
First push: ~11:37 UTC. Format:
  git add scripts/ analysis/ logs/SESSION_NOTES.md logs/WAKE_UP_INSTRUCTIONS.md
  git commit -m "P0 re-eval: N/67 done (M PI); exp_v1: K/7 trained, J/7 evaluated"
  git push origin main

DO NOT commit results/ artifacts (jsonl, run.log, lm_eval/*.json, merged/, checkpoints/).
.gitignore already excludes lm_eval_*/ but not lm_eval/. Be careful.

Working tree is dirty with 20 D-status (orchestrator renamed lm_eval/ → lm_eval_PRE_P0_FIX_TAINTED/).
DO NOT git add those. Use targeted `git add scripts/ analysis/ logs/*.md`.

## Hard goal next push: ≥5 PI P0 evals + ≥3 exp_v1 trained

## Key paths
- Code: /mnt/cpfs/junlongke/onlinelora/lora_obd/scripts/
- Manifest: /mnt/cpfs/junlongke/onlinelora/lora_obd/analysis/p0_tainted_manifest.json
- Orch logs: logs/p0_reeval/orchestrator.log + logs/exp_v1/orchestrator.log
- Train logs: results/stage3_v2/<m>/<d>/<method>/seed42/run.log
- Eval logs: logs/p0_reeval/<name>.eval.log
- Python envs:
    /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python  (Qwen3 dense)
    /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python (Qwen3.5 hybrid + lm_eval+vllm)

## Bugs/edge cases to watch
- vllm needs CUDA_HOME=/usr/local/cuda-12 (handled in orchestrator)
- merge_adapter.py runs on CPU (CUDA_VISIBLE_DEVICES="")
- HYBRID models qwen35-* skip merge, use HF + peft= (12 hybrid cells in queue)
- vllm OOM on merged_dir if gpu_memory_utilization too high. Already at 0.85 max_model_len=4096.
- If a P0 reeval fails (ok=False), check logs/p0_reeval/<name>.eval.log for traceback.
