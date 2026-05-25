# SESSION STATE — auto-saved (09:38 UTC 2026-05-25)

## All commits pushed to origin/main
- f9f88ca Commit A: DIAGNOSIS_v1 + scripts/find_p0_tainted_evals.py
- 57038fa Commit B: Task 2 keep_B_after_merge
- 490b35c Commit C: Task 3 OOD calib + p0_reeval_orchestrator.py
- dbf8790 Commit D: exp_v1_orchestrator.py
- f1af025 Commit E: SESSION_NOTES + WAKE_UP + exp_v1_eval.py
- 97ac5e0 COMM: GPU5 status reply
- 936c0b1 (PI) directives 2026-05-25 11:00

## Three orchestrators RUNNING (verify alive)
1. PID 2336363 exp_v1_orchestrator     GPU 0/1/2 max=3 (training queue)
2. PID 2336570 p0_reeval_orchestrator  GPU 3/4/5/6/7 max=5 (eval queue)
3. PID 2365272 exp_v1_eval             GPU 3-7 max=1 (watch+queue exp_v1 evals)
   (NEW, started 09:37 UTC per PI §3 — fires when summary.json appears)

Verify alive:
   ps -p 2336363 2336570 2365272 -o pid,etime,stat

If dead, restart commands in WAKE_UP_INSTRUCTIONS.md.

## Status as of 09:38 UTC (~2hr in)
- P0 reeval done: 9/67 (PI=0 done, 5 mistral PI cells running)
- exp_v1 trained: 0/7 (3 running step ~400/3000)

## PI directive 2026-05-25 11:00 acknowledgment
File: analysis/COMM_PI_TO_AGENT/2026-05-25_1100_pi_directives.md
Key changes I applied:
- §3: launched exp_v1_eval.py --watch mode (PID 2365272)
- §7: NO HF backend → patched p0_reeval_orchestrator.py to DEFER hybrid Qwen3.5
      cells. (Code updated, but running orchestrator still has old code; new code
      will activate on restart. Hybrid cells are alphabetically last in queue,
      so they won't be reached for hours; if I get there before user needs
      them, I can restart orchestrator.)
- §4: OLD vs NEW table format required in next push commit
- §6: bit-identical S3pos/random_drop NEW results = critical flag

## Next push: 11:37 UTC (4hr cycle)
Commit message must include:
- "ACK: PI directives 2026-05-25 11:00"
- "P0 re-eval: N/67 done; exp_v1: M/7 trained, K/7 evaluated"
- OLD vs NEW table for any mistral PI cells finished
- Critical flag if S3pos/random_drop NEW are still bit-identical

## TODO before 11:37 push
- [ ] Wait for ≥5 mistral PI cells to finish vllm eval (~10:00 UTC ETA)
- [ ] Generate OLD vs NEW table from results_*.json (gsm8k strict + flex)
- [ ] Check if any exp_v1 cell finished training & got eval triggered
- [ ] Detect+document any GPU failures from logs

## How to extract OLD vs NEW
OLD = lm_eval_PRE_P0_FIX_TAINTED/.../results_*.json
NEW = lm_eval/.../results_*.json (latest mtime)
Key: results > gsm8k > strict-match (exact_match,strict-match) and exact_match (flexible-extract)

Example helper command:
  for cell in mistral-7b/metamathqa-10k/{relora_baseline,relora_diag_gated_S3pos,relora_diag_gated_S3neg,relora_random_drop,relora_train_gated}; do
    base=results/stage3_v2/$cell/seed42
    old=$base/lm_eval_PRE_P0_FIX_TAINTED
    new=$base/lm_eval
    echo "=== $cell ==="
    echo "OLD:"
    find $old -name "results_*.json" 2>/dev/null | head -1 | xargs -r jq '.results.gsm8k'
    echo "NEW:"
    find $new -name "results_*.json" 2>/dev/null | head -1 | xargs -r jq '.results.gsm8k'
  done

## Hybrid cells deferred (12 cells, NOT counted in 67 vllm queue anymore)
qwen35-{0p8b,2b,4b,9b}/{tulu3-sft,metamathqa}/* — to be addressed after vllm-supported sweep done.
