#!/usr/bin/env bash
# F3: 8-task fan-out across 8 GPUs for ONE adapter.
#
# Usage:
#   bash scripts/run_lmeval_8parallel.sh <adapter_path> <base_model_path> [out_subdir]
#
# Notes:
#   - Each task gets one GPU. Tasks chosen to cover SFT-relevant capabilities
#     per `09_cloud_agent_followup_lmeval_expansion.md` §4.
#   - --log_samples is mandatory (required by F4 bootstrap_ci.py).
#   - Results go under <adapter_path>/lm_eval/<task_name>/.
#   - HumanEval is gated by HF_ALLOW_CODE_EVAL=1 (lm-eval default off).
set -euo pipefail

ADAPTER="${1:?adapter path required}"
BASEMODEL="${2:?base model path required}"
OUT_SUBDIR="${3:-lm_eval}"

if [[ ! -d "$ADAPTER" ]]; then
  echo "ERROR: adapter not found: $ADAPTER" >&2; exit 1
fi
if [[ ! -d "$BASEMODEL" ]]; then
  echo "ERROR: base model not found: $BASEMODEL" >&2; exit 1
fi

PY="${PY:-/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python}"
OUTDIR="$ADAPTER/$OUT_SUBDIR"
mkdir -p "$OUTDIR"
LOGDIR="$OUTDIR/_logs"
mkdir -p "$LOGDIR"

# Tasks aligned to GPUs 0-7. Few-shot per §4.2.
TASKS=(gsm8k mmlu mmlu_pro bbh hendrycks_math humaneval ifeval truthfulqa_mc1)
declare -A FEWSHOT=(
  [gsm8k]=5
  [mmlu]=5
  [mmlu_pro]=5
  [bbh]=3
  [hendrycks_math]=4
  [humaneval]=0
  [ifeval]=0
  [truthfulqa_mc1]=0
)

echo "=== run_lmeval_8parallel ==="
echo "ADAPTER  : $ADAPTER"
echo "BASEMODEL: $BASEMODEL"
echo "OUT      : $OUTDIR"
echo

PIDS=()
for i in "${!TASKS[@]}"; do
  T="${TASKS[$i]}"
  FS="${FEWSHOT[$T]}"
  TASK_OUT="$OUTDIR/$T"
  mkdir -p "$TASK_OUT"
  LOG="$LOGDIR/$T.log"

  ENV_PREFIX=""
  if [[ "$T" == "humaneval" ]]; then
    ENV_PREFIX="HF_ALLOW_CODE_EVAL=1"
  fi

  echo "[GPU $i] task=$T fewshot=$FS  -> $LOG"
  CMD="CUDA_VISIBLE_DEVICES=$i $ENV_PREFIX $PY -m lm_eval \
    --model hf \
    --model_args pretrained=$BASEMODEL,peft=$ADAPTER,dtype=bfloat16,attn_implementation=sdpa,trust_remote_code=True \
    --tasks $T \
    --num_fewshot $FS \
    --batch_size 4 \
    --log_samples \
    --output_path $TASK_OUT"
  bash -c "$CMD" > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo
echo "Launched ${#PIDS[@]} jobs: ${PIDS[*]}"
echo "Waiting for all tasks to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "WARN: pid $pid exited non-zero"
    FAILED=$((FAILED+1))
  fi
done
echo
echo "DONE. failed=$FAILED / ${#PIDS[@]}"
exit $FAILED
