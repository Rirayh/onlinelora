#!/usr/bin/env bash
# Phase C scout queue runner.
# Auto-fills 8 GPUs: when any GPU drops below MEM_THRESHOLD_MB free, do nothing
# (it's busy training); when a GPU has more than IDLE_THRESHOLD_MB free, launch next.
#
# Queue: 50 SFT cells = 5 new models * 2 datasets * 5 methods.
# Each (model, dataset, method) outputs to:
#   results/stage3_v2/$MODEL/$DATASET/$METHOD/seed42/
# Skips if summary.json already exists.
#
# Stop condition: queue empty AND all 8 GPUs idle.
set -u
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
ROOT=/mnt/cpfs/junlongke/onlinelora/lora_obd
LOGDIR=$ROOT/logs/scout
mkdir -p "$LOGDIR"
cd "$ROOT"

# (model_key | model_path | attn)
declare -A MP=(
  [gemma3-12b]=/mnt/cpfs/junlongke/onlinelora/models/gemma-3-12b-it
  [llama3-8b]=/mnt/cpfs/public_data/public_model/Meta-Llama-3-8B
  [r1-distill-7b]=/mnt/cpfs/junlongke/onlinelora/models/R1-Distill-Qwen-7B
  [acereason-7b]=/mnt/cpfs/junlongke/onlinelora/models/AceReason-Nemotron-7B
  [olmo2-7b]=/mnt/cpfs/junlongke/onlinelora/models/OLMo-2-7B
)
declare -A ATTN=(
  [gemma3-12b]=eager
  [llama3-8b]=sdpa
  [r1-distill-7b]=sdpa
  [acereason-7b]=sdpa
  [olmo2-7b]=sdpa
)

MODELS=(olmo2-7b acereason-7b llama3-8b r1-distill-7b gemma3-12b)
DATASETS=(metamathqa-10k tulu3-sft)
METHODS=(lora_vanilla relora_baseline relora_diag_gated_S3pos relora_random_drop dora)

# Build queue (skip cells already complete)
QUEUE=()
for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      OUT="results/stage3_v2/$MODEL/$DATASET/$METHOD/seed42"
      [[ -f "$OUT/summary.json" ]] && continue
      QUEUE+=("$MODEL|$DATASET|$METHOD")
    done
  done
done
echo "Queue: ${#QUEUE[@]} cells"

# Helper: pick a free GPU (free mem > 50GB)
pick_gpu() {
  while IFS=, read -r idx free; do
    free=$(echo "$free" | tr -d ' MiB')
    if (( free > 50000 )); then
      echo "$idx"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader)
  return 1
}

# Launch a cell on a given GPU
launch_cell() {
  local gpu=$1 model=$2 dataset=$3 method=$4
  local mp="${MP[$model]}"
  local attn="${ATTN[$model]}"
  local out="results/stage3_v2/$model/$dataset/$method/seed42"
  mkdir -p "$out"
  local log="$LOGDIR/${model}-${dataset}-${method}.log"
  # dora needs only 800 steps (small ckpts); other methods 3000
  local steps=3000
  [[ "$method" == "dora" ]] && steps=800
  echo "[$(date +%H:%M:%S)] LAUNCH GPU=$gpu $model/$dataset/$method steps=$steps -> $log"
  nohup env CUDA_VISIBLE_DEVICES=$gpu $PY scripts/stage3_run.py \
    --model_path "$mp" --model_key "$model" --dataset "$dataset" \
    --method "$method" --total_steps $steps \
    --merge_every 500 --eval_every 250 --ckpt_every 50 \
    --saliency_max_seq_len 512 --attn_implementation "$attn" \
    --save_adapter --seed 42 \
    --out_root "$out" \
    > "$log" 2>&1 &
}

# Main loop
QPTR=0
while (( QPTR < ${#QUEUE[@]} )); do
  GPU=$(pick_gpu) || true
  if [[ -z "${GPU:-}" ]]; then
    sleep 60
    continue
  fi
  IFS='|' read -r M D MET <<< "${QUEUE[$QPTR]}"
  launch_cell "$GPU" "$M" "$D" "$MET"
  QPTR=$((QPTR+1))
  sleep 45  # let the new job grab memory before pick_gpu re-reads
done

echo "All ${#QUEUE[@]} cells launched. Waiting for them to finish..."
wait
echo "Phase C scout complete."
