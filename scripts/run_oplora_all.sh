#!/usr/bin/env bash
# Run OPLoRA analysis on all (model, method, dataset) cells whose adapter exists.
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY=/mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python
OUT_DIR=$ROOT/analysis/oplora/jsons
LOG_DIR=$ROOT/analysis/oplora/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

declare -A BASE
BASE[qwen3-1p7b]=/mnt/cpfs/junlongke/onlinelora/models/qwen3-1p7b
BASE[qwen3-4b]=/mnt/cpfs/junlongke/onlinelora/models/qwen3-4b
BASE[qwen35-0p8b]=/mnt/cpfs/junlongke/onlinelora/models/qwen35-0p8b
BASE[qwen35-2b]=/mnt/cpfs/junlongke/onlinelora/models/qwen35-2b

MODELS=(qwen35-0p8b qwen3-1p7b qwen35-2b qwen3-4b)
METHODS=(lora_vanilla dora relora_diag_gated_S3pos)
DATASETS=(tulu3-sft)

for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    for me in "${METHODS[@]}"; do
      seed_dir="$ROOT/results/stage3_v2/$m/$d/$me/seed42"
      if [ ! -f "$seed_dir/adapter/adapter_model.safetensors" ]; then
        echo "[skip] $m/$d/$me (no adapter)"
        continue
      fi
      out_json="$OUT_DIR/${m}__${d}__${me}.json"
      if [ -f "$out_json" ] && [ "$(stat -c %s "$out_json")" -gt 1000 ]; then
        echo "[skip] $m/$d/$me (already analyzed)"
        continue
      fi
      log="$LOG_DIR/${m}__${d}__${me}.log"
      echo "[run]  $m/$d/$me -> $out_json"
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 19 "$PY" -u "$ROOT/scripts/oplora_analysis.py" \
        --model "$m" --dataset "$d" --method "$me" \
        --base "${BASE[$m]}" \
        --seed_dir "$seed_dir" \
        --out_dir "$OUT_DIR" \
        > "$log" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "  FAIL rc=$rc (see $log)"
      else
        echo "  ok"
      fi
    done
  done
done
echo "[done] all OPLoRA analyses complete"
