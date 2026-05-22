#!/usr/bin/env bash
# Phase D: download Qwen3.5 multimodal models via HF mirror (hf-mirror.com).
# Parallel-friendly. PI directive: do not block, run in bg.
set -e
cd /mnt/cpfs/junlongke/onlinelora
HF=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/hf
LOG=/mnt/cpfs/junlongke/onlinelora/lora_obd/logs/scout/dl_qwen35_seq.log

# HF mirror endpoint (China-friendly, avoids LFS bandwidth issues)
export HF_ENDPOINT=https://hf-mirror.com

dl() {
    local repo=$1
    local slug=$2
    echo "=== [$(date)] downloading $repo -> models/$slug (HF_ENDPOINT=$HF_ENDPOINT) ===" >> $LOG
    mkdir -p models/$slug
    $HF download "$repo" --local-dir models/$slug >> $LOG 2>&1
    echo "=== [$(date)] done $slug, size=$(du -sh models/$slug | awk '{print $1}') ===" >> $LOG
}

# 0.8B already downloaded (1.7GB). Skip.
dl Qwen/Qwen3.5-2B   qwen35-2b
dl Qwen/Qwen3.5-4B   qwen35-4b
dl Qwen/Qwen3.5-9B   qwen35-9b
dl Qwen/Qwen3.5-27B  qwen35-27b
echo "=== ALL QWEN3.5 DOWNLOADS DONE [$(date)] ===" >> $LOG
