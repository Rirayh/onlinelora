#!/usr/bin/env bash
# Sequential download of remaining Qwen3 dense models for Phase D.
# Started: 2026-05-21
set -e
cd /mnt/cpfs/junlongke/onlinelora
HF=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/hf
LOG=/mnt/cpfs/junlongke/onlinelora/lora_obd/logs/scout/dl_qwen3_seq.log

dl() {
    local repo=$1
    local slug=$2
    echo "=== [$(date)] downloading $repo -> models/$slug ===" >> $LOG
    mkdir -p models/$slug
    $HF download $repo --local-dir models/$slug >> $LOG 2>&1
    echo "=== [$(date)] done $slug, size=$(du -sh models/$slug | awk '{print $1}') ===" >> $LOG
}

dl Qwen/Qwen3-4B-Instruct-2507 qwen3-4b
dl Qwen/Qwen3-14B              qwen3-14b
dl Qwen/Qwen3-32B              qwen3-32b
echo "=== ALL QWEN3 DENSE DOWNLOADS DONE [$(date)] ===" >> $LOG
