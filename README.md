# lora-obd recycling

Diagnostic-validation-guided LoRA prune/merge/rotate pipeline.

See `STATUS.md` for current stage and `../lora_obd_handover_for_gpu_agent.md` for the execution plan.

## Quick commands

```bash
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python

# Stage 0: smoke (SST-2, RoBERTa-base + LoRA)
CUDA_VISIBLE_DEVICES=0 $PY scripts/stage0_smoke.py --config configs/stage1_sst2.yaml --smoke

# Stage 1: 3 tasks in parallel
CUDA_VISIBLE_DEVICES=0 $PY scripts/stage1_run.py --config configs/stage1_sst2.yaml > logs/s1_sst2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/stage1_run.py --config configs/stage1_mrpc.yaml > logs/s1_mrpc.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY scripts/stage1_run.py --config configs/stage1_rte.yaml  > logs/s1_rte.log  2>&1 &
wait
$PY scripts/stage1_plot.py --aggregate
```
