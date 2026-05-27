# CRITICAL AGENT STATE — 2026-05-27 07:35 UTC

## Active task
Per PI #5b: re-train 6 cells with `--save_merged_final` to fix the P0 ckpt bug. Smoke run on GPU 0 succeeded for training (final_val=1.4096, merge fired, drop_rate=0.495), but `--save_merged_final` block FAILED with `MissingCUDAException: CUDA_HOME does not exist`.

## Fix just applied to scripts/stage3_run.py
Replaced the failing PEFT `merge_and_unload()` path with a direct LoRA-wrapper stripping approach. Walks model, replaces each `LoraLayer` (has `base_layer` attr) with its `base_layer`, then calls `transformers.save_pretrained` on the unwrapped transformer. Bypasses PEFT's deepspeed-aware save chain.

**MUST RE-RUN SMOKE** to verify the fix works. Smoke was at:
- output dir: /mnt/cpfs/junlongke/onlinelora/lora_obd/results/smoke_pi5b/seed42
- training already done; just need to re-run save_merged_final block
- BUT: the smoke run process exited (PID 2705950 dead). model state lost.
- → need to re-run smoke from scratch (~13min) OR write a tiny standalone script that loads the latest peft adapter ckpt and does the unwrap-save logic to test.

## Best path forward
Since training already takes 12min for the 50-step smoke and we wasted 13min, just re-launch smoke directly:
```
cd /mnt/cpfs/junlongke/onlinelora/lora_obd
rm -rf results/smoke_pi5b
nohup /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python scripts/smoke_pi5b_patch.py > logs/smoke_pi5b_runner.log 2>&1 &
```
Wait ~13min, check `results/smoke_pi5b/seed42/merged_final/config.json` exists.

## After smoke passes
Launch 6-cell re-train via `scripts/s2_pi5b_v3_orchestrator.py`:
```
nohup /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python scripts/s2_pi5b_v3_orchestrator.py > logs/s2_pi5b_v3/orch.log 2>&1 &
```
Default uses GPUs 1-7 (excludes GPU 0). Wall-clock ~10h.

The 6 cells:
1. v1_S3pos (relora_diag_gated_S3pos)
2. v2_S3pos_IG_FDR (relora_diag_gated_S3pos + saliency_v2)
3. random_dr0.5
4. random_dr0.3
5. relora_baseline
6. lora_vanilla

## Key context
- Local commit pushed: 921d66c (ACK_pi_feedback_5_eval_pipeline_HALT)
- Remote commit: c085534 PI #5b (REVERSAL: method WORKS, signal hidden by P0 bug)
- All 8 GPUs free
- Pending push: stage3_run.py patch, smoke_pi5b_patch.py, s2_pi5b_v3_orchestrator.py

## Files written
- scripts/stage3_run.py (added --save_merged_final flag, custom unwrap-save logic at end)
- scripts/smoke_pi5b_patch.py (50-step micro-smoke test)
- scripts/s2_pi5b_v3_orchestrator.py (6-cell re-train launcher)
- analysis/audits/2026-05-27_adapter_hash_audit.txt (Action B+E results, ALREADY PUSHED)
- analysis/COMM_AGENT_TO_PI/2026-05-27_agent_ack_pi5_eval_halt.md (ALREADY PUSHED)

## Next concrete actions
1. git stash uncommitted, git pull (in case PI pushed more), git stash pop
2. Sanity check stage3_run.py syntax: `/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python -c "import ast; ast.parse(open('scripts/stage3_run.py').read()); print('OK')"`
3. Re-run smoke: rm -rf results/smoke_pi5b && launch
4. Wait ~13min, verify merged_final/config.json + safetensors exist
5. Run gen sanity check (already part of smoke_pi5b_patch.py)
6. If smoke passes: ACK + commit + push patch
7. Launch 6-cell re-train on GPUs 1-7

## Patch verification commands
```bash
# Check patch is in place:
grep -n "save_merged_final\|replaced.*LoRA wrappers" scripts/stage3_run.py
# Should show the new flag definition + the unwrap loop
```

## env paths
- training: /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
- vllm eval: /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python
- model: /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B
