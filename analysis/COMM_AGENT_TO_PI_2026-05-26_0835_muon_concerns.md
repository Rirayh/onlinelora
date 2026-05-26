# ACK + concerns: PI 2026-05-26 Muon Decoupling

## ACK

ACK: PI directives 2026-05-26 (Muon decoupling)

Read fully. Goals understood. STOP/DEFER list noted.

## Current state (2026-05-26 ~08:35 UTC, just before launch)

GPU usage:
- GPU 0: 30 GB (cola training, PID 2485357, 7h elapsed, step ~1800/3000) — last exp_v1 cell from previous task, will finish in ~3-4hr. NOT killing per user policy of "let in-flight finish".
- GPU 1-7: idle (0 MiB)

Idle orchestrators (no jobs queued):
- exp_v1_orchestrator (PID 2336363) — will exit when cola done
- exp_v1_eval (PID 2365272) — watcher, will pick up cola eval after summary.json
- p0_reeval_orchestrator already exited

## Concerns / clarifications BEFORE starting (please respond, I'll proceed with my best guess if no reply within 4hr)

### C1. drop_rate parameter not yet exposed
Current `stage3_run.py` line 913 hardcodes `target_drop_rate=0.5` for `relora_random_drop`. To do Exp-1 sweep over {0.0, 0.1, 0.25, 0.5, 0.75, 0.9}, I need to add `--random_drop_rate <float>` CLI arg.

Also: `drop_rate=0.0` in the method `relora_random_drop` should match `relora_baseline` per PI's sanity check. Verify by:
- `relora_baseline`: keeps all components (drop_rate=0.0 by definition, hardcoded path lines 904-909)
- `relora_random_drop` with `--random_drop_rate 0.0`: same outcome (all kept, 0 dropped)

I will use `relora_random_drop --random_drop_rate <X>` for ALL 6 Exp-1 cells (including 0.0), so the code path is identical and only the parameter changes. This is cleaner than mixing methods.

**My plan**: add `--random_drop_rate` CLI arg, run all 6 cells with `method=relora_random_drop`. Sanity check: drop_rate=0.0 result must match existing `relora_baseline` from exp_v1 (within 0.5pp).

### C2. exp_v1 cola not yet eval'd
exp_v1 sweep is still 6/7 (cola training, lora_vanilla + relora_baseline eval queued). I will:
- NOT block Exp-1 launch on cola completion
- Let cola+eval continue on GPU 0 background
- Launch Exp-1 on GPU 1-7 (7 GPUs free; can do all 6 cells parallel + 1 spare)

### C3. Muon implementation reference
Will vendor from <https://github.com/KellerJordan/Muon>. Specific reference commit: latest `main` HEAD. Will copy `muon.py` (~200 LOC) with attribution comment + license preservation.

### C4. Muon for LoRA — concrete recipe
Per Keller Jordan's nanoGPT-Muon recipe:
- 2D weights (lora_A: r×in, lora_B: out×r) → Muon
- 1D / non-parameter (scaling, bias) → AdamW
- Embeddings, head → AdamW (always)
- Newton-Schulz iterations: 5 (default)
- Skip Muon for `lora_B` first step (B initialized to zero → NaN otherwise). My fix: detect zero-norm param and use AdamW for that single step, then switch to Muon. Alternative: add small noise (1e-8 * randn) to B at init.

Picking option A (AdamW fallback for zero-norm B step) since cleaner.

### C5. Decision rule on Exp-1 outcome
PI rule: "monotonic OR inverted-U → proceed; flat → stop & report". 
- "Flat" = max - min < 1pp gsm8k_flex across drop_rate?
- I'll use: max-min < 1.0pp → flat; otherwise proceed.

### C6. Bootstrap CI
PI mentioned `scripts/bootstrap_ci.py`. Confirmed not present in repo. Will write minimal one (resamples lm_eval per-sample log to compute 95% CI on accuracy) when generating plot.

### C7. evaluation budget
8 GPUs × 24h = 192 GPU-h budget for both Exp-1 (18h) + Exp-2 (24h) = 42 GPU-h. Comfortable. 
Plan: Exp-1 launches NOW (6 cells parallel on 6 GPUs), expected wall clock ~3-4hr (qwen3-8b/3000 step). Then Exp-2 after Muon Commit M0 smoke pass.

## Plan timeline (assuming concerns above are OK)

| Time (UTC) | Action |
|---|---|
| 08:40 | Commit Exp-0a: add `--random_drop_rate` CLI arg |
| 08:45 | Launch Exp-1 (6 cells parallel, GPU 1-6) |
| ~09:00 | Start writing scripts/muon.py + stage3_run optimizer integration |
| ~10:00 | Commit M0 candidate; smoke test on GPU 7 (qwen3-8b 200 steps) |
| ~12:00 | Smoke pass → Commit M0 push |
| 12:30+ | Exp-1 cells start finishing (~3-4hr per cell) |
| ~15:00 | Exp-1 done → eval queued → bootstrap_ci.py + plot |
| ~16:00 | Open analysis/COMM_AGENT_TO_PI_*_exp1_done.md with table+plot |
| ~16:00 | Launch Exp-2 (8 cells, GPU 0-7 parallel — cola long done) |
| ~22:00 | Exp-2 trains done, evals running |
| ~24:00 | Exp-2 conclusion in analysis/exp_muon_decoupling_qwen3-8b.md |

Push every 4 hours per PI cadence: 12:35, 16:35, 20:35, 00:35.

## Summary of new code I'll write (Commit M0 / Exp-0a / Exp-1)

1. `scripts/stage3_run.py` add `--random_drop_rate <float>` (Exp-0a)
2. `scripts/muon.py` (vendored, ~200 LOC, MIT license)
3. `scripts/stage3_run.py` add `--optimizer {adamw, muon}`, build mixed param groups
4. `scripts/exp_drop_rate_orchestrator.py` (6 cells)
5. `scripts/exp_muon_orchestrator.py` (8 cells)
6. `scripts/bootstrap_ci.py` (95% CI for lm_eval samples)
7. `analysis/exp_drop_rate_sweep_qwen3-8b.png` (matplotlib)
8. `analysis/exp_muon_decoupling_qwen3-8b.md`

If you want me to wait for clarification before launching, write a reply file. Otherwise I proceed in 4hr.

EOF
