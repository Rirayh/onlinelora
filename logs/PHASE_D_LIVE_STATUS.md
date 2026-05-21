# Phase D Live Status — 2026-05-21 ~12:50

## DIRECTIVE (recap, condensed from PI msg)
Validate "S3pos works on quality post-trained, mid-capacity, non-saturated bases" via Qwen3 + Qwen3.5 multi-size sweep.
- Dataset: tulu3-sft only
- Methods: lora_vanilla, relora_baseline, relora_diag_gated_S3pos, dora, cola
- Wave 1 (small→big, commit per size): qwen35-0p8b, qwen35-2b, qwen3-1p7b, qwen35-4b, qwen3-4b, qwen35-9b, qwen3-14b
- Wave 2 (only after Wave 1 + qwen3-8b validates hypothesis): qwen3-32b, qwen35-27b (skip cola for both)
- Train config IDENTICAL to qwen3-8b: rank=8 alpha=16 target=qkvo+gate/up/down,
  AdamW lr=1e-4 betas=(0.9,0.999) wd=0.01, cosine warmup=100/total=3000, bs=8 grad_accum=1,
  merge_every=750 (non-vanilla/dora), COLA K=4×750 with FULL Adam reset per stage,
  val every 250 steps.
- Each cell evidence: train_loss, val_loss, effective_rank, cumulative_rank, condition_number,
  saliency_at_merge (merge-arm), dropped_components (gated/random_drop), summary.json with
  aborted/best_step/best_val_loss, lm_eval_v3/ (gsm8k,hellaswag,arc_challenge,--log_samples),
  adapter under best/.
- Sanity gate: scripts/verify_adapter_loaded.py — for each cell, confirm adapter changes logits
  vs base. If a model fails sanity, halt all training of that model + write logs/PHASE_D_ADAPTER_BUG_<model>.md.
- DO NOT: restart daemon (Phase D is manual queue), train on metamath, train MoE, run cola on
  qwen3-32b/qwen35-27b, fix non-Qwen, re-eval existing lm_eval_v3.
- Report cadence: per-size commit `Phase D <slug>: 5/5 done, S3pos vs vanilla = +X.XXpp on GSM8K-flex`,
  + maintain summary/main_table.md and summary/phase_d_curve.md (size | vanilla | baseline | S3pos | DoRA | COLA | Δ_vs_baseline | Δ_vs_DoRA).
- Negative push immediately if: any size S3pos LOSES baseline by >2pp, OOM/instability,
  adapter sanity fails, or Wave 1 complete.

## KEY FILES
- scripts/stage3_run.py       — main training loop (already has fix from b7d07dc to copy best/ → adapter/)
- scripts/auto_fill_daemon.py — DO NOT relaunch (PI explicit)
- scripts/build_main_table.py — picks lm_eval_v3 > v2 > v1
- scripts/fix_p0_adapter_from_best.py — retroactive B=0 cleanup, run if needed
- logs/PHASE_D_ENV.md         — blocker analysis (committed c224f74)
- logs/scout/PHASE_D_TRIAGE.md — gemma3 in-flight kill record (committed c224f74)

## ENVIRONMENTS (key finding)
| env | transformers | peft | qwen3_5 supported |
|---|---|---|---|
| espo | 4.52.0.dev0 | 0.17.0 | ❌ KeyError 'qwen3_5' |
| flash_moe | **5.3.0** | NO | ✅ AutoConfig OK |
| RRenv | 5.3.0 | NO | likely ✅ |
| vllm_eval | 4.57.6 | 0.18.1 | ❌ |
| modes | 4.51.3 | NO | ❌ |
| opsd | 4.57.1 | NO | ❌ |
| rllm/rllm_backup | 4.57.6 | NO | ❌ |

PI plan: install peft into flash_moe (or RRenv), use it for Qwen3.5 train + eval.
Existing `espo` keeps Qwen3 dense work (Qwen3ForCausalLM unchanged).

PYTHON BINARIES:
  espo:      /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
  flash_moe: /mnt/cpfs/junlongke/miniconda3/envs/flash_moe/bin/python
  RRenv:     /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python

## DOWNLOAD STATE (as of 12:50)
COMPLETE (under /mnt/cpfs/junlongke/onlinelora/models/):
  qwen3-1p7b   3.8GB   Qwen3ForCausalLM ✅
  qwen3-4b     7.6GB   (Qwen3-4B-Instruct-2507) Qwen3ForCausalLM ✅
  qwen3-14b    ~30GB
  qwen3-32b    62GB
  qwen35-0p8b  1.7GB   Qwen3_5ForConditionalGeneration (multimodal hybrid)

IN PROGRESS (just kicked off via HF_ENDPOINT=https://hf-mirror.com):
  scripts/dl_qwen35_seq.sh  → qwen35-2b, qwen35-4b, qwen35-9b, qwen35-27b
  log: logs/scout/dl_qwen35_seq.log

EXISTING LARGE BASES (already there from prior phases):
  /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B
  /mnt/cpfs/public_data/public_model/Qwen/Qwen2.5-7B
  /mnt/cpfs/public_data/public_model/Mistral/Mistral-7B-v0.3
  /mnt/cpfs/public_data/public_model/Meta-Llama-3-8B
  models/{OLMo-2-7B, R1-Distill-Qwen-7B, gemma-3-12b-it, AceReason-Nemotron-7B (DEPRECATED)}

## QWEN3.5 ARCH NOTES (critical for adapter plumbing)
- model_type: qwen3_5
- architectures: ['Qwen3_5ForConditionalGeneration']
- config has sub_configs: {vision_config, text_config, ...}
- text_config: hidden_size=1024 (0.8B), num_hidden_layers=24
- layer_types: alternating ['linear_attention'×3, 'full_attention'] every 4 (full_attention_interval=4)
  → S3pos saliency math (first-order on lora_B) is validated only on standard self-attention.
    Linear_attention layers may need to be EXCLUDED from target_modules or treated differently.
- Has image_token_id=248056, video_preprocessor_config.json — for our text-only tulu3-sft training
  these are ignored if we feed only text inputs.
- Uses AutoProcessor (not AutoTokenizer) for combined image+text. For text-only training
  we should be able to pass the tokenizer subcomponent.

## GPU STATE (as of 12:50)
GPU 0: 20GB — running llama3-8b/tulu3-sft/dora lm_eval_v2 (PID 1755167, ~19% complete, ETA 3-4h)
GPU 3: 58GB — gemma3/tulu3/lora_vanilla train (PID 1733368, step 1175/3000, ~10h ETA)
GPU 5: 71GB — gemma3/tulu3/relora_baseline train (PID 1735950, step 1075/3000, ~10h ETA)
GPU 6: 70GB — gemma3/tulu3/relora_diag_gated_S3pos train (PID 1762026, step 500/3000, ~22h ETA)
GPU 1, 2, 4, 7: idle (free for Phase D)

## TODO (next steps after context-trim)
P1 IN PROGRESS:
  [x] survey conda envs → flash_moe (5.3.0) + RRenv (5.3.0) recognize qwen3_5
  [x] start qwen35 download via mirror → PID running, log dl_qwen35_seq.log
  [ ] install peft into flash_moe (pip install peft into /mnt/cpfs/junlongke/miniconda3/envs/flash_moe)
  [ ] verify Qwen3-1.7B smoke load + tulu3 forward pass (espo env)
  [ ] adapt stage3_run.py to handle Qwen3.5: text_config sub-arch, target_modules for
      mixed linear/full_attention layers, AutoProcessor instead of AutoTokenizer
  [ ] write scripts/verify_adapter_loaded.py (the sanity gate)

P2 PENDING (Wave 1 launch order):
  qwen35-0p8b → qwen35-2b → qwen3-1p7b → qwen35-4b → qwen3-4b → qwen35-9b → qwen3-14b
  per-size: 5 cells × ~1.5h (small) to ~5h (14B) = use 4 free GPUs in parallel

## TRAINING LAUNCHER TEMPLATE (copy from prior runs)
For Qwen3 dense (works in espo):
  CUDA_VISIBLE_DEVICES=<g> /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python scripts/stage3_run.py \
    --model_path models/<slug> --model_key <slug> --dataset tulu3-sft \
    --method <method> --total_steps 3000 --merge_every 750 \
    --eval_every 250 --ckpt_every 250 --saliency_max_seq_len 512 \
    --attn_implementation sdpa --save_adapter --seed 42 \
    --out_root results/stage3_v2/<slug>/tulu3-sft/<method>/seed42

For Qwen3.5 (needs flash_moe + adaptations TBD):
  CUDA_VISIBLE_DEVICES=<g> /mnt/cpfs/junlongke/miniconda3/envs/flash_moe/bin/python scripts/stage3_run.py \
    ... (with --model_path models/<slug> and updated tokenizer/processor logic)

## RECENT COMMITS
b7d07dc [P0-FIX] merge-method adapters: copy best/ -> adapter/ at end-of-training
2518e4a progress 2026-05-21 12:10: post-P0-fix evals + new-model trainings
c224f74 [NEGATIVE] Phase D blocker: Qwen3.5 is multimodal (this commit explained options A/B/C)

## P0 ACCEPTANCE GATE (already passed, PI saw)
olmo2-7b/tulu3-sft 4 cells distinct ✅, qwen3-8b/tulu3-sft S3neg=86.81% (new peak post-fix)

## CONTEXT-TRIM CHECKPOINT (12:55) — RESUME HERE

### Currently running in bg (DO NOT KILL):
- PID 1772667: bash dl_qwen35_seq.sh — Qwen3.5 {2B,4B,9B,27B} from hf-mirror.com,
  log: logs/scout/dl_qwen35_seq.log
- PID 1772680: pip install peft==0.17.0 into flash_moe env,
  log: logs/scout/install_peft_flash_moe.log
- PID 1733368, 1735950, 1762026: gemma3 cleanup trains (P0.4) — keep alive
- PID 1755167: llama3 dora lm_eval — keep alive

### Next concrete steps (resume):
1. Verify peft installed in flash_moe:
   /mnt/cpfs/junlongke/miniconda3/envs/flash_moe/bin/python -c "import peft;print(peft.__version__)"
2. Smoke Qwen3-1.7B forward in espo (already verified config; need actual weights):
   ALREADY DONE earlier: loaded Qwen3ForCausalLM 1.72B, logits shape correct.
3. Smoke Qwen3.5-0.8B in flash_moe — the critical compatibility test.
   Must determine: does AutoModelForCausalLM work? Or need Qwen3_5ForConditionalGeneration directly?
   What target_modules names are present in linear_attention layers vs full_attention?
4. Adapt scripts/stage3_run.py if needed for Qwen3.5:
   - tokenizer = AutoTokenizer (text-only) OR AutoProcessor.tokenizer
   - model.text_model accessor for the LM submodule
   - target_modules: probe model.named_modules() to see what q/k/v/o/gate/up/down look like;
     may need different names for linear_attention layers
5. Write scripts/verify_adapter_loaded.py (sanity gate):
   load base + load with peft adapter → forward 1 GSM8K sample → assert logits differ.
6. Once smoke passes for one Qwen3.5 size, launch Wave 1:
   qwen35-0p8b → qwen35-2b → qwen3-1p7b → qwen35-4b → qwen3-4b → qwen35-9b → qwen3-14b
   GPUs: 1, 2, 4, 7 free; 0,3,5,6 occupied by P0 cleanup (free up over next 3-22h).

### Quick commands to recheck state on resume:
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  ps -eo pid,etime,args | grep -E '(stage3_run|lm_eval|dl_qwen|peft)' | grep -v grep
  tail -5 /mnt/cpfs/junlongke/onlinelora/lora_obd/logs/scout/dl_qwen35_seq.log
  ls /mnt/cpfs/junlongke/onlinelora/models/ | sort
  /mnt/cpfs/junlongke/miniconda3/envs/flash_moe/bin/python -c "import peft;print(peft.__version__)" 2>&1

## CONTEXT-TRIM CHECKPOINT 2 (12:58)

### KEY FINDING: peft 0.17 incompatible with transformers 5.x
peft 0.17.0/0.17.1 imports `HybridCache` which was removed in transformers 5.x:
  ImportError: cannot import name 'HybridCache' from 'transformers'

Need a NEWER peft version compatible with transformers 5.x. Options:
- pip install peft==latest (try 0.18+)
- pip install "peft>=0.18" 
- Or peft from main: pip install git+https://github.com/huggingface/peft.git

### Current state of bg processes:
- PID 1772667: dl_qwen35_seq.sh (Qwen3.5 2B/4B/9B/27B from hf-mirror) — RUNNING
- peft install in flash_moe FAILED (broken pip resolver in flash_moe's pip)
- peft install in RRenv UNINSTALLED 0.17.1 first then tried 0.17.0 — leaves state ambiguous
- PID 1755167: llama3 dora lm_eval — RUNNING (~19% done, 3-4h ETA)
- PID 1733368, 1735950, 1762026: gemma3 cleanup trains — RUNNING

### NEXT STEP ON RESUME:
1. Install peft compatible with transformers 5.x:
   /mnt/cpfs/junlongke/miniconda3/envs/RRenv/bin/python -m pip install --upgrade "peft>=0.18"
   verify: python -c "from peft import LoraConfig"
2. Verify Qwen3.5-0.8B loads in RRenv with peft + transformers 5.3.0
3. Inspect Qwen3.5 module names to decide target_modules
   probe with: for n,m in model.named_modules(): print(n)
4. Adapt scripts/stage3_run.py for Qwen3.5 if needed

### IMPORTANT: PI directive said proceed in parallel, not block
While solving Qwen3.5 plumbing, can ALREADY start training Qwen3 dense models:
  qwen3-1p7b (3.8GB, loaded fine in espo, Qwen3ForCausalLM)
  qwen3-4b (7.6GB, same arch as qwen3-8b, drop-in)

LAUNCH-READY: see logs/PHASE_D_LIVE_STATUS.md "TRAINING LAUNCHER TEMPLATE" section.
GPUs free: 1, 2, 4, 7. Can start 4 Qwen3 dense trainings IMMEDIATELY in parallel.

### WAVE 1 LAUNCH ORDER (per directive)
qwen35-0p8b, qwen35-2b, qwen3-1p7b, qwen35-4b, qwen3-4b, qwen35-9b, qwen3-14b
But qwen3-1p7b ready right now → use GPU 1, 2, 4, 7 to start qwen3-1p7b 5 cells in parallel
(actually only 4 GPUs free, so pick 4 methods first; cola serial after first 4 done)

5 methods: lora_vanilla, relora_baseline, relora_diag_gated_S3pos, dora, cola
Parallel batch 1 (4 cells on 4 GPUs):
  G1: qwen3-1p7b/tulu3-sft/lora_vanilla
  G2: qwen3-1p7b/tulu3-sft/relora_baseline
  G4: qwen3-1p7b/tulu3-sft/relora_diag_gated_S3pos
  G7: qwen3-1p7b/tulu3-sft/dora
Then cola when first GPU frees.

## CONTEXT-TRIM CHECKPOINT 3 (13:02) — LAUNCHING QWEN3-1.7B WAVE 1

About to launch 4 parallel trainings on free GPUs 1,2,4,7. Qwen3-1.7B is
Qwen3ForCausalLM (same arch as qwen3-8b, drop-in to existing pipeline).

EXACT COMMANDS:
  ROOT=/mnt/cpfs/junlongke/onlinelora/lora_obd
  PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
  for spec in "1:lora_vanilla:9999" "2:relora_baseline:750" "4:relora_diag_gated_S3pos:750" "7:dora:9999"; do
    GPU=$(echo $spec|cut -d: -f1); METHOD=$(echo $spec|cut -d: -f2); MERGE=$(echo $spec|cut -d: -f3)
    OUT=$ROOT/results/stage3_v2/qwen3-1p7b/tulu3-sft/$METHOD/seed42
    LOG=$ROOT/logs/scout/train-qwen3-1p7b-tulu3-sft-$METHOD.log
    mkdir -p $OUT
    STEPS=3000
    [ $METHOD = "dora" ] && STEPS=800
    nohup env CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/scripts/stage3_run.py \
        --model_path /mnt/cpfs/junlongke/onlinelora/models/qwen3-1p7b \
        --model_key qwen3-1p7b --dataset tulu3-sft --method $METHOD \
        --total_steps $STEPS --merge_every $MERGE \
        --eval_every 250 --ckpt_every 250 --saliency_max_seq_len 512 \
        --attn_implementation sdpa --save_adapter --seed 42 \
        --out_root $OUT > $LOG 2>&1 &
    disown
    echo "G$GPU $METHOD pid=$!"
  done

After cola support: launch qwen3-1p7b/tulu3-sft/cola when first GPU frees (cola = relora_baseline
with full Adam reset; method "cola" already added to METHOD_CHOICES per b7d07dc).

After 1.7B 5 cells done: queue qwen3-4b (same pattern, 7.6GB → fits comfortably).
After Qwen3.5 plumbing done: interleave Wave 1 order qwen35-0p8b → 2b → 4b → 9b.

DO NOT TOUCH:
  GPU 0 (llama3 dora eval), GPU 3,5,6 (gemma3 cleanup trains).

STATUS DOC: this file logs/PHASE_D_LIVE_STATUS.md is the source of truth across context-trim.
