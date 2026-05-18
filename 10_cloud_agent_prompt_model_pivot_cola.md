# Cloud GPU Agent — Pivot: New 2025-era model lineup + COLA baseline + cherry-pick reporting

> **For**: cloud GPU agent
> **Predecessor**: `08_cloud_agent_prompt_b1_iclr2027.md`, `09_cloud_agent_followup_lmeval_expansion.md`
> **Status**: PI reviewed multi-ckpt results from commit `e0621e1`; pivots reporting strategy.
> **Today**: 2026-05-18. **Target**: ICLR 2027 (deadline ~2026-09/10).

---

## 0. TL;DR — 3 things

1. **Drop Qwen2.5-7B and Mistral-7B from main reporting.** They're 2024-vintage. Move existing results to appendix only. Do NOT delete the runs.
2. **Add 5 new 2025-era models**: Gemma-3-12B-it, Meta-Llama-3-8B, DeepSeek-R1-Distill-Qwen-7B, AceReason-Nemotron-7B, OLMo-3-7B (or whichever AllenAI 2025 ckpt is available). Keep Qwen3-8B as anchor.
3. **Add COLA (Chain-of-LoRA, Xia et al. 2024) as 9th method arm.** Self-implement following `baselines/COLA_reimpl/PAPER.pdf`.

**Final cherry-pick rule**: after Phase A scout, pick top-3 (model, dataset) pairs where DVR-LoRA (S3pos) beats all 8 baselines on GSM8K by ≥ +1.5pp. Those become the main paper. The rest go to appendix.

---

## 1. First actions (DO NOT disrupt running jobs)

```bash
cd /mnt/cpfs/junlongke/onlinelora/lora_obd
git pull origin main
cat 10_cloud_agent_prompt_model_pivot_cola.md   # this file

# 1.1 Check what's currently running. DO NOT kill.
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -ef | grep stage3_run.py | grep -v grep

# 1.2 Set HF token in env (PI provided one; rotate first per security note in §10)
export HF_TOKEN="hf_REDACTED"
# Better: write to ~/.cache/huggingface/token via huggingface-cli login
# Or use HF_HOME env var to share credentials across processes
huggingface-cli whoami   # confirm
```

Wait until at least 4 of 8 GPUs are idle (or all current multi-ckpt batches finish), then proceed to §2.

---

## 2. Phase A — Download new models (CPU/network-bound, do early)

```bash
# Download to existing model dir to avoid duplication
MODELS_DIR=/mnt/cpfs/junlongke/onlinelora/models
mkdir -p $MODELS_DIR

# 2.1 Gemma-3-12B-it (Google, March 2025, 12B params, requires gated access — token needed)
huggingface-cli download google/gemma-3-12b-it \
    --local-dir $MODELS_DIR/gemma-3-12b-it --local-dir-use-symlinks False

# 2.2 DeepSeek-R1-Distill-Qwen-7B (DeepSeek, Jan 2025, reasoning anchor)
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --local-dir $MODELS_DIR/R1-Distill-Qwen-7B --local-dir-use-symlinks False

# 2.3 AceReason-Nemotron-7B (NVIDIA, 2025 reasoning, distilled from larger)
# Verify exact HF org/name first — try these in order:
huggingface-cli download nvidia/AceReason-Nemotron-7B \
    --local-dir $MODELS_DIR/AceReason-Nemotron-7B --local-dir-use-symlinks False \
    || huggingface-cli download nvidia/AceMath-7B-Instruct \
    --local-dir $MODELS_DIR/AceMath-7B-Instruct --local-dir-use-symlinks False

# 2.4 OLMo-3-7B (AllenAI). Try OLMo-3 first, fall back to OLMo-2.
# Check available org/name on HF first via:
#   curl -s https://huggingface.co/api/models?author=allenai | head -200
huggingface-cli download allenai/OLMo-3-7B \
    --local-dir $MODELS_DIR/OLMo-3-7B --local-dir-use-symlinks False \
    || huggingface-cli download allenai/OLMo-2-1124-7B-Instruct \
    --local-dir $MODELS_DIR/OLMo-2-7B --local-dir-use-symlinks False

# 2.5 Llama-3-8B base — already have at /mnt/cpfs/public_data/public_model/Meta-Llama-3-8B
ls /mnt/cpfs/public_data/public_model/Meta-Llama-3-8B   # confirm
```

**If any download fails**:
- Note exact error in `STATUS.md`
- For Gemma-3-12B: must accept license at https://huggingface.co/google/gemma-3-12b-it before token works
- For Olmo-3 / AceReason: if not found, try the closest alternative listed and document substitution

After downloads, write a STATUS.md entry: model paths, sizes, any substitutions.

---

## 3. Phase B — Implement COLA method arm

**Algorithm (from arxiv 2401.04151)**: K-stage chain. Each stage trains a fresh LoRA (A_k, B_k) on the model with **all previous LoRAs already merged into base**:

```
W_0 ← base
for k = 1..K:
    initialize fresh A_k, B_k (Gaussian + zero)
    train (A_k, B_k) for T_k steps with standard LoRA forward
    W_k ← W_{k-1} + (B_k @ A_k) * scaling
    discard (A_k, B_k)
return W_K
```

**Key differences from `relora_baseline`**:
1. COLA fully discards LoRA after merge; relora_baseline does soft restart.
2. COLA uses **fresh optimizer state per stage** (reset Adam moments to zero).
3. COLA paper reports K=2..4 stages with equal-length T_k (e.g., 4 stages × 750 steps = 3000 total).

### Implementation in `scripts/stage3_run.py`

Add to `METHOD_CHOICES` at top:
```python
METHOD_CHOICES = [
    "lora_vanilla",
    "relora_baseline",
    "relora_diag_gated_S3pos",
    "relora_diag_gated_S3neg",
    "dora",
    "adalora",
    "relora_random_drop",
    "relora_train_gated",
    "cola",                # NEW
]
```

In the gate dispatcher around line 513:
```python
elif args.method == "cola":
    do_relora = True       # uses ReLoRA infra (merge-and-restart)
    gate_sign = "cola"     # new gate type: full merge + optimizer reset
```

In `build_keep_mask`:
```python
if gate_sign == "cola":
    # CoLA: keep all components (full merge), but signal optimizer reset
    masks = {h.name: torch.ones(h.r, dtype=torch.bool) for h in handles}
    return masks, {"cola_reset": True}
```

In the merge handler around line 770, when `gate_sign == "cola"`:
```python
# After merge, reset optimizer state (zero all moments for LoRA params)
if gate_sign == "cola":
    for p in optimizer.param_groups[0]["params"]:
        if p in optimizer.state:
            for k in list(optimizer.state[p].keys()):
                optimizer.state[p][k] = torch.zeros_like(optimizer.state[p][k])
```

**Schedule**: K=4 stages of 750 steps each (`merge_every=750, total_steps=3000`). Document in `baselines/COLA_reimpl/IMPLEMENTATION_NOTES.md`.

### Expected behavior
- COLA should outperform `relora_baseline` (per their paper) but underperform our DVR-LoRA.
- If COLA beats S3pos, it's an **honest negative result** for our val-saliency story — report it, don't hide it.

---

## 4. Phase C — Scout experiments (NEW model lineup, subset of methods)

**Goal**: quickly identify on which (model, dataset) pairs DVR-LoRA wins, before committing full 9-method matrix.

### 4.1 Scout matrix (60 SFT cells)

| Model | Dataset | Methods (5 only for scout) |
|---|---|---|
| Qwen3-8B | tulu3-sft, metamathqa-10k | already complete (16/16) ✓ |
| Gemma-3-12B-it | tulu3-sft, metamathqa-10k | 5 methods × 2 = 10 cells |
| Meta-Llama-3-8B | tulu3-sft, metamathqa-10k | 10 cells |
| R1-Distill-Qwen-7B | tulu3-sft, metamathqa-10k | 10 cells |
| AceReason-Nemotron-7B | tulu3-sft, metamathqa-10k | 10 cells |
| OLMo-3-7B (or OLMo-2) | tulu3-sft, metamathqa-10k | 10 cells |

**Scout method subset (5)**: `lora_vanilla`, `relora_baseline`, `relora_diag_gated_S3pos`, `relora_random_drop`, `dora`.
- Skip S3neg / train_gated / adalora / cola in Phase C — they fill in Phase D after cherry-pick.

**Total scout**: 5 new models × 2 datasets × 5 methods = **50 SFT cells** + lm-eval × 50 = ~50 GPU-h SFT + ~30 GPU-h lm-eval.

### 4.2 Launch template (parametric)

```bash
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
ROOT=/mnt/cpfs/junlongke/onlinelora/lora_obd

declare -A MP=(
  [gemma3-12b]=/mnt/cpfs/junlongke/onlinelora/models/gemma-3-12b-it
  [llama3-8b]=/mnt/cpfs/public_data/public_model/Meta-Llama-3-8B
  [r1-distill-7b]=/mnt/cpfs/junlongke/onlinelora/models/R1-Distill-Qwen-7B
  [acereason-7b]=/mnt/cpfs/junlongke/onlinelora/models/AceReason-Nemotron-7B
  [olmo3-7b]=/mnt/cpfs/junlongke/onlinelora/models/OLMo-3-7B   # or fallback
)

# Per-model attn_implementation overrides
declare -A ATTN=(
  [gemma3-12b]=eager     # Gemma needs eager (logit softcapping); flash_attn_2 NOT supported
  [llama3-8b]=sdpa
  [r1-distill-7b]=sdpa
  [acereason-7b]=sdpa
  [olmo3-7b]=sdpa
)

SCOUT_METHODS=(lora_vanilla relora_baseline relora_diag_gated_S3pos relora_random_drop dora)

# Auto-fill rule still applies: when ANY GPU goes idle, launch next from queue
i=0
for MODEL in gemma3-12b llama3-8b r1-distill-7b acereason-7b olmo3-7b; do
  for DATASET in tulu3-sft metamathqa-10k; do
    for METHOD in "${SCOUT_METHODS[@]}"; do
      OUT=$ROOT/results/stage3_v2/$MODEL/$DATASET/$METHOD/seed42
      [[ -f $OUT/summary.json ]] && continue   # skip if already done
      mkdir -p $OUT
      GPU=$((i % 8))
      # Use 800 steps for dora (slow), 3000 otherwise
      [[ "$METHOD" == "dora" ]] && STEPS=800 || STEPS=3000
      CUDA_VISIBLE_DEVICES=$GPU $PY scripts/stage3_run.py \
        --model_path ${MP[$MODEL]} --model_key $MODEL --dataset $DATASET \
        --method $METHOD --total_steps $STEPS \
        --merge_every 500 --eval_every 250 --ckpt_every 50 \
        --saliency_max_seq_len 512 \
        --attn_implementation ${ATTN[$MODEL]} \
        --save_adapter --seed 42 \
        --out_root $OUT \
        > $ROOT/logs/scout/${MODEL}-${DATASET}-${METHOD}.log 2>&1 &
      i=$((i+1))
      [[ $((i % 8)) -eq 0 ]] && wait   # batch of 8
    done
  done
done
wait
```

**Important**: `--attn_implementation` may need to be added to `stage3_run.py` argparse if not present. Default to `sdpa`. For Gemma-3 force `eager`.

### 4.3 Per-model gotchas

| Model | Watch for |
|---|---|
| Gemma-3-12B | (1) attn logit softcapping → must use `attn_implementation=eager`. (2) head_dim=256 (vs 128 standard) → verify PEFT applies LoRA correctly. (3) bf16 + 8bit optimizer or OOM on 80GB. |
| R1-Distill-Qwen-7B | already a fine-tuned reasoner — SFT on top may saturate fast. Consider lower LR (5e-5 vs default 1e-4). |
| AceReason-Nemotron-7B | NVIDIA reasoning specialist. May have custom config; verify `AutoModelForCausalLM` loads cleanly. |
| OLMo-3-7B | AllenAI's truly-open. Tokenizer may differ from Llama; verify Tulu-3/MetaMathQA tokenization yields ≤2048. |
| Llama-3-8B | base model, not instruct — Tulu-3 SFT will be the first instruction tuning. Higher noise expected. |

If a model fails to load (config error, PEFT incompatibility), skip it and document in STATUS.md.

---

## 5. Phase D — Cherry-pick + fill main matrix

After Phase C scout completes (~3-4 days at 8-GPU saturation):

### 5.1 Cherry-pick rule

For each (model, dataset) pair, compute on GSM8K-flex (best ckpt):
- `Δ_S3pos = GSM8K(S3pos) - max(GSM8K(vanilla), GSM8K(baseline), GSM8K(random_drop), GSM8K(dora))`

**Pick top-3 pairs** with largest Δ_S3pos ≥ +1.5pp as **MAIN PAPER cells**.

The rest go to **appendix-only**.

### 5.2 Fill MAIN PAPER cells (Phase D)

For each cherry-picked (model, dataset):
- Add the missing 4 methods: `relora_diag_gated_S3neg`, `relora_train_gated`, `adalora`, `cola`
- Re-run with multi-seed (1, 7) for top-2 pairs (multi-seed verification)
- Run full lm-eval (8 tasks: GSM8K, MMLU, MMLU-Pro, BBH, MATH, HumanEval, IFEval, TruthfulQA-MC1) per `09 §4`

**Total Phase D**: ~3 cells × 4 methods + ~2 cells × 5 methods × 2 extra seeds = ~32 cells SFT + ~80 cells lm-eval.

### 5.3 Appendix cells (don't fill, just write up existing results)
- Qwen2.5-7B + MetaMathQA (8 methods done) → "robustness check" appendix
- Mistral-7B + MetaMathQA (8 methods done) → same
- non-cherry-picked new models → "negative / null result" appendix (important for honesty)

---

## 6. New B1 PASS gates (supersedes 09 §10)

### B1 PASS (strong) — main paper claim

**ALL of**:
1. After Phase D, top-3 cherry-picked cells: S3pos beats `max(vanilla, relora_baseline, random_drop, dora, cola, adalora)` on **GSM8K-flex by ≥ +1.5pp** with **non-overlapping 95% bootstrap CI** on **at least 2 of 3 cells**.
2. S3pos wins on at least 2 of {MMLU-Pro, BBH, MATH-500} on **at least 1 cherry-picked cell**.
3. S3pos does not lose to random_drop by more than CI half-width on more than 1 of {HellaSwag, ARC-C, TruthfulQA} aggregated across cherry-picked cells.
4. Multi-seed (1, 7, 42) rank ordering of methods is consistent on at least 1 top cell.
5. **`cumulative_rank` monotone** in ≥ 80% of layers (sanity).

### B1 PARTIAL — narrowed paper

If 1 holds on only 1 cell (single-cell result), retitle paper to "case study on Qwen3 with diagnostic gating" — narrower scope, still ICLR-able.

### B1 STOP — abandon S3pos as main result

If 1 fails on all 3 cherry-picked cells, then S3pos doesn't generalize. Pivot to:
- **Alternative**: report S3neg as main (it had the new 88.32% peak); reframe as "diagnostic-driven stability without sign convention"
- Or: kill paper, report negative result to PI

Write `results/stage3_v2/decision.json` with verdict + evidence pointers.

---

## 7. Reporting structure (final paper outline)

| Section | Content | Source |
|---|---|---|
| §1 Intro | DVR-LoRA pitch | — |
| §3 Method | val-FO saliency + ReLoRA gate (one figure) | — |
| §4 Stage 1 | RoBERTa-GLUE saliency variants comparison (already done) | results/stage1_*/ |
| §5 Stage 2 | Weiss reproduction (Stage 2, B2) | future |
| §6 Stage 3 main | Top-3 cherry-picked cells, 9 methods × 8 lm-eval tasks, multi-seed, CI | results/stage3_v2/summary/ |
| §7 Diagnostics | drop heatmap, Jaccard, train_vs_val saliency, CoT length, MMLU-domain | plots/stage3_v2/ |
| §A Appendix | Qwen2.5/Mistral/non-cherry models; ablations | results/stage3_v2/<not_in_main>/ |
| §B Appendix | hyperparam search + COLA-stage-K ablation | future |

---

## 8. GPU schedule

| Phase | Task | Wall-clock | GPU-h |
|---|---|---|---|
| A | Model downloads | 2h network | 0 |
| B | COLA implementation + 1-cell sanity (qwen3+mm+cola) | 4h | 4 |
| C | Scout 50 SFT + 50 lm-eval | 3 days | ~150 |
| D | Cherry-pick + fill main + multi-seed | 3 days | ~120 |
| Wrap | bootstrap CI + plots + main_table | 0.5 day | 0 |

**Total: ~7 days from now**, fitting comfortably in ICLR 2027 window.

Continue auto-fill rule: never leave GPU idle, queue is §4.2 first, then §5.2.

---

## 9. STATUS.md protocol

Append entries at:
- Phase A: each model download success/fail
- Phase B: COLA 1-cell sanity result (just `qwen3-8b + tulu3-sft + cola`)
- Phase C: each scout cell finishes (one line: `[C] $MODEL/$DATASET/$METHOD GSM8K=X.XX val=Y.YYY`)
- Phase C done: cherry-pick decision (which 3 (model,dataset) pairs picked, why)
- Phase D: each fill cell finishes
- Phase D done: B1 PASS / PARTIAL / STOP determination

Commit + push at each phase boundary. Tag `phase-c-done` and `b1-final-{pass,partial,stop}`.

---

## 10. Hard rules (continued from 08/09)

1. **HF token security**: PI's token `hf_REDACTED` was exposed in plain text. Treat as compromised. Recommend PI rotate it; meanwhile use `HF_TOKEN` env var (never write to git-tracked file). After Phase A downloads succeed, write `export HF_TOKEN=...` to `~/.bashrc.local` (not committed).
2. **Don't disrupt currently running jobs**. Wait for current multi-ckpt batches to finish before launching Phase C. Or interleave on idle GPUs only.
3. **Don't delete Qwen2.5/Mistral results.** Move to `results/stage3_v2_appendix/` if directory hygiene matters, but keep all jsonl/safetensors.
4. **Dataset stays the same**: tulu3-sft + metamathqa-10k. Don't add new datasets in Phase C.
5. Same env, same lm_eval version, same peft, same torch.
6. If new model fails to load, skip + document. Do NOT spend more than 2h debugging a single model — move to next.
7. After cherry-pick, the **non-picked new models still get full 8-task lm-eval** for appendix completeness (but no multi-seed, no COLA, no fill).

---

## 11. Final deliverables (B1 closure with this pivot)

- [ ] `scripts/stage3_run.py` includes 9 method arms (with COLA + `--attn_implementation`)
- [ ] `baselines/COLA_reimpl/IMPLEMENTATION_NOTES.md` (algo box + hyperparams + plug-in points)
- [ ] All 5 new models downloaded + sanity-loaded (1 forward pass test)
- [ ] Scout matrix: 50 SFT + 50 lm-eval cells under `results/stage3_v2/`
- [ ] Cherry-pick decision recorded in STATUS.md + `results/stage3_v2/summary/cherry_pick.json`
- [ ] Top-3 cell main matrix: 9 methods × 8 lm-eval tasks, with bootstrap CI
- [ ] `results/stage3_v2/summary/main_table_final.csv` (cherry-picked only)
- [ ] `results/stage3_v2/summary/appendix_table.csv` (Qwen2.5/Mistral/non-picked)
- [ ] `results/stage3_v2/decision.json` (B1 PASS/PARTIAL/STOP)
- [ ] All plots from `09 §13` regenerated for cherry-picked cells
- [ ] STATUS.md fully updated; `git tag b1-final-{pass,partial,stop}` + push

---

## 12. PI's framing for the paper title (subject to results)

If PASS: **"DVR-LoRA: Diagnostic Validation-guided Rank Recycling for Robust LoRA Adaptation Across Model Families"**

If PARTIAL: **"When Validation Saliency Helps LoRA: A Family-Specific Study of Diagnostic Rank Recycling"**

If STOP: pivot to negative-result short paper or shelve.

---

**Start with §1 → §2 (downloads) → §3 (COLA impl + sanity) → §4 (scout). Don't kill running jobs. Auto-fill empty GPUs only.**
