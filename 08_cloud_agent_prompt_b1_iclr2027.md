# Cloud GPU Agent — Prompt for Phase B1 (ICLR 2027 Submission)

> **Copy-paste this entire block into the cloud GPU agent's first message.**
> **Target venue**: ICLR 2027 (deadline ~2026-09/10).
> **Today**: 2026-05-14. You have ~4 months to deliver B1 + B2 + B3.

---

## 0. Mission

You are the cloud GPU agent for the **onlinelora** project (DVR-LoRA / Diagnostic
Validation-guided Rank-Recycling LoRA). Stages 0–3 are already done; the
hypothesis is validated. Your job for the next 4 months is to close the evidence
gaps identified in `07_missing_experiments_for_paper.md` and deliver an
**ICLR 2027-grade paper**.

Order of execution: **B1 → B2 → B3 → B1.5 (reimpl baselines, only if time)**.
Each batch has a hard PASS/STOP gate; do not auto-skip.

---

## 1. First actions (do these in order, do not skip)

```bash
# 1.1 Get the latest code
cd <your-workdir>
git clone git@github.com:Rirayh/onlinelora.git
cd onlinelora
git log --oneline -5                          # confirm commit a80e9d1 or later is HEAD

# 1.2 Pull the baseline upstream repos at pinned commits
bash baselines/setup_baselines.sh             # populates baselines/*_official/ and *_reference/
ls baselines/                                  # confirm 4 _official + 4 _reimpl dirs exist

# 1.3 Read ALL of these BEFORE touching code (in this order)
#     - STATUS.md                                 (full history, append-only)
#     - 07_missing_experiments_for_paper.md       (THIS batch's spec; §0/§7/§8/§12)
#     - 03_handover_for_gpu_agent.md              (original execution handbook)
#     - 02_research_v2_baselines_theory.md §4     (baseline white-space)
#     - baselines/MANIFEST.md                     (what code lives where)
#     - 05_pi_response_AB_parallel.md             (sign-convention decisions)

# 1.4 Env probe
CONDA=/mnt/cpfs/junlongke/miniconda3/bin/conda
$CONDA env list
nvidia-smi --query-gpu=index,name,memory.free --format=csv
# Reuse an existing env if compatible (no pip install in shared envs);
# otherwise create at /mnt/cpfs/<you>/envs/lora-obd-b1
```

Then **write the first STATUS.md append entry**: env path, GPU inventory, commit
hash you started from, planned batch B1 start time.

---

## 2. Phase B1 — Stage-3 main table completion (priority #1, ~150 GPU-h, 3 days)

### 2.1 Goal

In `Llama-3.1-8B-Instruct` × `Qwen3-8B` × `{Tulu-3 SFT, MetaMathQA-10k}` settings,
fill the Stage 3 main table with **all official-code baselines + key ablations**,
plus full lm-eval-harness suite. Stop B1 the moment §2.7 gates either PASS or STOP.

### 2.2 Anchors (2026-locked, see 07_missing_experiments §12.8)

| Slot | Model | HuggingFace ID |
|---|---|---|
| Main 8B | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Cross-family 8B | Qwen3-8B | `Qwen/Qwen3-8B` |

**Datasets**:
- `allenai/tulu-3-sft-mixture` (140k) — replaces Alpaca
- `meta-math/MetaMathQA` subsample to 10k — replaces raw GSM8K-train (Qwen overfitting limitation)

### 2.3 Methods to run (Phase B1, no self-reimpl yet)

Implement these 8 method arms in `scripts/stage3_run.py`:

| Method arm name | Source | Notes |
|---|---|---|
| `lora_vanilla` | already implemented | r=16, attn+MLP linear |
| **`dora`** | NEW: `peft.LoraConfig(use_dora=True)` | sanity-cross-check vs `baselines/DoRA_official/` |
| **`adalora`** | NEW: `peft.AdaLoraConfig` | use defaults from `baselines/AdaLoRA_official/` |
| `relora_baseline` | already implemented | merge every 1000 steps, optim reset |
| `relora_diag_gated_s3pos` | ours | drop S3_fo_val_signed > 0 |
| `relora_diag_gated_s3neg` | ours | drop S3_fo_val_signed < 0 |
| **`relora_random_drop`** | NEW ablation | random saliency, same drop_rate as ours |
| **`relora_train_gated`** | NEW ablation | gate = S2 (train FO), **the critical Sensitivity-LoRA-flavored sanity-check** |

> **Deferred to B1.5** (do NOT do now): COLA / Sensitivity-LoRA / CTR-LoRA / PrunedLoRA
> self-reimplementations. Their PDFs are at `baselines/*_reimpl/PAPER.pdf`. If B1
> finishes early and you have leftover GPU-h, start with Sensitivity-LoRA (it is
> the most critical must-beat per `02_research_v2 §1.1`).

### 2.4 Evidence to save per run (HARD requirement, reviewer-ready)

For each run `results/stage3_v2/<model>/<dataset>/<method>/<seed>/`:

```
├── config.yaml                    # full hyperparameters, seed, commit hash, env hash
├── train_loss.jsonl               # one line per step: {step, train_loss, lr}
├── val_loss.jsonl                 # one line per eval: {step, val_loss, paloma_ppl?}
├── effective_rank.jsonl           # every 200 steps, all LoRA layers (NOT sampled)
├── condition_number.jsonl         # every 200 steps, all layers
├── cumulative_rank.jsonl          # NEW: every merge event, SVD-rank of merged Δ_stable
├── saliency_at_merge.jsonl        # at each merge: {step, layer, comp, S2, S3, S3_signed, decision}
├── dropped_components.jsonl       # NEW: each merge: {step, layer, comp, score, threshold, decision}
├── adapter_final/                 # save_pretrained — needed for lm-eval
├── wall_clock.json                # {start, end, gpu_h, peak_mem_gb, gpu_name}
└── ABORTED.flag (if abort_factor=1.5 triggered)
```

**`cumulative_rank.jsonl` and `dropped_components.jsonl` are NEW** — these did not
exist in Stage 3 v1. Add them to `stage3_run.py`. They produce the narrative-core
plot `fig9_active_vs_cumulative_rank.png`.

### 2.5 Evaluation (lm-eval-harness, vLLM backend)

After all SFT runs finish, evaluate every saved adapter on:

| Benchmark | Setting | Why |
|---|---|---|
| GSM8K | 5-shot, strict-match | math |
| MMLU | 5-shot | general knowledge (keep, but note saturated; B3 will add MMLU-Pro) |
| IFEval | 0-shot | instruction following |
| BBH | 3-shot, CoT | multi-step reasoning |
| HumanEval | 0-shot, pass@1 | code (Tulu-3 has code) |

**Implementation rules (2026 standard)**:

```bash
# Pin lm-eval-harness to a single commit hash across ALL runs
LM_EVAL_COMMIT=<pick one stable tag at the moment you start>
pip install --target=$HOME/.local/lm-eval \
    "git+https://github.com/EleutherAI/lm-evaluation-harness@${LM_EVAL_COMMIT}"

# Use vLLM as backend (10x faster than HF transformers)
lm_eval \
  --model vllm \
  --model_args pretrained=<base_model>,peft=<adapter_path>,dtype=bfloat16,tensor_parallel_size=1 \
  --tasks gsm8k,mmlu,ifeval,bbh,humaneval \
  --batch_size auto \
  --output_path results/stage3_v2/<model>/<dataset>/<method>/lm_eval/ \
  --log_samples
```

Record `LM_EVAL_COMMIT` in STATUS.md once and never change it for the rest of B1+B2+B3.

### 2.6 Plots to generate (after eval is done)

Use `scripts/plot_from_json.py` style (PI §4.4 hard constraint: every figure
first writes its source `*.plot.json`, then renders PNG). Save under
`plots/stage3_v2/`:

- **`fig9_active_vs_cumulative_rank.png`** — narrative-core: active LoRA rank
  stays at r=16 (flat); cumulative rank of Δ_stable grows monotonically across
  merges. One line per model × method.
- **`fig10_main_table_heatmap.png`** — 8 method × 2 model × 2 dataset × 5
  benchmark heatmap (or grouped bar).
- **`fig11_lm_eval_gsm8k_bars.png`** — 8 methods × 2 models, GSM8K 5-shot
  accuracy bars with 95% CI.
- **`fig12_ablation_grid.png`** — random-drop / train-gated / val-gated /
  val-signed-gated 4-row comparison (val_loss + GSM8K).

### 2.7 B1 PASS / STOP gates (binding)

**PASS** if **all** of:
1. `relora_diag_gated_s3pos` beats `relora_baseline` and `relora_train_gated`
   on **GSM8K + at least 2 of {MMLU, IFEval, BBH, HumanEval}** by ≥ 1.0 absolute
   point on **at least one model** (Llama-3.1 or Qwen3).
2. `relora_random_drop` is **worse** than `relora_diag_gated_s3pos` on val_loss
   on ≥ 1 of 2 datasets (proves the signal, not the pruning, drives the gain).
3. `cumulative_rank.jsonl` shows monotonic growth in ≥ 80% of layers for
   `relora_diag_gated_s3pos`.

**STOP and write `results/stage3_v2/decision.json` with `{go: false, ...}`** if:
- `relora_diag_gated_s3pos` is statistically tied with `relora_baseline` on
  both models on all 5 benchmarks, OR
- `relora_random_drop` matches or beats ours on val_loss (signal is null).

**AMBIGUOUS** → append to STATUS.md, request PI input, do NOT proceed to B2.

### 2.8 B1 launch order (8 GPU example)

```bash
# Phase B1.a — Llama-3.1-8B SFT (4 parallel)
for METHOD in lora_vanilla dora adalora relora_baseline; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/stage3_run.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset allenai/tulu-3-sft-mixture \
    --method $METHOD --seed 42 \
    > logs/b1_llama_tulu_${METHOD}.log 2>&1 &
done
wait

# Phase B1.b — Llama-3.1-8B ablation arms (4 parallel)
for METHOD in relora_diag_gated_s3pos relora_diag_gated_s3neg relora_random_drop relora_train_gated; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/stage3_run.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset allenai/tulu-3-sft-mixture \
    --method $METHOD --seed 42 \
    > logs/b1_llama_tulu_${METHOD}.log 2>&1 &
done
wait

# Phase B1.c — same 8 methods on MetaMathQA-10k (Llama-3.1)
# Phase B1.d — same 8 methods on Tulu-3       (Qwen3-8B)
# Phase B1.e — same 8 methods on MetaMathQA-10k (Qwen3-8B)
# Phase B1.f — lm-eval across all 32 adapters (8 method x 2 model x 2 dataset)
# Phase B1.g — plots + decision.json + STATUS.md append + commit + tag b1-{pass,stop}
```

Track every job's PID in `STATUS.md` per the protocol in `03_handover §1.4`.

### 2.9 Hard constraints inherited from handover §9

1. Never touch the diagnostic set or test_holdout for training.
2. Never auto-skip a stage gate.
3. Never add EPI (arXiv:2604.14010) as baseline (concurrent work, §9 rule 9).
4. Pin all random seeds (default 42; for B3 ablation, also run seed=1,7,42 for ≥3 cells).
5. Save every per-checkpoint state_dict (`save_steps` such that ≥5 checkpoints per run).
6. Bootstrap CIs (1000 resamples) for any headline metric, not just means.
7. Never mutate shared conda envs.
8. Redirect every parallel job's stdout/stderr to `logs/`.
9. Check `nvidia-smi` before launching; respect other users' cards.

---

## 3. Phase B2 — Stage-2 Weiss reproduction (after B1 PASS, ~250 GPU-h, 5 days)

Only enter B2 if B1 PASSes per §2.7. See `07_missing_experiments §7.2` for the
spec. Key items:

- 11M / 33M / 66M LLaMA-style decoder, 3 methods each (`full_rank`,
  `relora_baseline`, `relora_diag_gated`)
- **5B tokens** on C4 subset (not wikitext-2 — that was the v1 mistake)
- Add **Paloma PPL** evaluation (`allenai/paloma`)
- Add `cumulative_rank.jsonl` to Stage 2 runs (same schema as B1 §2.4)
- Plots: fig5/6/7/8 per `07_missing_experiments §4`

B2 gates: see `07_missing_experiments §10 (DoD)` and original handover §4.5.

---

## 4. Phase B3 — ICLR 2027 polish (after B2 PASS, ~280 GPU-h, 6 days)

See `07_missing_experiments §7.3` + §12. Headline additions for 2026:

1. **Reasoning anchor**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` × `OpenThoughts-114k`
   + `MetaMathQA-10k` mix, evaluated on **AIME-2024 / MATH-500 / MUSR / GPQA-Diamond**.
   New tracked metric: `reasoning_trace_length_retention` (avg CoT length post-SFT
   / pre-SFT, target ≥ 0.8 means we don't break reasoning).
2. **Small-model ablation grid** on `Qwen/Qwen3-1.7B` (or `google/gemma-3-4b-it`):
   merge_every ∈ {500,1000,2000,4000}; drop_ratio ∈ {25,50,75}%;
   rank r ∈ {8,16,32,64}; saliency_batches ∈ {4,8,16,32}.
3. **Upgrade benchmarks**: add MMLU-Pro and HumanEval+ across Phase B1's main table.
4. **Concurrent work check** (per §12.5): re-search arxiv 2025-09 to 2026-08 for
   "diagnostic LoRA" / "val-gated PEFT" / "saliency-based LoRA pruning". If any
   collision found, append to STATUS.md and adjust framing.

---

## 5. Phase B1.5 (OPTIONAL, only if you finish B1+B2+B3 with time to spare)

Self-reimplement the 4 baselines that have no public code, in priority order:

1. **Sensitivity-LoRA** (arXiv 2509.09119) — most critical must-beat.
2. **CTR-LoRA** (arXiv 2510.15962) — trust-region regularizer arm.
3. **COLA** (arXiv 2401.04151) — Frank-Wolfe residual learning.
4. **PrunedLoRA** (arXiv 2510.00192) — gradient-based structured pruning.

For each: read `baselines/<name>_reimpl/PAPER.pdf`, write
`baselines/<name>_reimpl/IMPLEMENTATION_NOTES.md` first (algorithm box + hyper
table + plug-in points), then implement, then rerun ONLY the main table cells
needed (Llama-3.1-8B + Tulu-3 + 5 benchmarks). Mark each method as
"our re-implementation" in the paper.

---

## 6. Reproducibility deliverables (before paper submission)

By the time you finish B3, the repo must contain:

- [ ] `reproduce_all.sh` — one-button replay of B1+B2+B3 (modulo wall-clock)
- [ ] `requirements_b1.txt` / `requirements_b2.txt` / `requirements_b3.txt` —
      pinned to single commits / versions
- [ ] `results/cost_table.csv` — wall-clock and peak GPU-mem for every cell
- [ ] `results/<stage>/decision.json` for each stage gate
- [ ] All `*.plot.json` source files alongside their PNGs
- [ ] `STATUS.md` final summary section
- [ ] Final tagged commit: `iclr-2027-submission`

---

## 7. Communication protocol with PI (the human)

Write to `STATUS.md` (append-only, dated entries) at these moments:

| Trigger | What to write |
|---|---|
| B1 first method × dataset cell done | first-impression table, any surprises |
| B1 all SFT done, eval starting | summary of train/val_loss separation |
| B1 fully done | decision.json + 1-page rationale |
| B2 11M baseline reproduces Weiss failure | "Weiss failure reproduced" + ER curve link |
| B2 fully done | decision.json + rationale |
| B3 reasoning-anchor cell done | reasoning trace length numbers (must be reported) |
| Any STOP gate triggered | reason + partial results + 3 candidate next-steps |
| Any unanticipated failure | freeze, write up, request PI input |
| Concurrent-work hit found (§4.4) | freeze framing, request PI input |

Do not silently skip stages. Do not silently change saliency formulas. Do not
silently swap models/datasets — if a planned one fails (e.g., HF model gated),
write STATUS.md entry with the substitution and rationale, then proceed.

---

## 8. Quick sanity-checks before launching B1

```bash
# 1. Reproduce Stage 0 smoke to confirm env is sane
CUDA_VISIBLE_DEVICES=0 python scripts/stage0_smoke.py --config configs/stage1_sst2.yaml --smoke
# Expect dev acc ≥ 92.0% on SST-2

# 2. Tiny dry-run of B1 method arm on Llama-3.1 with 10 SFT steps to validate
#    new code paths (dora / adalora / random_drop / train_gated)
python scripts/stage3_run.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset allenai/tulu-3-sft-mixture \
  --method dora --seed 42 --dry_run_steps 10

# 3. Confirm lm-eval-harness pinned commit works on a base model first
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
  --tasks gsm8k --num_fewshot 5 --limit 50 --batch_size auto
# Expect base Llama-3.1-8B-Instruct GSM8K 5-shot ≈ 60-65%
```

If any of these fails, STOP and write STATUS.md before launching the full B1.

---

## 9. Decision authority

You are autonomous within these batches. Decisions you make on your own:
- GPU scheduling / job ordering
- Code-level implementation choices (logging, batching, sub-routines)
- Substituting a method's hyperparameter within ±20% if the original is unstable

Decisions you MUST escalate to PI via STATUS.md:
- Any stage gate STOP or AMBIGUOUS
- Concurrent-work collision in §4.4 check
- Cost overrun > 1.5× the per-batch budget
- License / data access issues (e.g., a HF model becomes gated)
- Need to add an unplanned baseline or remove a planned one

---

**End of B1-launch prompt. Target: ICLR 2027.**
**Start by reading §1.3 docs, then write your first STATUS.md entry.**
