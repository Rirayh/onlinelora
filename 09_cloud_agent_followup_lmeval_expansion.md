# Cloud GPU Agent — Follow-up: lm-eval expansion + diagnostics + B1 completion

> **For**: cloud GPU agent
> **Predecessor**: `08_cloud_agent_prompt_b1_iclr2027.md` (still binding for §1, §6, §7, §9)
> **Status of B1 as of 2026-05-15 12:15** (per STATUS.md): partial — 1 model × 1 dataset cell complete (Qwen3-8B × Tulu-3 SFT, lm-eval on 3 tasks); cross-model exploration (Mistral-7B, Qwen2.5-7B × MetaMathQA) done, no lm-eval yet.
> **Today**: 2026-05-15. **Target venue: ICLR 2027** (deadline ~2026-09/10).

---

## 0. Why this follow-up exists

PI reviewed the partial Qwen3-8B + Tulu-3 lm-eval results and identified a real concern that must be evidence-resolved before B1 PASS:

| Method | GSM8K | ARC-C | HellaSwag |
|---|---|---|---|
| **S3pos** | **87.95%** | 66.13% | 76.09% |
| S3neg | 86.88% | 67.15% | 77.82% |
| random_drop | 86.43% | 67.24% | 77.14% |

**Concern**: On ARC-C and HellaSwag, S3pos is **slightly worse** than random_drop and S3neg.
On GSM8K, S3pos beats baseline by +7.89pp — so the method is clearly working on-distribution.

**This is the core question reviewers will ask. The narrative is workable
("on-target specializer with mild OOD tradeoff") IF AND ONLY IF we have evidence that:**

1. **The OOD differences are within bootstrap CI** (95%) — otherwise they are real degradation that needs explanation.
2. **The on-distribution win generalizes to MMLU-Pro / BBH / MATH-500** — these are CoT-heavy benchmarks where S3pos should also win by a wide margin.
3. **S3pos and S3neg actually do different things mechanically** (drop different layers / components) — otherwise the sign-convention story is hollow.
4. **Sign convention task-specificity from Stage 1 reproduces at SFT scale** — the IFEval / TruthfulQA / per-MMLU-domain breakdown is where this shows.

This document specifies the experiments and analyses needed to either confirm
the narrative or trigger an honest rewrite.

---

## 1. First actions (do these before any new training)

```bash
# 1.1 Pull latest
cd /mnt/cpfs/junlongke/onlinelora/lora_obd
git pull origin main
git log --oneline -5

# 1.2 Read THIS doc end-to-end before doing anything
cat 09_cloud_agent_followup_lmeval_expansion.md   # this file

# 1.3 Confirm GPU availability
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv

# 1.4 Confirm what was already done so we don't redo it
ls results/stage3_v2/qwen3-8b/tulu3-sft/*/seed42/summary.json
ls results/stage3_v2/{mistral-7b,qwen25-7b}/metamathqa-10k/*/seed42/summary.json
ls results/stage3_v2/qwen3-8b/tulu3-sft/*/seed42/lm_eval/   # only 3 task files exist
```

Then write a STATUS.md entry: "F1 follow-up start, env probe, current adapters
inventory" with one line per existing adapter.

---

## 2. Task F1 — Fix train_gated OOM and complete the 1.5×-stability ablation

**Background**: `relora_train_gated` OOMed at step 2500 of Qwen3-8B + Tulu-3
because it uses raw train batches (long Tulu-3 sequences) for saliency backward.
S3pos uses val batches that are shorter.

### Fix (in `scripts/stage3_run.py` around line 758)

Add a CLI arg and use it inside the saliency call:

```python
# in argparse
p.add_argument("--saliency_max_seq_len", type=int, default=512,
               help="Hard cap on seq_len for saliency batches; reduces OOM risk for long-form train data.")

# around line 758 where sal_loader is built
sal_loader = diag_loader if saliency_source == "val" else train_loader
# NEW: rebuild a length-capped subset for saliency only
if args.saliency_max_seq_len < args.seq_len:
    sal_loader = build_truncated_loader(sal_loader, max_len=args.saliency_max_seq_len)
```

Implement `build_truncated_loader` near the data loaders: it wraps the source
loader, truncates input_ids/labels/attention_mask to `max_len`, and yields the
same schema. Pin `args.saliency_max_seq_len = 512` for `train_gated`,
keep `2048` (or current value) for `S3pos / S3neg`.

### Re-run train_gated cell

```bash
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
QWEN=/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B
ROOT=/mnt/cpfs/junlongke/onlinelora/lora_obd
OUT=$ROOT/results/stage3_v2/qwen3-8b/tulu3-sft/relora_train_gated/seed42

# Backup the partial run
mv $OUT ${OUT}_oom_partial 2>/dev/null

CUDA_VISIBLE_DEVICES=0 $PY scripts/stage3_run.py \
    --model_path $QWEN --model_key qwen3-8b --dataset tulu3-sft \
    --method relora_train_gated --total_steps 3000 \
    --merge_every 500 --eval_every 250 --ckpt_every 50 \
    --saliency_max_seq_len 512 \
    --save_adapter --seed 42 \
    --out_root $OUT \
    > $ROOT/logs/b1/qwen3-tulu3-relora_train_gated_rerun.log 2>&1 &
```

Verify it completes 3000 steps without OOM. If it OOMs again, halve to 256.

---

## 3. Task F2 — Launch B1 Batches 2 / 3 / 4 (planned but not started)

The cross-model exploration has filled some cells unplanned. Re-plan to fill the
ICLR-grade main table per `08_cloud_agent_prompt §2.8`:

### Required cells (for main table)

| model | dataset | methods needed |
|---|---|---|
| qwen3-8b | tulu3-sft | ✅ 7/8 done; train_gated rerun via §2 |
| qwen3-8b | metamathqa-10k | **8 needed** (Batch 2) |
| llama3-8b | tulu3-sft | **8 needed** (Batch 3) |
| llama3-8b | metamathqa-10k | **8 needed** (Batch 4) |
| mistral-7b | metamathqa-10k | ✅ 4 done (lora_vanilla, baseline, S3pos, random_drop); add S3neg + adalora |
| qwen25-7b | metamathqa-10k | ✅ 4 done; add S3neg + adalora |

Total new SFT runs needed: 24 (B2/B3/B4) + 4 (cross-model fill) = 28 jobs.

### Launch order (8 GPU rotation)

Use the bash blocks already drafted in STATUS.md (search "LAUNCH BATCH 2", "BATCH 3", "BATCH 4"). For the cross-model fill:

```bash
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
ROOT=/mnt/cpfs/junlongke/onlinelora/lora_obd
declare -A MP=(
  [mistral-7b]=/mnt/cpfs/public_data/public_model/Mistral/Mistral-7B-v0.3
  [qwen25-7b]=/mnt/cpfs/public_data/public_model/Qwen2.5/Qwen2.5-7B-Instruct
)
i=0
for MODEL in mistral-7b qwen25-7b; do
  for METHOD in relora_diag_gated_S3neg adalora; do
    OUT=$ROOT/results/stage3_v2/$MODEL/metamathqa-10k/$METHOD/seed42
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=$i $PY scripts/stage3_run.py \
      --model_path ${MP[$MODEL]} --model_key $MODEL --dataset metamathqa-10k \
      --method $METHOD --total_steps 3000 --merge_every 500 \
      --eval_every 250 --ckpt_every 50 --save_adapter --seed 42 \
      --out_root $OUT \
      > $ROOT/logs/b1/${MODEL}-metamath-${METHOD}.log 2>&1 &
    i=$((i+1))
  done
done
```

Apply the OOM fix from §2 to **every** rerun of train_gated and to any future
runs on Tulu-3 with saliency_source=train.

---

## 4. Task F3 — Parallel lm-eval expansion (THE CORE OF THIS FOLLOW-UP)

### 4.1 Tasks to add (current vs. needed)

| Task | Current | Needed | Role |
|---|---|---|---|
| GSM8K 5-shot | ✅ Qwen3+Tulu (6 methods) | extend to all adapters | on-distribution math |
| ARC-Challenge | ✅ Qwen3+Tulu (6 methods) | extend to all | OOD MC reasoning |
| HellaSwag | ✅ Qwen3+Tulu (6 methods) | extend to all | OOD commonsense |
| **MMLU** 5-shot | ❌ | **all adapters** | broad knowledge (per-subject breakdown for diagnostics) |
| **MMLU-Pro** 5-shot CoT | ❌ | **all adapters** | reasoning, anti-saturation |
| **BBH** 3-shot CoT | ❌ | **all adapters** | multi-step reasoning |
| **MATH-500** 0-shot | ❌ | **all adapters** | hard math (or 4-shot if MATH-500 not in your harness; fall back to `hendrycks_math` 4-shot) |
| **HumanEval** 0-shot | ❌ | all adapters with Tulu-3 SFT | code (Tulu-3 has code) |
| **IFEval** 0-shot strict | ❌ | all adapters | instruction-following |
| **TruthfulQA-MC1** 0-shot | ❌ | all adapters | OOD closed-book MC (validates "SFT barely moves these") |

**Total**: 10 tasks × ~14 adapters = 140 cells.

### 4.2 Parallelization strategy: 8 tasks in parallel for ONE adapter at a time

vLLM is not installed; we use HF backend. Per-task wall-clock on Qwen3-8B with
HF backend, batch_size auto, single GPU: gsm8k ~25min, mmlu ~40min, bbh ~30min,
mmlu_pro ~50min, ifeval ~10min, humaneval ~15min, math ~30min, truthfulqa ~5min.

**Strategy**: lock one adapter, fan 8 tasks across 8 GPUs. ~50min total per adapter (bottleneck = mmlu_pro).

```bash
# scripts/run_lmeval_8parallel.sh (NEW — add this script to repo)
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
ROOT=/mnt/cpfs/junlongke/onlinelora/lora_obd
ADAPTER="$1"        # e.g. results/stage3_v2/qwen3-8b/tulu3-sft/relora_diag_gated_S3pos/seed42/checkpoints/best
BASEMODEL="$2"      # e.g. /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B
OUTDIR="$(dirname $ADAPTER)/../lm_eval"
mkdir -p $OUTDIR

declare -a TASKS=(
  "gsm8k"
  "mmlu"
  "mmlu_pro"
  "bbh"
  "math_hendrycks"        # fall back name; check `lm_eval --tasks list | grep math` if needed
  "humaneval"
  "ifeval"
  "truthfulqa_mc1"
)
declare -A FEWSHOT=(
  [gsm8k]=5 [mmlu]=5 [mmlu_pro]=5 [bbh]=3
  [math_hendrycks]=4 [humaneval]=0 [ifeval]=0 [truthfulqa_mc1]=0
)

for i in "${!TASKS[@]}"; do
  T="${TASKS[$i]}"
  CUDA_VISIBLE_DEVICES=$i $PY -m lm_eval --model hf \
    --model_args "pretrained=$BASEMODEL,peft=$ADAPTER,dtype=bfloat16,attn_implementation=sdpa" \
    --tasks $T --num_fewshot ${FEWSHOT[$T]} --batch_size auto \
    --output_path $OUTDIR/${T} \
    --log_samples \
    > $ROOT/logs/lmeval/${ADAPTER##*stage3_v2/}_${T}.log 2>&1 &
done
wait
echo "[done] $ADAPTER all 8 tasks"
```

Save this as `scripts/run_lmeval_8parallel.sh`, chmod +x, then loop over all
adapters serially:

```bash
# Top-level driver
ADAPTERS=$(find $ROOT/results/stage3_v2 -name "best" -type d -path "*/checkpoints/best")
for A in $ADAPTERS; do
  # derive base model from path
  case "$A" in
    *qwen3-8b*) BM=/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B ;;
    *llama3-8b*) BM=/mnt/cpfs/public_data/public_model/Meta-Llama-3-8B ;;
    *mistral-7b*) BM=/mnt/cpfs/public_data/public_model/Mistral/Mistral-7B-v0.3 ;;
    *qwen25-7b*) BM=/mnt/cpfs/public_data/public_model/Qwen2.5/Qwen2.5-7B-Instruct ;;
  esac
  bash scripts/run_lmeval_8parallel.sh "$A" "$BM"
done
```

**Critical**: pass `--log_samples` so each task writes a per-sample jsonl —
needed for bootstrap CI in §5.

If a task name in `lm_eval --tasks list` differs (the harness changed names
between versions), substitute. **Pin `lm_eval==0.4.12`** as STATUS.md says is
installed. Do not upgrade.

### 4.3 If a task can't run (OOM or task name unknown)

- Halve `batch_size` to 4 (or 2) and rerun that single task.
- If task name not in your harness version, swap: `mmlu_pro` ↔ `mmlu_pro_5shot`, `math_hendrycks` ↔ `hendrycks_math` ↔ `math_qa`, `humaneval` ↔ `humaneval_instruct`. Whichever exists.
- Log substitutions to STATUS.md.

### 4.4 Parking lot: optional Big-3 (R1-distill anchor preview)

If §4.2 finishes early, preview B3's reasoning anchor with **just one** cell:

```bash
# Download
$PY -c "from huggingface_hub import snapshot_download; \
  snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', \
    local_dir='/mnt/cpfs/junlongke/onlinelora/models/R1-Distill-Qwen-7B')"

# SFT one cell: R1-Distill-Qwen-7B + MetaMathQA-10k + S3pos, 3000 steps
# Then lm-eval on AIME-2024 + MATH-500 + GPQA-Diamond + MMLU-Pro
```

Skip if §4.2 takes the full GPU window.

---

## 5. Task F4 — Bootstrap CI on every cell (REQUIRED for paper, no exceptions)

Without 95% CI, all the close numbers (HellaSwag 1.7pp, ARC-C 1.1pp) are
unfalsifiable. Add this script:

### `scripts/bootstrap_ci.py` (NEW)

```python
# Reads lm_eval --log_samples jsonl, computes mean ± 95% CI via 1000 bootstrap.
import argparse, json, glob, os, numpy as np
from collections import defaultdict

def metric_value(sample):
    # lm-eval samples write per-task; for accuracy-style we read 'acc' or 'exact_match'
    for k in ("acc", "exact_match", "acc_norm", "pass@1"):
        if k in sample:
            return float(sample[k])
    raise KeyError(f"no metric in {list(sample.keys())[:6]}")

def bootstrap_ci(values, n=1000, q=(2.5, 97.5)):
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(42)
    boots = np.array([
        rng.choice(arr, size=arr.size, replace=True).mean()
        for _ in range(n)
    ])
    return float(arr.mean()), float(np.percentile(boots, q[0])), float(np.percentile(boots, q[1]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/stage3_v2")
    p.add_argument("--out", default="results/stage3_v2/summary/bootstrap_ci.csv")
    args = p.parse_args()

    rows = []
    for samples_jsonl in glob.glob(f"{args.root}/**/lm_eval/**/samples_*.jsonl", recursive=True):
        # Path: .../stage3_v2/<model>/<dataset>/<method>/seed42/lm_eval/<task>/samples_<task>.jsonl
        parts = samples_jsonl.split("/")
        model, dataset, method = parts[-7], parts[-6], parts[-5]
        task = parts[-2]
        with open(samples_jsonl) as f:
            vals = [metric_value(json.loads(l)) for l in f]
        if not vals:
            continue
        mean, lo, hi = bootstrap_ci(vals)
        rows.append((model, dataset, method, task, len(vals), mean, lo, hi))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("model,dataset,method,task,n,mean,ci_lo,ci_hi\n")
        for r in sorted(rows):
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"[done] wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
```

Run after §4 finishes:
```bash
$PY scripts/bootstrap_ci.py --root results/stage3_v2 \
    --out results/stage3_v2/summary/bootstrap_ci.csv
```

**Acceptance**: every cell in main_table.csv has columns `mean ± [ci_lo, ci_hi]`.
If S3pos vs random_drop CIs overlap on HellaSwag, write "statistically tied" in the paper, not "S3pos worse".

---

## 6. Task F5 — Diagnostic analyses (this is what distinguishes S3pos from S3neg)

The point of these analyses: produce evidence for the **mechanism** behind the
sign-convention story, not just the outcome.

### 6.1 `scripts/analyze_dropped_components.py` (NEW)

Reads `dropped_components.jsonl` from each method × cell, produces:

- **Per-method total drop count** (sanity check that drop rates are comparable)
- **Per-layer drop heatmap**: heatmap with rows = layers (e.g., 32 layers × 7 modules), cols = methods (S3pos / S3neg / random_drop / train_gated). Values = total drops over the run.
- **Jaccard similarity** between dropped sets across methods on the same (model, dataset, seed): a 4×4 matrix per cell.
- **Saliency-direction alignment**: for each merge event, scatter S3_signed (val) vs S2_signed (train) component-wise — proves the gate signals are different (not collinear).

Output: `results/stage3_v2/summary/dropped_components_analysis.json` + 4 PNGs in `plots/stage3_v2/`:
- `fig_drop_heatmap_per_layer.png`
- `fig_drop_jaccard_matrix.png`
- `fig_saliency_train_vs_val_scatter.png`
- `fig_drop_count_per_method_bars.png`

### 6.2 `scripts/analyze_cot_length.py` (NEW)

For GSM8K and MATH samples (from `--log_samples` jsonl), extract per-sample
generation token count, compute mean CoT length per (model, dataset, method).

Hypothesis to verify: **S3pos generates shorter, more focused CoTs than S3neg/random**.
If true, this is a real mechanistic finding — diagnostic gate concentrates reasoning.

Output: `results/stage3_v2/summary/cot_length.csv` + bar plot
`plots/stage3_v2/fig_cot_length_per_method.png`.

### 6.3 `scripts/analyze_mmlu_per_domain.py` (NEW)

MMLU has 57 subjects in 4 categories: STEM / Humanities / Social Sciences / Other.
Read `samples_mmlu.jsonl`, group by subject category, compute per-category mean.

Hypothesis to verify: **S3pos > random on STEM, ≈ on Humanities/Social/Other**.
If true, this is exactly the "specialization vs preservation" tradeoff the
narrative claims.

Output: `results/stage3_v2/summary/mmlu_per_domain.csv` + grouped bar plot
`plots/stage3_v2/fig_mmlu_per_domain.png`.

### 6.4 `scripts/analyze_active_vs_cumulative_rank.py` (NEW)

Read `cumulative_rank.jsonl` for each method × cell. For S3pos / S3neg /
random_drop / baseline, plot:

- x: training step (or merge event count)
- y_left: active-LoRA rank (= r=16 constant for all merge methods)
- y_right: cumulative rank of merged Δ_stable (computed via SVD of accumulated update)

Lines: one per method. **This is fig9, the narrative core.**

Output: `plots/stage3_v2/fig9_active_vs_cumulative_rank.png` + JSON source.

### 6.5 `scripts/build_main_table.py` (NEW)

Joins `bootstrap_ci.csv` + `cot_length.csv` + `summary.json` (val_loss, ABORTED
flag, wall_clock) into one wide CSV:

`results/stage3_v2/summary/main_table.csv`

Columns: model, dataset, method, val_loss, gsm8k, mmlu, mmlu_pro, bbh,
math_hendrycks, humaneval, ifeval, truthfulqa_mc1, hellaswag, arc_c,
cot_len_mean, aborted, wall_clock_h.

This becomes Table 1 of the paper directly.

---

## 7. Task F6 — Sign convention probe: relora_diag_gated_S3abs

To validate H3 (sign convention is task-specific) at SFT scale:

Add a 9th method arm `relora_diag_gated_S3abs`:
- Uses |S3_fo_val_signed| as the magnitude criterion
- Drops the top-k components by absolute value (no signed convention)
- Otherwise identical to S3pos

In `scripts/stage3_run.py` add to `METHOD_CHOICES` and route in `build_keep_mask`
with `gate_sign="S3abs_drops"`.

Run **one cell only** for now: Qwen3-8B + Tulu-3 + S3abs + seed=42 + 3000 steps.
Then full lm-eval per §4.2.

**Expected outcome interpretation**:
- If S3abs ≈ random on val_loss but ≈ S3pos on GSM8K → **the magnitude is the signal, sign just refines** (H3 weak)
- If S3abs ≈ S3pos on all benchmarks → **sign convention is unnecessary**, kill the signed gate altogether (rewrite would be required)
- If S3abs ≈ S3neg on val_loss → **sign convention is essential, our framing holds** (H3 strong)

---

## 8. Task F7 — Multi-seed verification (only after F1–F6 done)

Once main table is filled and all CIs computed, pick the **2 most contested
cells** (anywhere with overlap between S3pos and another method) and rerun with
seeds 1 and 7. Compare 3-seed mean and CI spread.

Default cells to triple-seed:
- `qwen3-8b/tulu3-sft/relora_diag_gated_S3pos` and same with `relora_baseline`
- `mistral-7b/metamathqa-10k/relora_diag_gated_S3pos`

Output: `results/stage3_v2/summary/multiseed_consistency.csv` (rank ordering of
methods across 3 seeds).

---

## 9. Task ordering, GPU budget, milestones

| Order | Task | Wall-clock | GPU-h |
|---|---|---|---|
| 1 | F1 (train_gated rerun, 1 cell) | 3h | 3 |
| 2 | F2 (Batch 2/3/4 + cross-model fill = 28 SFT runs) | 12h (parallel 8) | 96 |
| 3 | F3 (lm-eval 14 adapters × 8 tasks parallel) | 14 × 50min = ~12h | 96 |
| 4 | F4 (bootstrap CI script + run) | 0.5h CPU | 0 |
| 5 | F5 (diagnostic analyses, 5 scripts + plots) | 2h CPU | 0 |
| 6 | F6 (S3abs probe: 1 SFT + 8 lm-eval tasks) | 4h | 4 |
| 7 | F7 (multi-seed, 6 cells × 1.5h ≈ 9h on 6 GPU) | 9h | 54 |

**Total**: ~50 GPU-h additional (well within B1 budget). End-to-end ~3 days
with idle/queue overhead.

### Hard checkpoints

- **CHECKPOINT 1 (F1+F2 done)**: STATUS.md entry "all 32 main-table SFT cells complete or salvaged"
- **CHECKPOINT 2 (F3+F4 done)**: `main_table.csv` with all 14 adapters × 8 tasks + 95% CI
- **CHECKPOINT 3 (F5 done)**: 4 diagnostic plots in `plots/stage3_v2/`
- **CHECKPOINT 4 (F6 done)**: S3abs cell complete, sign-convention determination written to STATUS.md
- **CHECKPOINT 5 (F7 done)**: multi-seed table, B1 PASS/STOP decision.json written

At each checkpoint, `git add . && git commit -m "checkpoint: ..." && git push origin main`.
If push is rejected by ruleset, go through a PR or notify the PI in STATUS.md.

---

## 10. Updated B1 PASS/STOP gates (supersedes 08_cloud_agent §2.7 for benchmark thresholds)

The original gate ("S3pos beats baseline+train_gated on ≥2 of 5 benchmarks") was
written before we observed the OOD softness on ARC-C/HellaSwag. The updated
gate has three tiers:

### B1 PASS (strong) — claim main paper

**ALL of**:
1. `relora_diag_gated_S3pos` beats `relora_baseline` on **GSM8K, MMLU-Pro, BBH, MATH-500** by ≥ +1.0pp on **at least 2 of 4 (model, dataset) cells**.
2. `relora_diag_gated_S3pos` beats `relora_random_drop` on **GSM8K + at least 2 of {MMLU, MMLU-Pro, BBH, MATH-500}** with **non-overlapping 95% CI** on at least 1 cell.
3. `relora_diag_gated_S3pos` does not lose to `relora_random_drop` on **HellaSwag, ARC-C, TruthfulQA-MC1** by more than the 95% CI half-width on more than 1 cell.
4. `cumulative_rank` is monotone in ≥ 80% of layers across all `relora_diag_gated_S3pos` runs.
5. `dropped_components` Jaccard between S3pos and random_drop ≤ 0.4 (i.e., they really select different components).

### B1 PARTIAL — claim narrowed paper (specialization narrative)

If 1+2+5 PASS but 3 fails (S3pos significantly underperforms on OOD), narrative becomes:
> "Diagnostic-gate ReLoRA is a controllable on-distribution specializer. We show that signed val-saliency selects components useful for the diagnostic distribution; off-distribution we observe a measurable specialization tradeoff. We characterize this tradeoff and provide guidance on diagnostic set choice."

This is still ICLR-worthy; just reframe.

### B1 STOP — back to drawing board

If 1 fails (S3pos does not beat baseline on the on-distribution benchmarks
at scale), the method is not delivering. Write `results/stage3_v2/decision.json`
with `{go: false, reason: "..."}` and request PI input.

### Sign convention sub-gate (orthogonal)

After F6:
- If S3abs ≈ S3pos on all benchmarks → kill sign convention from method, simplify paper.
- If S3abs ≠ S3pos in expected direction → keep signed gate as core contribution.

---

## 11. Communication / STATUS.md protocol (continued)

Append to STATUS.md at each:
- F1 train_gated rerun finishes (success or another OOM with details)
- F2 each batch finishes
- F3 each adapter completes 8-task lm-eval (one line: `[F3] adapter=X done in Yh`)
- F3 fully done (table summary)
- F4 CI computed (one-liner with example overlap finding for HellaSwag)
- F5 each plot generated
- F6 S3abs decision (3-line interpretation)
- F7 multi-seed done (rank stability summary)
- Final B1 PASS/PARTIAL/STOP determination

Keep STATUS.md append-only, dated entries, never edit past entries.

---

## 12. Hard constraints (continued from 08_cloud_agent §2.9)

1. Never delete existing `results/stage3_v2/<...>` directories — back up to `*_oom_partial` or similar suffix.
2. Never upgrade `lm_eval`, `peft`, `transformers`, `torch` in the shared env. Use `--target` install if you need a different version of any tool, with virtualenv path documented in STATUS.md.
3. If `--log_samples` doubles disk usage, prune oldest logs > 14 days, keep all jsonl outputs from the current B1.
4. Pin `lm_eval` commit hash in `requirements_b1.txt` and write the hash to STATUS.md after first run; never change it for the rest of B1+B2+B3.
5. Multi-seed (F7) only after F1–F6 are complete; do not interleave seeds with main table cells (avoids confounding).
6. If you run out of disk on `/mnt/cpfs/junlongke/`, **do not** delete adapters — request PI input.

---

## 13. Final deliverables checklist (B1 closure)

- [ ] `scripts/stage3_run.py` includes 9 method arms (8 + S3abs) with `--saliency_max_seq_len` flag
- [ ] `scripts/run_lmeval_8parallel.sh`
- [ ] `scripts/bootstrap_ci.py`
- [ ] `scripts/analyze_dropped_components.py`
- [ ] `scripts/analyze_cot_length.py`
- [ ] `scripts/analyze_mmlu_per_domain.py`
- [ ] `scripts/analyze_active_vs_cumulative_rank.py`
- [ ] `scripts/build_main_table.py`
- [ ] `results/stage3_v2/summary/main_table.csv` — full main table with CI
- [ ] `results/stage3_v2/summary/bootstrap_ci.csv`
- [ ] `results/stage3_v2/summary/dropped_components_analysis.json`
- [ ] `results/stage3_v2/summary/cot_length.csv`
- [ ] `results/stage3_v2/summary/mmlu_per_domain.csv`
- [ ] `results/stage3_v2/summary/multiseed_consistency.csv`
- [ ] `results/stage3_v2/decision.json` (B1 PASS / PARTIAL / STOP + sign-convention determination)
- [ ] `plots/stage3_v2/fig9_active_vs_cumulative_rank.png` (+ `.plot.json`)
- [ ] `plots/stage3_v2/fig_drop_heatmap_per_layer.png`
- [ ] `plots/stage3_v2/fig_drop_jaccard_matrix.png`
- [ ] `plots/stage3_v2/fig_saliency_train_vs_val_scatter.png`
- [ ] `plots/stage3_v2/fig_drop_count_per_method_bars.png`
- [ ] `plots/stage3_v2/fig_cot_length_per_method.png`
- [ ] `plots/stage3_v2/fig_mmlu_per_domain.png`
- [ ] STATUS.md fully updated with all checkpoints
- [ ] Git commits + push at every checkpoint with descriptive messages
- [ ] Final tag: `git tag b1-{pass,partial,stop} && git push origin b1-...`

---

## 14. After B1 closure → B2 (Stage-2 Weiss reproduction)

See `08_cloud_agent_prompt §3` and `07_missing_experiments §7.2`. Do not start
B2 until B1 PASS or PARTIAL is signed off.

---

**End of follow-up. Start by reading §1 and §2, then write a STATUS.md entry.**
