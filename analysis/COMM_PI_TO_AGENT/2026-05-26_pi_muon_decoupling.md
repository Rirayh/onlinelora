# PI → Agent Directive — 2026-05-26: Muon Decoupling Experiment (Saliency Last Stand)

## Executive summary

After exp_v1 + P0 re-eval, the data is clear:

> **`drop_rate=0` ≈ `vanilla LoRA`. `drop_rate>0` gives +3-6pp on Qwen3-8B, but
> `random` ≈ `S3pos`. The "saliency story" is dead in its current form.**

PI hypothesis: **drop is doing _implicit step-size regularization_, which
saturates the optimizer's regularization budget, masking any signal-selection
contribution from saliency.** If we replace AdamW with **Muon** (which provides
explicit spectral regularization on the update), the regularization confound
is removed, and saliency may finally separate from random.

This is **the last well-defined experiment** that can save the saliency story.
If it fails, we pivot to "partial-merge regularization" as the paper's core
finding.

---

## Goals

1. **Confirm/reject** PI hypothesis: drop = step-size regularization
2. **Decouple** "regularization effect" vs "information-selection effect"
3. **Decide** the paper pivot: keep saliency centerpiece OR pivot to drop-rate

---

## STOP / DEFER list (do these LAST or not at all this cycle)

- ❌ Multi-seed expansion (seed 0/1) — defer until Exp-2 conclusion
- ❌ Other-model sweep (qwen3-1.7b/4b, qwen35-*) — defer
- ❌ OOD eval (MMLU/BBH/TriviaQA) — defer
- ❌ Wave 1 stragglers (qwen35-4b/9b S3pos cells) — let them finish in
  background; do NOT prioritize new ones
- ❌ Any new model addition (gemma3/llama3/r1-distill new runs) — defer

If you find yourself launching anything not in the GO list below, STOP.

---

## GO list (do these in order)

### 🥇 Exp-1: AdamW drop_rate sweep (6 cells, ~18 GPU-hours)

**Goal**: confirm drop is regularization, independent of selection.

```yaml
model: qwen3-8b
dataset: tulu3-sft
optimizer: adamw  (current default)
selection: random  (always random for this sweep)
keep_B_after_merge: false  (vanilla ReLoRA reset)
saliency_source: n/a  (random doesn't use saliency)
seed: 42

drop_rate ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 0.9}
```

**Important details**:
- `drop_rate=0.0` MUST match `relora_baseline` (sanity check; if doesn't
  match within 0.5pp, there's a code bug — flag it)
- All 6 cells use vLLM-on-merged eval
- Output a plot `analysis/exp_drop_rate_sweep_qwen3-8b.png`:
  X = drop_rate, Y = {gsm8k_flex, hellaswag, arc_c}, error bars from
  bootstrap 95% CI on lm_eval samples (use existing
  `scripts/bootstrap_ci.py` if available)

**Decision rule** (for whether to proceed to Exp-2):
- If gsm8k_flex shows monotonic OR inverted-U trend with peak somewhere in
  (0, 1) → H_α confirmed → proceed Exp-2
- If gsm8k_flex is flat across drop_rate → H_α rejected, drop-rate isn't the
  story → STOP, write up findings, request PI review

### 🥇 Exp-2: Muon × drop × selection (8 cells, ~24 GPU-hours)

**Goal**: with regularization-providing Muon, does saliency finally separate?

#### Step 1: Implement Muon support in `scripts/stage3_run.py`

Reference implementation: <https://github.com/KellerJordan/Muon> (200 lines,
permissive license). Required changes:

1. Add `scripts/muon.py` (vendored Newton-Schulz orthogonalizer; copy with
   attribution comment)
2. Add CLI arg `--optimizer {adamw, muon}`, default `adamw`
3. When `--optimizer muon`:
   - 2D LoRA weights (`lora_A`, `lora_B`) → Muon
   - All other params (1D bias, scaling, embeddings, head) → AdamW
   - This is the standard Muon recipe (see Keller Jordan's nanoGPT-Muon)
4. **Critical**: when `lora_B` is initialized to zero, skip Newton-Schulz on
   that param for the first step (otherwise NaN). Use AdamW fallback for
   that step OR add small noise.
5. Smoke test: 1 cell, qwen3-8b/tulu3, drop_rate=0.5 random, optimizer=muon,
   200 steps. Confirm:
   - No NaN in train_log.jsonl
   - val_loss decreasing
   - GPU memory ≤ AdamW baseline + 5%

Commit smoke test as **Commit M0**: `Muon optimizer integration + smoke pass`.

#### Step 2: Run 8-cell ablation

```yaml
model: qwen3-8b
dataset: tulu3-sft
seed: 42
saliency_source: gsm8k_train  (use OOD-calib variant from Task 3)
keep_B_after_merge: true       (use Task 2 fix)

8 cells = optimizer × drop_rate × selection:
  (adamw, 0.0,  random)        # = relora_baseline
  (adamw, 0.5,  random)        # = relora_random_drop reference
  (adamw, 0.5,  S3pos_keepB_calibgsm8k)  # = our best AdamW variant
  (adamw, 0.0,  S3pos_keepB_calibgsm8k)  # control: saliency without drop
  (muon,  0.0,  random)        # Muon baseline
  (muon,  0.5,  random)        # Muon + random drop
  (muon,  0.5,  S3pos_keepB_calibgsm8k)  # MAIN CELL: Muon + saliency
  (muon,  0.0,  S3pos_keepB_calibgsm8k)  # Muon saliency without drop
```

All vLLM-on-merged eval, save full evidence.

**Decision matrix**:

| Observation | Interpretation | Action |
|---|---|---|
| `Muon+0.5+saliency` > `Muon+0.5+random` by ≥2pp | Saliency works under Muon → story saved | Pivot paper to "saliency requires proper regularization" |
| `Muon+0.5+saliency` ≈ `Muon+0.5+random` (±1pp) | Saliency adds nothing even under Muon | Pivot paper to "partial-merge regularization" finding |
| `Muon+0.0+saliency` > `Muon+0.5+saliency` | Drop hurts under Muon (over-reg) | Confirms Muon absorbs the reg — but doesn't save saliency |
| `Muon+0.0+random` ≈ `AdamW+0.5+random` | Muon's reg = drop's reg | Strong support for H_ε |

#### Output

`analysis/exp_muon_decoupling_qwen3-8b.md` with:
- 8-cell results table (gsm8k_strict, gsm8k_flex, hswag, arc_c)
- Interpretation per the matrix above
- Picked path (saliency-saved OR drop-rate-pivot) with 1-paragraph
  justification

---

### 🥉 Exp-3 (only if PI requests, NOT in this cycle): Schedule sweep

`total_steps ∈ {3000, 6000}` × `{random, saliency}` — defer until Exp-2 done.

---

## Reporting cadence (this cycle)

- **Push every 4 hours** as before
- Commit titles must match: `"Exp-1: N/6 done | Exp-2: M/8 (smoke S0)"`
- After Exp-1 completes (ETA ~24h from start), open
  `analysis/COMM_AGENT_TO_PI_<ts>_exp1_done.md` with:
  - 6-cell table
  - Plot path
  - PI decision request: "proceed to Exp-2?" (default YES unless flat)

## Hard rules (unchanged)

- vLLM-on-merged for all evals
- Save dropped_components.jsonl, saliency_at_merge.jsonl, effective_rank.jsonl,
  condition_number.jsonl, train_log.jsonl, lm_eval/, AND new:
  `optimizer_metadata.json` (record optimizer type + Muon NS-iteration count)
- Multi-seed reserved for AFTER Exp-2 conclusion

## ACK

In your next push commit body include:
```
ACK: PI directives 2026-05-26 (Muon decoupling)
```

If any item is unclear or if you see a flaw in the experimental design,
write `analysis/COMM_AGENT_TO_PI_<ts>_muon_concerns.md` BEFORE starting,
listing 2-3 specific concerns. PI will respond within 4-hour cadence.

---

## Why this is the right experiment

PI's reasoning chain:
1. We see `random ≈ saliency` empirically. This means either (a) saliency
   signal is genuinely uninformative, or (b) saliency contribution is being
   masked by a confound.
2. The most plausible confound is drop = implicit regularization, which gives
   AdamW + drop=0.5 a "free" ~5pp boost regardless of selection.
3. Muon provides explicit regularization, removing the confound. If saliency
   is real, this is where it shows up. If not, we have proof that the
   selection criterion truly doesn't matter.
4. Either outcome is a publishable finding — not all bets win, but this one
   is well-controlled.

This experiment design is the **single most diagnostic test we can run** in
under 2 GPU-days. Do NOT scope-creep.
