# LoRA OBD-Recycling: GPU Agent Handover

> **Audience**: Autonomous coding agent on a GPU machine.
> **Goal**: Run the two kill-or-seal experiments that decide whether the val-Hessian-gated LoRA pipeline is worth pursuing.
> **Scope**: This document is self-contained. You should not need to ask back unless explicitly instructed below.
> **Reference docs (read first, in order)**:
> 1. `01_research_v1.md` (v1, 712 lines): method formulation, design choices, broad baselines.
> 2. `02_research_v2_baselines_theory.md` (v2): updated white-space, latest baselines (Sensitivity-LoRA, CTR-LoRA), 3 theoretical viewpoints, **concurrent work note on EPI (§1.7, treat as parallel — not a baseline)**, and the **two kill-or-seal experiments** (§5).
> 3. This document: how to actually run those experiments.

---

## 0. Mission summary (read once, internalize)

**Hypothesis**: Saliency of LoRA rank-1 components computed on a **held-out diagnostic set** (val-side) is a **better predictor** of which components, if removed, would **improve held-out test loss**, than saliency computed on the training set. If this holds, it justifies the entire "diagnostic-gated prune-merge-rotate" pipeline.

**You will execute three stages in order**:

| Stage | What | Wall-clock (with parallelism, ≥3× A100-80G) | Total GPU-hours | Decision output |
|---|---|---|---|---|
| **Stage 0** | Environment probe + smoke test | 1–2 h | 1–2 | env_ready: bool |
| **Stage 1** | Predictive validity of val saliency vs train saliency | **~5 h** (3 tasks parallel) | ~15 | go_for_stage_2: bool |
| **Stage 2** | Reproduce ReLoRA failure mode + diagnostic fix | **~36 h** (fan-out on 6–9 GPUs) | ~250 | method_works: bool |
| **Stage 3** (optional, only if Stage 2 passes) | Scale to Llama-3.1-8B SFT | **~24 h** (8 methods parallel on 8 GPUs) | ~192 | scales_up: bool |

**Parallelism mandate**: this machine has multiple A100-80G cards. **Run experiments in parallel wherever the dependency graph allows.** Specifically:
- Stage 1: 3 tasks (SST-2 / MRPC / RTE) on 3 GPUs in parallel — no inter-task dependency.
- Stage 2 Phase A (11M smoke): 3 methods (full_rank / relora_baseline / relora_diag_gated) in parallel on 3 GPUs. Phase B (33M + 66M): fan out 6 (size × method) jobs across all free GPUs.
- Stage 3: 8 methods on 8 GPUs in parallel.

See §11 for the parallelism / scaling table.

**Hard rule**: If Stage 1 fails (decision criteria in §3.8), DO NOT auto-proceed to Stage 2. Stop and write a final report. Stage 2 burns much more compute and must be gated.

**Soft rule**: At the end of each stage, commit results and write a 1-paragraph status update to `STATUS.md` at the repo root.

---

## 1. Environment setup (Stage 0a)

### 1.1 Hardware assumptions

- **Target machine**: cluster node with **multiple A100 80GB cards**. Use **all available cards** unless instructed otherwise.
- Check available GPUs first: `nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv`. Plan parallel runs based on free memory.
- **Default parallelism strategy**:
  - Stage 1: each task (SST-2 / MRPC / RTE) runs on a separate GPU → 3 tasks finish in parallel. If more GPUs are free, also parallelize across checkpoints (sequential within a task is fine; the bottleneck is the oracle ablation, see §3.5).
  - Stage 2: each model size (11M / 33M / 66M) on its own GPU(s) in parallel.
  - Stage 3: distribute the 8 methods across GPUs in parallel where memory allows.
- Disk: ≥ 200 GB free for HF cache + checkpoints. Use a shared `HF_HOME=/mnt/cpfs/<user>/hf_cache` if available so multiple runs share downloads.
- RAM: ≥ 64 GB.

### 1.2 Conda env (reuse if already configured)

**Strategy**: **prefer reusing an existing env over creating a new one**. The cluster may already have a usable env; check before building.

```bash
CONDA=/mnt/cpfs/junlongke/miniconda3/bin/conda

# Step 1: list existing envs
$CONDA env list

# Step 2: probe candidate envs for compatibility. Look for an env that already has
# torch >= 2.3, transformers >= 4.40, peft >= 0.10, accelerate, datasets.
# Common names to try in priority order:
for ENV in lora-obd peft-env llm-sft torch24 base; do
  if $CONDA env list | grep -qE "^${ENV}\s"; then
    echo "Probing $ENV ..."
    source /mnt/cpfs/junlongke/miniconda3/bin/activate "$ENV"
    python - <<'PY'
import importlib, sys
need = {"torch":"2.3", "transformers":"4.40", "peft":"0.10",
        "accelerate":"0.30", "datasets":"2.18", "scipy":None, "matplotlib":None}
ok = True
for pkg, minv in need.items():
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "?")
        print(f"  {pkg}={v}")
    except Exception as e:
        print(f"  MISSING {pkg}: {e}"); ok = False
sys.exit(0 if ok else 1)
PY
    if [ $? -eq 0 ]; then echo "✅ Using $ENV"; export VERDENT_ENV="$ENV"; break; fi
  fi
done
```

**Decision rule**:
1. If a probed env passes → use it as-is. **Do not `pip install` or `conda install` into it** (you risk breaking other users' work). If a non-critical package is missing, install **only into a user-local site** with `pip install --user --target /mnt/cpfs/<you>/site-pkgs <pkg>` and prepend to `PYTHONPATH`.
2. If no env passes → create a private one:
   ```bash
   $CONDA env create -f environment.yml -p /mnt/cpfs/<you>/envs/lora-obd
   source /mnt/cpfs/junlongke/miniconda3/bin/activate /mnt/cpfs/<you>/envs/lora-obd
   ```
   Use a **path-based** env (`-p`) under your user dir, not a `-n` name in the shared root, to avoid polluting the shared env list.
3. Record the chosen env path in `STATUS.md` (Stage 0) so subsequent stages reuse it.

If you need to create your own env, `environment.yml`:

```yaml
name: lora-obd
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - cudatoolkit=11.8
  - pip:
      - torch==2.3.1
      - transformers==4.44.2
      - datasets==2.21.0
      - peft==0.12.0
      - accelerate==0.33.0
      - bitsandbytes==0.43.3
      - scipy==1.13.1
      - numpy==1.26.4
      - pandas==2.2.2
      - matplotlib==3.9.2
      - seaborn==0.13.2
      - tqdm==4.66.5
      - wandb==0.17.7
      - scikit-learn==1.5.1
      - einops==0.8.0
      - pyyaml==6.0.2
```

**Hard rule on environment hygiene**: do **not** modify shared envs, do **not** run `pip install` without `--user --target` against an env you didn't create. If in doubt, build your own private env at `/mnt/cpfs/<you>/envs/lora-obd`.

### 1.3 Repo layout to create

```
lora_obd/
├── README.md
├── STATUS.md                # 1-paragraph updates per stage
├── environment.yml
├── configs/
│   ├── stage1_sst2.yaml
│   ├── stage1_mrpc.yaml
│   ├── stage1_rte.yaml
│   ├── stage2_relora_slm.yaml
│   └── stage3_llama8b_sft.yaml
├── src/
│   ├── __init__.py
│   ├── data.py              # dataset loading + 3-way split
│   ├── model.py             # LoRA wrapping, expose B/A explicitly
│   ├── saliency.py          # the 5 saliency formulas (§5.4)
│   ├── ablation.py          # oracle component-removal ablation (§5.5)
│   ├── relora.py            # standard + diagnostic-gated ReLoRA
│   ├── effective_rank.py    # effective rank + condition number
│   └── utils.py             # seeding, logging, plotting helpers
├── scripts/
│   ├── stage0_smoke.py      # vanilla LoRA training as sanity
│   ├── stage1_run.py        # full Stage 1 experiment
│   ├── stage1_plot.py       # generate the 4 plots
│   ├── stage2_run.py        # Stage 2 experiment
│   └── stage3_run.py        # Stage 3 experiment
├── results/
│   ├── stage0/
│   ├── stage1/
│   └── stage2/
└── plots/
    ├── stage1/
    └── stage2/
```

### 1.4 Reproducibility hygiene

- Set seeds at the top of every script: `torch.manual_seed(42); np.random.seed(42); random.seed(42)`.
- Use `torch.use_deterministic_algorithms(True)` only if it doesn't crash; otherwise log non-determinism.
- Log every config to `results/<stage>/<run_name>/config.yaml`.
- Log every metric to `results/<stage>/<run_name>/metrics.jsonl` (one line per checkpoint).
- Use `wandb` if `WANDB_API_KEY` is set, otherwise file-only.

---

## 2. Stage 0: Smoke test (1–2 h)

### 2.1 Goal

Verify environment works end-to-end before burning hours on the real experiment.

### 2.2 What to do

Train **vanilla LoRA** on RoBERTa-base + SST-2 for 3 epochs.

```python
# scripts/stage0_smoke.py — pseudocode

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2)
tok = AutoTokenizer.from_pretrained("roberta-base")

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0,
    target_modules=["query", "value"],
    bias="none", task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_cfg)

ds = load_dataset("glue", "sst2")
# tokenize to max_len=128, batch_size=32
# train 3 epochs, lr=2e-4, AdamW, linear schedule, warmup=100

# Acceptance: dev accuracy ≥ 92.0%
```

### 2.3 Acceptance criteria for Stage 0

- [ ] Training completes without OOM on the target GPU.
- [ ] SST-2 dev accuracy ≥ **92.0%**.
- [ ] `wandb` or local log shows steady loss decrease.
- [ ] Memory peak logged.

If the accuracy threshold isn't hit, **do not proceed**. Common causes:
- Wrong target_modules naming for RoBERTa (use `query`, `value` not `q_proj`, `v_proj`).
- LR too high/low (try 1e-4 to 5e-4).
- Batch size mismatch.

---

## 3. Stage 1: Kill-or-seal experiment 1 — Predictive validity of saliency

### 3.1 The hypothesis (formal)

For a LoRA rank-1 component $i$, define:

- $\Delta_i^{\text{test}}$ = change in held-out test loss when component $i$ is set to zero (oracle, requires a forward pass per component on the test set).
- $s_i^{\text{train}}$ = saliency of component $i$ computed using gradients on the training set.
- $s_i^{\text{val}}$ = saliency of component $i$ computed using gradients on a held-out diagnostic set (disjoint from both train and test).

**Hypothesis $H_1$**: $\rho_{\text{Spearman}}(s^{\text{val}}, \Delta^{\text{test}}) > \rho_{\text{Spearman}}(s^{\text{train}}, \Delta^{\text{test}})$ with effect size $\geq 0.10$, consistent across:
- 3 datasets (SST-2, MRPC, RTE),
- 5 training checkpoints (1k, 2k, 3k, 4k, 5k steps),
- 5 saliency variants (see §3.4).

**Null $H_0$**: train and val saliencies have indistinguishable predictive validity.

### 3.2 Datasets and 3-way split

Three GLUE tasks, increasing difficulty:

| Task | Why included | n_train (full) |
|---|---|---|
| SST-2 | Easy, fast, large; sanity-positive case | 67k |
| MRPC | Smaller, harder; expected to show overfit signal | 3.7k |
| RTE | Smallest, hardest; strongest expected effect | 2.5k |

**Split rule** (apply per task):

```python
# Use the original GLUE train as our pool; redefine splits.
# We need a true held-out test that NEVER touches saliency or training.

train_pool = glue[task]["train"]
val_official = glue[task]["validation"]

# The official val becomes our held-out test (sealed).
test_holdout = val_official

# Re-split train_pool into:
#   train_main (80%)  : used for LoRA optimization
#   diagnostic (20%)  : used for val-saliency only (NEVER seen by the optimizer)
rng = np.random.default_rng(seed=42)
idx = rng.permutation(len(train_pool))
n_diag = int(len(train_pool) * 0.2)
diagnostic_idx = idx[:n_diag]
train_main_idx = idx[n_diag:]
```

This is the **canonical 3-split** used throughout. Document it in `data.py` and never reuse `test_holdout` for any non-final-evaluation purpose.

### 3.3 Model + LoRA config (per-task)

Same model: `roberta-base`. Same LoRA: `r=8, target_modules=["query","value"]`. Same optimizer: AdamW, lr=2e-4, linear schedule with warmup=100 steps, weight_decay=0.0.

Per task batch size and total steps:

| Task | batch | total_steps | save checkpoints at |
|---|---|---|---|
| SST-2 | 32 | 5000 | 1000, 2000, 3000, 4000, 5000 |
| MRPC | 16 | 2000 | 400, 800, 1200, 1600, 2000 |
| RTE | 16 | 2000 | 400, 800, 1200, 1600, 2000 |

Save full state_dict at each checkpoint.

### 3.4 Saliency variants to compute

For each LoRA layer $\ell$ and each rank component $i \in \{1, ..., r\}$, compute **5 saliency variants**.

**Setup**: each `LoRA` layer has parameters $B \in \mathbb{R}^{d_{\text{out}} \times r}$ and $A \in \mathbb{R}^{r \times d_{\text{in}}}$. The merged update is $\Delta W = BA = \sum_{i=1}^r b_i a_i^\top$, where $b_i$ is the $i$-th column of $B$ and $a_i$ is the $i$-th row of $A$.

**Key identity** (proof in v2 doc; verify in `saliency.py` with a unit test):

$$
s_i^{\text{first-order}} \equiv -\langle G, b_i a_i^\top \rangle = -\langle \nabla_A L \,[i,:],\, A[i,:]\rangle = -\langle \nabla_B L \,[:,i],\, B[:,i]\rangle
$$

where $G = \partial L / \partial(\Delta W)$ is the gradient w.r.t. the merged update. Both equivalent forms exist; either is fine, but pick one and stick with it.

The 5 variants:

| Name | Formula | Source | Notes |
|---|---|---|---|
| `S1_magnitude` | $\|b_i\| \cdot \|a_i\|$ | LoRA-drop, magnitude pruning | Baseline; signal-free |
| `S2_first_order_train` | $\|\langle \nabla_A^{\text{train}} L_{[i,:]}, A_{[i,:]}\rangle\|$ | AdaLoRA-like | Sign-stripped first-order |
| `S3_first_order_val` | $\|\langle \nabla_A^{\text{val}} L_{[i,:]}, A_{[i,:]}\rangle\|$ | **Our minimal val variant** | Same formula, val data |
| `S4_fisher_train` | $\mathbb{E}_{x\in\text{train}}\bigl[\langle\nabla_A L_x[i,:], A_{[i,:]}\rangle^2\bigr]$ | Sensitivity-LoRA (train Hessian diag approx) | Per-sample grads |
| `S5_fisher_val` | $\mathbb{E}_{x\in\text{val}}\bigl[\langle\nabla_A L_x[i,:], A_{[i,:]}\rangle^2\bigr]$ | **Our main variant** | Per-sample grads, val data |

**Important — keep the sign** when you also want the *signed* version (used to detect harmful updates). Compute both:
- `_abs` variant: absolute value (for rank-correlation with $\|\Delta^{\text{test}}\|$).
- `_signed` variant: signed (for binary classification of harmful vs helpful, see §3.6).

**Per-sample gradients** (needed for S4, S5): use `torch.func.vmap(grad(...))` if compatible, else loop with `torch.autograd.grad` over a small batch. For RoBERTa-base + r=8 + 12 layers × 2 (q,v), per-sample gradient memory is manageable on a single 24GB card up to batch 16; if OOM, lower to batch 4 and accumulate.

### 3.5 Oracle ablation (the ground truth $\Delta_i^{\text{test}}$)

For checkpoint at step $t$, for each layer $\ell$ and each component $i$:

```python
# Pseudocode
def oracle_ablation(model, layer_idx, comp_idx, test_loader):
    # Save originals
    B = model.lora_layers[layer_idx].B.detach().clone()
    A = model.lora_layers[layer_idx].A.detach().clone()

    # Zero out component i
    with torch.no_grad():
        model.lora_layers[layer_idx].B[:, comp_idx] = 0
        model.lora_layers[layer_idx].A[comp_idx, :] = 0

    # Eval on test_holdout
    loss_after = evaluate(model, test_loader)

    # Restore
    with torch.no_grad():
        model.lora_layers[layer_idx].B.copy_(B)
        model.lora_layers[layer_idx].A.copy_(A)
    return loss_after - loss_baseline
```

Run for **every component** in **every LoRA layer**. For RoBERTa-base + r=8 + 24 LoRA layers (q, v at each of 12 attention layers), that's 24 × 8 = 192 forward passes on `test_holdout`. At batch 64 and ~872 SST-2 dev examples, that's ~3 batches per pass, totaling ~600 batches per checkpoint. Should finish in under 10 minutes per checkpoint on an A100.

**Optimization**: batch the ablations. Forward pass is the same except for one zeroed component; can be done with hooks if you're careful, but the simple loop is fine for r=8.

### 3.6 Optional: signed saliency for harmful detection

A *harmful* component is one where $\Delta_i^{\text{test}} < 0$, i.e., removing it **improves** the test loss. The hypothesis is that val saliency, with sign preserved, can identify these.

Compute:

- `S3_signed_val` = $\langle \nabla_A^{\text{val}} L_{[i,:]}, A_{[i,:]}\rangle$ (signed first-order val).

Define binary labels $y_i = \mathbb{1}[\Delta_i^{\text{test}} < 0]$ (component is harmful → label 1).

Compute AUC of using $-s_i^{\text{val,signed}}$ as the harmful score (negative because negative saliency means the gradient points to *increase* loss, so removing it → improvement).

### 3.7 Outputs to compute and save (per task, per checkpoint)

For each checkpoint $t \in \{1k, 2k, 3k, 4k, 5k\}$ (or task-specific list) and each task:

```
results/stage1/<task>/<step>/
├── components.jsonl        # one line per (layer, comp) record
│   keys: layer_idx, comp_idx, S1_mag, S2_fo_tr, S3_fo_val,
│          S3_fo_val_signed, S4_fisher_tr, S5_fisher_val,
│          delta_test, harmful_flag
├── correlations.json       # Spearman rho per saliency vs delta_test
├── auc_signed.json         # AUC of S3_fo_val_signed vs harmful_flag
├── train_loss_curve.npy
├── test_loss_curve.npy
└── config.yaml
```

Aggregated across checkpoints/tasks:

```
results/stage1/summary/
├── correlation_matrix.csv       # rows = (task, step, saliency), cols = rho_spearman
├── correlation_aggregate.json   # mean/std per saliency variant across all (task,step)
└── decision.json                # see §3.7
```

### 3.8 Decision rule for Stage 1 → Stage 2 (binding)

Compute `delta_rho = rho(S5_fisher_val) - rho(S4_fisher_train)` and `delta_rho_fo = rho(S3_fo_val) - rho(S2_fo_train)` for each (task, step). Aggregate as mean across all (task, step) pairs (15 total).

**Go to Stage 2** if **all** of:
1. `mean(delta_rho) >= 0.10` AND `mean(delta_rho_fo) >= 0.05`,
2. The effect is positive on **at least 2 of 3 tasks** (paired sign test: positive on at least 10 of 15 (task,step) pairs).
3. AUC for `S3_fo_val_signed` ≥ **0.65** for harmful detection on at least one task at the latest checkpoint.

**Stop and write report** if:
- `mean(delta_rho) < 0` AND on more than 8 of 15 pairs val is *worse*,
- OR all AUCs are below 0.55 (val signal is no better than chance for harmful detection).

**Ambiguous (rare)**: write a `STATUS.md` entry, record the partial result, and ask user via `STATUS.md` for next direction. Do NOT silently proceed.

### 3.9 Stage 1 plots to generate

Save under `plots/stage1/`:

1. **`fig1_correlation_grid.png`**: 3×5 grid of scatter plots. Rows = tasks. Columns = checkpoints. Each subplot: x = $\Delta_i^{\text{test}}$, y = $s_i$ (color-coded by saliency variant). Annotate Spearman ρ.
2. **`fig2_rho_over_time.png`**: line plot. x = step. y = Spearman ρ. One line per saliency variant, faceted by task.
3. **`fig3_train_vs_val_paired.png`**: paired scatter. Each point is one (task, step). x = ρ(train saliency). y = ρ(val saliency). Diagonal y=x reference. Points above the diagonal support $H_1$.
4. **`fig4_harmful_auc.png`**: bar plot of AUC for harmful detection per task per saliency variant.

### 3.10 Stage 1 estimated GPU hours

Per-task budget (sequential within a single task):

| Item | A100-80G hours |
|---|---|
| 5 checkpoints × training | 2 |
| 5 checkpoints × oracle ablation (192 components × forward) | 1.5 |
| Saliency computation (5 variants) | 0.7 |
| Plotting + analysis | 0.1 |
| Buffer | 0.7 |
| **Per-task total** | **~5 hours** |

**Wall-clock with parallelism**:
- 3+ GPUs available → run SST-2 / MRPC / RTE in parallel → **~5–6 h wall-clock** total.
- 1 GPU only → sequential → ~16 h.

Launch in parallel via:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage1_run.py --config configs/stage1_sst2.yaml &
CUDA_VISIBLE_DEVICES=1 python scripts/stage1_run.py --config configs/stage1_mrpc.yaml &
CUDA_VISIBLE_DEVICES=2 python scripts/stage1_run.py --config configs/stage1_rte.yaml &
wait
python scripts/stage1_plot.py --aggregate
```

### 3.11 Stage 1 implementation skeleton (key parts)

Here's the most important function — get this **right** because every decision depends on it:

```python
# src/saliency.py

import torch
from torch.func import functional_call, grad, vmap

@torch.no_grad()
def get_lora_BA_handles(peft_model):
    """Return list of dicts: [{'name': str, 'B': nn.Parameter, 'A': nn.Parameter}].
    The shape convention: B is (out, r), A is (r, in)."""
    handles = []
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            # peft's default key is 'default'
            A = mod.lora_A["default"].weight   # shape (r, in)
            B = mod.lora_B["default"].weight   # shape (out, r)
            handles.append({"name": name, "B": B, "A": A})
    return handles

def compute_first_order_saliency(model, loader, device, signed=False):
    """Compute S2 / S3 (first-order, train or val depending on loader).
    Returns: dict[layer_name] -> tensor of shape (r,) per component."""
    model.eval()
    handles = get_lora_BA_handles(model)
    grad_acc = {h["name"]: {"A_grad": torch.zeros_like(h["A"]),
                             "n": 0} for h in handles}

    # Accumulate full-batch gradient
    total_loss = 0.0
    for batch in loader:
        model.zero_grad()
        out = model(**{k: v.to(device) for k, v in batch.items()})
        loss = out.loss
        loss.backward()
        for h in handles:
            grad_acc[h["name"]]["A_grad"] += h["A"].grad.detach().clone()
            grad_acc[h["name"]]["n"] += 1
        total_loss += loss.item()
    
    # Normalize and form per-component saliency
    saliency = {}
    for h in handles:
        n = grad_acc[h["name"]]["n"]
        avg_A_grad = grad_acc[h["name"]]["A_grad"] / n          # (r, in)
        A = h["A"].detach()                                     # (r, in)
        # Per-component dot product: row-wise inner product
        per_comp = (avg_A_grad * A).sum(dim=1)                  # (r,)
        if not signed:
            per_comp = per_comp.abs()
        saliency[h["name"]] = -per_comp                         # negative because removing comp i adds delta = -b_i a_i^T
        # Note: for ranking purposes the sign of the global negation doesn't change correlations
        # if we use abs; but keep it consistent.
    return saliency

def compute_fisher_saliency(model, loader, device):
    """Compute S4 / S5 (Fisher diagonal approx via per-sample squared gradients).
    Returns: dict[layer_name] -> tensor of shape (r,)."""
    model.eval()
    handles = get_lora_BA_handles(model)
    fisher = {h["name"]: torch.zeros(h["A"].shape[0], device=device)
              for h in handles}
    n_samples = 0
    
    for batch in loader:
        # Process one example at a time (or use vmap if memory allows)
        bsz = batch["input_ids"].size(0)
        for j in range(bsz):
            model.zero_grad()
            single = {k: v[j:j+1].to(device) for k, v in batch.items()}
            out = model(**single)
            out.loss.backward()
            for h in handles:
                A_grad = h["A"].grad.detach()                   # (r, in)
                A = h["A"].detach()                             # (r, in)
                per_comp = (A_grad * A).sum(dim=1)              # (r,)
                fisher[h["name"]] += per_comp ** 2              # accumulate squared
            n_samples += 1
    
    for k in fisher:
        fisher[k] /= n_samples
    return fisher

def saliency_dict_to_flat(sal_dict):
    """Convert {layer_name -> (r,)} to flat array with per-component ids."""
    rows = []
    for name, vec in sal_dict.items():
        for i, v in enumerate(vec.cpu().numpy()):
            rows.append({"layer": name, "comp": i, "value": float(v)})
    return rows
```

```python
# src/ablation.py

@torch.no_grad()
def oracle_ablation(model, loader, device, baseline_loss):
    """For every (layer, comp), zero it out and measure delta test loss.
    Returns: list of {'layer', 'comp', 'delta_test'}."""
    handles = get_lora_BA_handles(model)
    results = []
    for h in handles:
        r = h["A"].shape[0]
        for i in range(r):
            # Save
            B_col = h["B"][:, i].clone()
            A_row = h["A"][i, :].clone()
            # Zero
            h["B"][:, i] = 0
            h["A"][i, :] = 0
            # Eval
            loss = evaluate(model, loader, device)
            # Restore
            h["B"][:, i].copy_(B_col)
            h["A"][i, :].copy_(A_row)
            results.append({"layer": h["name"], "comp": i,
                            "delta_test": loss - baseline_loss})
    return results

def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            out = model(**{k: v.to(device) for k, v in batch.items()})
            total += out.loss.item() * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
    return total / n
```

### 3.12 Unit tests to add (do NOT skip)

In `tests/test_saliency.py`:

```python
def test_first_order_identity():
    """The two equivalent forms must agree numerically."""
    # Build a tiny linear model with explicit B, A
    # Compute s via <A.grad[i], A[i]> and via <B.grad[:, i], B[:, i]>
    # Assert they differ by < 1e-5

def test_zeroing_component_matches_saliency_sign():
    """Sign of first-order saliency should agree with the direction of test loss change
    in the limit of small components."""
    # Train a tiny LoRA. For random components, scale them by epsilon.
    # Compute first-order saliency. Verify that the sign predicts the sign of
    # the test loss change for the scaled (small) component.
```

These tests catch silent bugs in gradient sign conventions that would invalidate the entire experiment.

---

## 4. Stage 2: Reproduce ReLoRA failure mode + diagnostic fix

**Only enter Stage 2 if Stage 1 decision is GO.**

### 4.1 Goal

Reproduce the central finding of Weiss et al. 2025 (arXiv:2509.12960) on small LMs: ReLoRA degrades effective rank and induces ill-conditioning. Then show that diagnostic-gated ReLoRA fixes it.

### 4.2 Setup (mirror Weiss 2025 closely)

- **Models**: 11M, 33M, 66M parameter LLaMA-style decoder-only LMs (architecture in their §3). Use `torchtitan` or a minimal in-house implementation.
- **Data**: SlimPajama-6B subset (sample 1B tokens). Or: C4 1B subset.
- **Tokenizer**: LLaMA-2 tokenizer.
- **Sequence length**: 1024.
- **Batch size**: tune to fit GPU; aim for global batch ≥ 256.
- **Total training tokens**: 5B (matches Weiss).
- **LR**: 3e-4 cosine decay, warmup=500.
- **ReLoRA**: r=64, merge every 5000 steps, optimizer reset per merge (Lialin 2023's protocol).

### 4.3 Three runs (per model size)

| Run | Description |
|---|---|
| `full_rank` | Standard pretraining (upper bound) |
| `relora_baseline` | Vanilla ReLoRA (Lialin 2023) |
| `relora_diag_gated` | **Our method**: per-component val saliency at each merge; only merge components where `S5_fisher_val > 0` (i.e., merging would reduce val loss). Components below threshold are dropped (rank slot reset). |

For 11M model only, also run:
| Run | Description |
|---|---|
| `relora_signed` | ReLoRA but actively reverts components with negative `S3_fo_val_signed` |

### 4.4 What to track per training step

```
results/stage2/<model_size>/<run_name>/
├── train_loss.jsonl
├── val_loss_paloma.jsonl    # Paloma perplexity (Weiss's main metric)
├── effective_rank.jsonl     # per-layer per-step effective rank of (W_0 + stable)
├── condition_number.jsonl   # per-layer per-step condition number
└── saliency_at_merge.jsonl  # at each merge: distribution of per-component saliency
```

**Effective rank** definition (Roy & Vetterli 2007, also used by Weiss):

```python
def effective_rank(M, eps=1e-10):
    s = torch.linalg.svdvals(M)
    p = s / (s.sum() + eps)
    H = -(p * (p + eps).log()).sum()
    return torch.exp(H).item()
```

**Condition number**: ratio of max to min nonzero singular value (clip min at eps).

Compute these on `(W_0 + ΔW_stable)` (the merged total) every 500 steps.

### 4.5 Decision rule for Stage 2 → Stage 3

**Method works** if **all** of:
1. `relora_baseline` reproduces Weiss's failure: effective rank curve trends *downward* over merges (or at least is non-monotone) on at least the 11M model, AND val loss is worse than `full_rank`.
2. `relora_diag_gated` shows effective rank trending *upward* (or stable + close to `full_rank`), AND val loss is closer to `full_rank` than `relora_baseline` is.
3. The improvement gap (val loss `relora_diag_gated` vs `relora_baseline`) is ≥ 5% relative on at least one of the three model sizes.

**Method does not work** if:
- `relora_diag_gated` is statistically indistinguishable from `relora_baseline` on all three sizes,
- OR effective rank curves look identical.

### 4.6 Stage 2 estimated GPU hours

| Run | A100-80G hours |
|---|---|
| 11M × 3 runs (full_rank, relora_baseline, relora_diag_gated) | 36 |
| 33M × 3 runs | 72 |
| 66M × 3 runs | 108 |
| Buffer | 30 |
| **Total GPU-hours** | **~250** |

**Wall-clock with parallelism** (the **default**):
- **9 GPUs, max parallel**: assign each (size × method) pair to one GPU → wall-clock = max single-run = **~36 h (1.5 days)**.
- **3 GPUs, per-size parallel**: each size runs its 3 methods sequentially → wall-clock = ~36 h (gated by 66M).
- **3 GPUs, per-method parallel**: full_rank/baseline/diag_gated each runs 11M→33M→66M sequentially → wall-clock = ~72 h.
- **1 GPU**: ~10 days. Avoid this if possible.

**Default launch order** (assumes ≥ 3 GPUs):

```bash
# Phase A: 11M (fast, validates the pipeline). Use 3 GPUs in parallel.
CUDA_VISIBLE_DEVICES=0 python scripts/stage2_run.py --size 11M --method full_rank &
CUDA_VISIBLE_DEVICES=1 python scripts/stage2_run.py --size 11M --method relora_baseline &
CUDA_VISIBLE_DEVICES=2 python scripts/stage2_run.py --size 11M --method relora_diag_gated &
wait
# Inspect: did relora_baseline reproduce the Weiss failure? If not, see §6.4.

# Phase B: 33M + 66M in parallel if GPU count allows.
# With ≥ 6 GPUs free, run all 6 (size × method) jobs concurrently.
for SIZE in 33M 66M; do
  for METHOD in full_rank relora_baseline relora_diag_gated; do
    GPU=$(next_free_gpu)  # implement a simple scheduler
    CUDA_VISIBLE_DEVICES=$GPU python scripts/stage2_run.py --size $SIZE --method $METHOD &
  done
done
wait
```

If GPU budget is tight, start with **11M only** and use that to make the go/no-go call.

### 4.7 Stage 2 plots

1. **`fig5_effective_rank_curves.png`**: x = training tokens, y = effective rank. Lines = {full_rank, relora_baseline, relora_diag_gated}. Shaded across layers. **The headline figure if it works.**
2. **`fig6_condition_number_curves.png`**: same axes, y = log10(condition number).
3. **`fig7_paloma_perplexity.png`**: x = training tokens, y = Paloma PPL.
4. **`fig8_saliency_dist_at_merges.png`**: violin plots of per-component saliency at each merge event, showing how many components fall below threshold.

---

## 5. Stage 3 (optional): Scale to Llama-3.1-8B SFT

**Only enter Stage 3 if Stage 2 decision is GO.**

### 5.1 Goal

Demonstrate the method's value on a SOTA-relevant SFT setting and beat the strongest baselines (Sensitivity-LoRA, CTR-LoRA, AdaLoRA).

### 5.2 Setup

- Base model: `meta-llama/Llama-3.1-8B`.
- SFT data: a moderate-size dataset where overfitting is non-trivial:
  - Primary: Tulu-3 SFT mixture (140k examples), or
  - Alternative: Alpaca-cleaned + GSM8K-train mixture.
- Eval: GSM8K, MMLU, BBH, IFEval (instruction following).
- LoRA: r=16, target_modules = all linear in attention + MLP.
- 7 baselines from v2 doc §4: LoRA, DoRA, AdaLoRA, Sensitivity-LoRA, CTR-LoRA, ReLoRA, COLA.
- Our method: diagnostic-gated rank recycling per v1 §1.

### 5.3 Decision rule

**Method scales** if our method beats Sensitivity-LoRA and CTR-LoRA by ≥ 1.0 average points across the 4 evals at the same total adapter-parameter budget.

### 5.4 Stage 3 estimated GPU hours

8 methods × ~24 hours each = ~192 GPU-hours per single-GPU run. With Llama-3.1-8B + LoRA on A100-80G, FSDP/DDP is **not required** (LoRA training fits in one card with batch ~8 and bf16). Run **each method on its own GPU in parallel**:

- **8 GPUs**: all 8 methods concurrently → wall-clock ~24 h (1 day).
- **4 GPUs**: 2 waves of 4 methods → wall-clock ~48 h.

```bash
METHODS=(lora dora adalora sensitivity_lora ctr_lora relora cola ours)
for i in "${!METHODS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python scripts/stage3_run.py --method ${METHODS[$i]} &
done
wait
python scripts/stage3_eval.py --aggregate
```

---

## 6. Common failure modes and what to do

### 6.1 OOM during per-sample gradients (Stage 1, S4/S5)

- Reduce per-sample batch to 1, accumulate.
- Use gradient checkpointing (`model.gradient_checkpointing_enable()`).
- If still OOM: skip S4 (train Fisher) and only compute S5 (val Fisher) on a subsample. Note this in the report.

### 6.2 Spearman ρ unstable (Stage 1)

- This is expected if r is small (only 8 × 24 = 192 components per checkpoint). To stabilize:
  - Bootstrap: 1000 resamples of components, report mean ρ ± 95% CI.
  - Pool across layers: report ρ at the per-layer-aggregate level if per-component is too noisy.

### 6.3 Oracle ablation is too slow

- Subsample test set to 500 examples — variance grows but feasibility wins.
- Or: do oracle only at the **last checkpoint** for each task. This still gives 3 (task) × 192 (component) data points.

### 6.4 ReLoRA reproduction fails (Stage 2.5.1 doesn't trigger)

If `relora_baseline` doesn't reproduce the Weiss failure, the comparison loses its punch. In that case:
- Increase merge frequency (every 2000 steps instead of 5000).
- Try smaller model (7M params if you can train one).
- If still no failure: pivot to "our method is at least as good as ReLoRA on standard pretraining, and additionally provides a diagnostic signal" — weaker but defensible.

### 6.5 Disagreement with v1's named saliency variants

v1 §4.2 uses slightly different naming. Treat v1 as **method-design source of truth**, this doc as **execution source of truth**. If a name conflicts, use this doc's name and add a mapping note at the top of `STATUS.md`.

### 6.6 You think you found a bug in the math

Don't silently fix. Add a comment in `src/saliency.py`, write a unit test that demonstrates the bug, and note it in `STATUS.md`. Continue with what's documented; flag for review.

---

## 7. What to commit (per-stage deliverables)

### After Stage 0:
- [ ] `STATUS.md` with env hash, GPU info, smoke test accuracy.
- [ ] `results/stage0/smoke.json` with metrics.
- [ ] git commit `stage 0 smoke test pass`

### After Stage 1:
- [ ] `STATUS.md` updated.
- [ ] `results/stage1/**` (all per-task / per-checkpoint records).
- [ ] `plots/stage1/fig1` ... `fig4`.
- [ ] `results/stage1/decision.json` with `{"go": bool, "delta_rho_fo": float, "delta_rho_fisher": float, "auc_per_task": {...}, "rationale": str}`.
- [ ] One-page summary in `results/stage1/report.md` (figures embedded).
- [ ] git commit + tag `stage1-decision-{go|stop}`.

### After Stage 2:
- [ ] `STATUS.md` updated.
- [ ] All training logs.
- [ ] `plots/stage2/fig5` ... `fig8`.
- [ ] `results/stage2/decision.json`.
- [ ] `results/stage2/report.md`.
- [ ] git commit + tag.

### After Stage 3:
- [ ] Final paper-quality table and plots.
- [ ] Reproducibility script `reproduce_all.sh`.
- [ ] git tag `final`.

---

## 8. Communication protocol with the human

You are autonomous, but write to `STATUS.md` at these moments:

| Trigger | Section to add |
|---|---|
| Stage 0 done | env summary, smoke result |
| Stage 1 first task done | first ρ table, first impressions |
| Stage 1 fully done | decision and rationale |
| Stage 2 reproduction confirmed | "Weiss failure reproduced" + plot link |
| Stage 2 fully done | decision and rationale |
| Any decision rule triggers `STOP` | reason, partial results, candidate next steps |
| Any unexpected failure mode that v6 didn't anticipate | freeze, write up, request human input |

Use plain markdown, dated, append-only.

---

## 9. Hard constraints

1. **Never** train on the diagnostic set or the test_holdout. Triple-check splits in `data.py`.
2. **Never** auto-skip a stage.
3. **Never** merge stable updates by default — keep them as a separate `Δ_stable` adapter at least until a final phase. v1 §4.3 explains why.
4. **Always** log random seeds and pin them.
5. **Always** save per-checkpoint state_dicts so any analysis can be re-run without retraining.
6. **Always** report effect sizes with bootstrapped CIs, not just means.
7. **Do not** silently change the saliency formulas. If you propose a modification, document it in `STATUS.md` first.
8. **Do not** use the official GLUE val set for any saliency or pruning decision; it is `test_holdout` for our purposes.
9. **Do not** add EPI (arXiv:2604.14010) as a baseline. It is **concurrent work** (April 2026, ≤ 1 month before us) and operates on a different problem (base-parameter freezing for SFT). See v2 §1.7. You may borrow its design heuristics (e.g., "low score for $k$ consecutive checks before release", layer-wise normalization) — these are noted in v2 §1.7 as adoptable practices.
10. **Do not** mutate shared conda envs. If a pre-existing env is compatible (§1.2 probe), use it read-only — no `pip install`, no `conda install`. Missing packages go into a user-local `--target` site or a fresh env at `/mnt/cpfs/<you>/envs/`. Always log the chosen env path in `STATUS.md`.
11. **Always** redirect parallel-job stdout/stderr to per-run files under `logs/`. Never let two background jobs share a stream.
12. **Always** check `nvidia-smi` before launching parallel jobs; respect cards already in use by other users (don't OOM them).

---

## 10. Quick-start checklist

**Default assumption: ≥ 3 free A100-80G cards. Run experiments in parallel.**

```bash
# === Stage 0: env probe + smoke ===
CONDA=/mnt/cpfs/junlongke/miniconda3/bin/conda
$CONDA env list                                   # 1. see what exists
# 2. probe; reuse if a compatible env is found (see §1.2). Only create if needed:
#    $CONDA env create -f environment.yml -p /mnt/cpfs/<you>/envs/lora-obd
source /mnt/cpfs/junlongke/miniconda3/bin/activate <env-path-or-name>
nvidia-smi --query-gpu=index,memory.free --format=csv
CUDA_VISIBLE_DEVICES=0 python scripts/stage0_smoke.py --config configs/stage1_sst2.yaml --smoke
# ✅ accuracy ≥ 92.0% → continue

# === Stage 1: 3 tasks IN PARALLEL on 3 GPUs ===
CUDA_VISIBLE_DEVICES=0 python scripts/stage1_run.py --config configs/stage1_sst2.yaml > logs/s1_sst2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/stage1_run.py --config configs/stage1_mrpc.yaml > logs/s1_mrpc.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/stage1_run.py --config configs/stage1_rte.yaml  > logs/s1_rte.log  2>&1 &
wait
python scripts/stage1_plot.py --aggregate
# Read results/stage1/decision.json → go/stop

# === Stage 2: (only if Stage 1 says go) max parallelism ===
# Phase A: 11M × 3 methods on 3 GPUs (use this to verify the Weiss reproduction).
CUDA_VISIBLE_DEVICES=0 python scripts/stage2_run.py --size 11M --method full_rank          > logs/s2_11M_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/stage2_run.py --size 11M --method relora_baseline    > logs/s2_11M_relo.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/stage2_run.py --size 11M --method relora_diag_gated  > logs/s2_11M_diag.log 2>&1 &
wait
# Phase B: 33M + 66M, fan out across remaining GPUs (up to 6 jobs in parallel).
bash scripts/stage2_fanout.sh   # implement to dispatch (size×method) pairs to free GPUs
python scripts/stage2_plot.py
# Read results/stage2/decision.json → go/stop

# === Stage 3: 8 methods IN PARALLEL across 8 GPUs ===
METHODS=(lora dora adalora sensitivity_lora ctr_lora relora cola ours)
for i in "${!METHODS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python scripts/stage3_run.py --method ${METHODS[$i]} > logs/s3_${METHODS[$i]}.log 2>&1 &
done
wait
python scripts/stage3_eval.py --aggregate
```

---

## 11. Appendix: parallelism / scaling guide

Default plan assumes **multi-A100-80G availability** and aggressive parallelism.

| GPUs available | Stage 1 wall-clock | Stage 2 wall-clock (11M+33M+66M, 3 methods each) | Stage 3 wall-clock (8 methods) |
|---|---|---|---|
| 1 | ~16 h | ~10 d | ~8 d |
| 3 | **~5 h** | **~3 d** (size-parallel) | ~3 d (3 waves) |
| 6 | ~5 h | **~36 h** | ~36 h (2 waves) |
| 8 | ~5 h | ~36 h | **~24 h (all parallel)** |
| 9 | ~5 h | **~36 h (full fan-out)** | ~24 h |

Rules:
1. Always check `nvidia-smi` first; don't assume cards are free.
2. Each Stage-1 run fits comfortably in <30 GB → never reserve a whole card for it; co-location with another small job is fine if memory budget allows.
3. Stage-2 11M / 33M / 66M each fit in <60 GB → one card per run is safe.
4. Stage-3 Llama-3.1-8B + LoRA(r=16) in bf16 fits in ~50 GB on A100-80G with batch=8.
5. **Always redirect stdout/stderr to per-run log files** (`logs/`) so parallel runs don't interleave.

---

## 12. Final note from the human

The single most important thing this experiment can produce is **`fig3_train_vs_val_paired.png`** and the associated decision in `decision.json`. Everything else is supporting material. If a corner has to be cut, cut Stage 2 model sizes and Stage 3 baselines, never cut the predictive-validity analysis in Stage 1.

— end of handover —
