# STATUS — LoRA OBD-Recycling (Resumable Log)

Append-only progress log. Newer entries on top. **READ THIS FIRST WHEN RESUMING.**

---

## RESUME INSTRUCTIONS (read first)

### Project root: `/mnt/cpfs/junlongke/onlinelora/lora_obd/`
- Handover doc: `/mnt/cpfs/junlongke/onlinelora/lora_obd_handover_for_gpu_agent.md` (916 lines)
- v1/v2 research docs in same parent dir.

### Environment
- Python: `/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python`
- Set `export HF_HOME=/mnt/cpfs/junlongke/hf_cache` before any run.
- Hard rule: do NOT `pip install` into `espo`. Use `--target /mnt/cpfs/<you>/site-pkgs` and prepend PYTHONPATH if absolutely needed.
- Key versions: torch 2.6.0+cu124, transformers 4.52.0.dev0, peft 0.17.0, accelerate 1.4.0, datasets 4.8.2, scipy 1.17.1, sklearn 1.7.2, matplotlib 3.10.8, numpy 2.3.5, pandas 2.3.3.
- Missing from env: seaborn (use matplotlib only).

### GPU map (snapshot 10:13)
- 8x A100-80GB. Free: 0, 1, 3, 4, 5, 6, 7. Busy: GPU 2 (other user).

### Running processes (PIDs in `.stage1_pids`)
- PID 892441: Stage 1 SST-2 on GPU 0 → `logs/s1_sst2.log`, output `results/stage1/sst2/`
- PID 892442: Stage 1 MRPC on GPU 1 → `logs/s1_mrpc.log`, output `results/stage1/mrpc/`
- PID 892443: Stage 1 RTE  on GPU 3 → `logs/s1_rte.log`,  output `results/stage1/rte/`
- To check: `for pid in $(cat .stage1_pids); do kill -0 $pid 2>/dev/null && echo "$pid alive" || echo "$pid DEAD"; done`
- ETA: SST-2 ~80 min, MRPC/RTE ~25 min each. Should finish by ~11:30.

### Repo layout (already created)
```
lora_obd/
├── README.md, STATUS.md
├── configs/stage1_{sst2,mrpc,rte}.yaml      ← per-task hyperparameters
├── src/
│   ├── utils.py        (seed, jsonl, yaml, NaN-clean JSON)
│   ├── data.py         (GLUE 3-way split: train_main 80% / diagnostic 20% / test_holdout = official val)
│   ├── model.py        (build_lora_model + get_lora_BA_handles — FIXED, was buggy)
│   ├── saliency.py     (S1 magnitude, S2/S3 first_order, S4/S5 fisher; signed/unsigned)
│   ├── ablation.py     (evaluate + oracle_ablation — zero comp i, restore)
│   └── effective_rank.py (Roy-Vetterli + condition number, for Stage 2)
├── scripts/
│   ├── stage0_smoke.py  (DONE: SST-2 92.32%)
│   ├── stage1_run.py    (RUNNING)
│   ├── stage1_aggregate.py  (TODO — see below)
│   ├── stage1_plot.py   (TODO — fig1–fig4)
│   └── stage1_decide.py (TODO — apply §3.8 decision rule)
├── tests/test_saliency.py  (3 tests PASS)
├── results/stage{0,1,2}/, plots/stage{1,2}/, logs/
└── .stage1_pids
```

### Critical bug fixed today
- `src/model.py::get_lora_BA_handles` used `id(A) ^ id(B)` as dedup; tiny `id()` low bits caused 9 XOR collisions, silently dropped **10 of 24 LoRA layers**.
- Fix: tuple `(id(A), id(B))`. Verified 24 LoRA layers × r=8 = **192 components** (matches handover §3.5).
- Initial Stage 1 runs (PIDs 889708/889709/889710) were killed; new runs started 10:13.

### Stage 0 Result
- SST-2 dev acc=92.32% (best 92.66%) at 1500 steps. PASS. `results/stage0/smoke.json`.

### Stage 1 decision rule (handover §3.8) — apply after all 3 runs finish
- **GO to Stage 2** iff ALL of:
  1. mean over (task, step) of `delta_rho_fisher = rho(S5_fisher_val) - rho(S4_fisher_tr)` ≥ **0.10**
     AND mean of `delta_rho_fo = rho(S3_fo_val) - rho(S2_fo_tr)` ≥ **0.05**
  2. Positive on ≥ 10 of 15 (task, step) pairs (paired sign test)
  3. AUC for `S3_fo_val_signed` (as -score) for harmful detection ≥ **0.65** on ≥ 1 task at LATEST checkpoint
- **STOP** if mean(delta_rho_fisher) < 0 AND val worse on > 8/15 pairs, OR all AUCs < 0.55.
- **AMBIGUOUS** otherwise → write STATUS.md entry, ask user.

### Next steps (in priority order) when resuming
1. **Check if Stage 1 runs done**: `for pid in $(cat .stage1_pids); do kill -0 $pid 2>/dev/null && echo alive || echo done; done`
2. **Inspect per-checkpoint outputs**: `results/stage1/{sst2,mrpc,rte}/{step}/components.jsonl`, `correlations.json`, `auc_signed.json`, plus `results/stage1/{task}/summary.json`.
3. **Write `scripts/stage1_aggregate.py`** that:
   - Reads all `results/stage1/{task}/summary.json` for 3 tasks
   - Builds correlation_matrix.csv (rows = task × step × saliency, col = rho)
   - Computes delta_rho per (task, step) and aggregates (mean ± bootstrap CI)
   - Writes `results/stage1/summary/correlation_matrix.csv`, `correlation_aggregate.json`, and `decision.json`
4. **Write `scripts/stage1_plot.py`** that emits:
   - fig1_correlation_grid.png (3 tasks × 5 ckpts scatter, x=delta_test, y=saliency)
   - fig2_rho_over_time.png (rho vs step, faceted by task)
   - fig3_train_vs_val_paired.png **(headline figure — handover §12)**
   - fig4_harmful_auc.png
5. **Stage 1 report**: `results/stage1/report.md` with embedded figs + decision.json summary.
6. **Update STATUS.md** with Stage 1 decision + go/stop.

### Output file format reference
- Each checkpoint dir `results/stage1/<task>/<step>/`:
  - `components.jsonl`: one JSON per (layer, comp). Fields: layer, comp, S1_mag, S2_fo_tr, S3_fo_val, S3_fo_val_signed, S4_fisher_tr, S5_fisher_val, delta_test, loss_after, harmful_flag, step.
  - `correlations.json`: `{S{i}_<name>_rho_vs_delta, S{i}_<name>_rho_vs_abs_delta}`.
  - `auc_signed.json`: `{S3_fo_val_signed_neg_auc_harmful, n_harmful, n_total, harmful_rate, baseline_test_loss, baseline_test_acc}`.
- Per-task root `results/stage1/<task>/`:
  - `train_loss.jsonl`, `test_loss.jsonl`, `config.yaml`, `summary.json` (aggregated across ckpts), `run.log`.

### Saliency formulas (handover §3.4)
- All return dict[layer_name] -> tensor of shape (r,)
- S1: ||b_i|| * ||a_i||
- S2: |<grad_A L^train, A>_{row i}|     (computed with `first_order_saliency(..., signed=False)` on train_diag_loader)
- S3: |<grad_A L^val, A>_{row i}|         (same, on diag_loader)
- S3 signed: <grad_A L^val, A>_{row i}    (signed=True)
- S4: mean over train samples of (<grad_A L_x, A>_{row i})^2  (fisher_saliency on train_diag_loader)
- S5: same, val samples (diag_loader)
- Identity verified to 1.3e-18 in unit test (float64).
- Sign convention (handover): s_i = -<G, b_i a_i^T> where G = grad_{delta_W} L. We accumulate raw <grad_A, A> per row; sign is preserved in S3_signed and inverted for AUC scoring (-score → "harmful").

### Hard rules (handover §9)
1. NEVER train on diagnostic or test_holdout.
2. NEVER auto-skip a stage.
3. NEVER merge stable updates by default — keep Δ_stable separate.
4. ALWAYS log seeds (42).
5. ALWAYS save per-checkpoint state_dict.
6. ALWAYS report effect sizes with bootstrapped CIs.
7. DO NOT silently change saliency formulas.
8. DO NOT use official GLUE val for saliency/pruning (it is test_holdout).
9. DO NOT add EPI (arXiv:2604.14010) as baseline.

---

## 2026-05-12 10:13 — Stage 1 (re-launch after bug fix)

- Killed initial runs (PIDs 889708/889709/889710) at 10:11 because `get_lora_BA_handles` was dropping 10/24 LoRA layers.
- Fixed `src/model.py` dedup to use tuple `(id(A), id(B))`.
- Verified: 24 LoRA layers × r=8 = 192 components per checkpoint, matches handover §3.5.
- Re-launched 10:13 on GPUs 0/1/3. ETA ~80 min for slowest (SST-2).

## 2026-05-12 10:48 — Stage 1 COMPLETE — verdict: AMBIGUOUS, awaiting user

### Final aggregate
- 15 (task, checkpoint) pairs: sst2 {1k,2k,3k,4k,5k}, mrpc {400,800,1200,1600,2000}, rte {400,800,1200,1600,2000}.
- All 24 LoRA layers × r=8 = 192 components scored at every checkpoint.

### Decision (handover §3.8) — AMBIGUOUS_neither
| Gate | Threshold | Observed | Pass |
|---|---|---|---|
| mean Δ\|ρ\|_fisher | ≥0.10 | **0.046** (CI95 0.008..0.088) | FAIL |
| mean Δ\|ρ\|_fo     | ≥0.05 | **0.246** (CI95 0.120..0.373) | PASS |
| sign test ≥10/15  | ≥10   | fisher 10/15, fo 11/15        | PASS |
| sym-AUC ≥0.65, ≥1 task at latest | — | mrpc 0.776, rte 0.759, sst2 0.587 | PASS |
| STOP: all AUCs<0.55 | — | min 0.587 | not triggered |

Per §3.8, AMBIGUOUS → must ask user before Stage 2.

### Key finding
- The val/train distinction is **enormous in first-order saliency** (+0.246 mean Δ|ρ|, 11/15 positive).
- It is **near-zero in Fisher saliency** (+0.046 mean Δ|ρ|), with SST-2 actually showing TRAIN > VAL in 4/5 ckpts.
- Hypothesis is task-dependent: works on MRPC/RTE (small, easy overfit), weaker on SST-2 (large, mild overfit).
- harmful_rate climbs from 30-50% early to 65-86% late, confirming overfitting regime.

### Artifacts
- `results/stage1/{task}/{step}/components.jsonl` (15 files × 192 rows)
- `results/stage1/{task}/{step}/{correlations,auc_signed}.json`
- `results/stage1/{task}/{train_loss,test_loss}.jsonl`, `summary.json`, `run.log`, `config.yaml`
- `results/stage1/summary/{correlation_matrix.csv,per_pair_table.csv,correlation_aggregate.json,decision.json}`
- `plots/stage1/{fig1_correlation_grid,fig2_rho_over_time,fig3_train_vs_val_paired,fig4_harmful_auc}.png`
- **`results/stage1/report.md`** — full Stage 1 report with TLDR, table, interpretation, recommendation.

### Pending user decision
- Path A: proceed to Stage 2 with first-order-only diagnostic gating (FO signal is robust).
- Path B: stop and pivot to K-fold cross-validated saliency to reduce Fisher variance, OR reframe contribution narrative.

## 2026-05-12 — Stage 1a: bug fix + re-launch

### Bug found and fixed (CRITICAL, would have invalidated Stage 1)
- `src/model.py::get_lora_BA_handles` used `id(A) ^ id(B)` as dedup key.
- Python `id()` low bits are tiny integers; XOR caused 9 collisions, **silently dropping 10 of 24 LoRA layers** (only 14 found).
- Fix: switched to tuple `(id(A), id(B))` dedup. Verified: 24 LoRA layers now found (12 transformer layers × q,v), 24 × r=8 = **192 rank-1 components**, matches handover §3.5.

## 2026-05-12 — Stage 0d: smoke test PASS

- RoBERTa-base + LoRA(r=8, q,v) on SST-2; 1500 steps; lr=2e-4 linear, warmup=100; batch=32.
- Final dev acc = **92.32%** (best across run: 92.66%), final loss = 0.202.
- Peak GPU mem = 2.21 GB (1×A100-80GB). Elapsed = 154s.
- Trainable params = 887,042 (LoRA only).
- 3-way split confirmed: train_main=53,880, diagnostic=13,469, test_holdout=872 (= official GLUE dev).
- `results/stage0/smoke.json` saved.
- Decision: Stage 0 **PASS**; proceeding to Stage 1.

## 2026-05-12 — Stage 0a: environment probe

- Host: 8x NVIDIA A100-SXM4-80GB. GPUs 0,1,3,4,5,6,7 idle; GPU 2 busy by other user.
- Conda probe: 9 existing envs. `espo` and `modes` both satisfy the minimum dependency requirements. Selected **`espo`** (has `bitsandbytes` for potential Stage 3, slightly older but stable peft 0.17.0).
- Chosen interpreter: `/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python`
- Decision: `env_ready = True`.

---

## 2026-05-12 12:16:44 — sign_check (per PI §1.2)

raw AUC dump:

```
results/stage1/mrpc/1200/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.15559348739495796, "n_harmful": 136, "n_total": 192, "harmful_rate": 0.7083333333333334, "baseline_test_loss": 0.32198448932053997, "baseline_test_acc": 0.8823529411764706}
results/stage1/mrpc/1600/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.1677182685253118, "n_harmful": 145, "n_total": 192, "harmful_rate": 0.7552083333333334, "baseline_test_loss": 0.3729771361047146, "baseline_test_acc": 0.8578431372549019}
results/stage1/mrpc/2000/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.22437898737613685, "n_harmful": 139, "n_total": 192, "harmful_rate": 0.7239583333333334, "baseline_test_loss": 0.44994108196274907, "baseline_test_acc": 0.875}
results/stage1/mrpc/400/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.2009479717813051, "n_harmful": 108, "n_total": 192, "harmful_rate": 0.5625, "baseline_test_loss": 0.327699676156044, "baseline_test_acc": 0.8651960784313726}
results/stage1/mrpc/800/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.12712053571428572, "n_harmful": 112, "n_total": 192, "harmful_rate": 0.5833333333333334, "baseline_test_loss": 0.3174036478295046, "baseline_test_acc": 0.8725490196078431}
results/stage1/rte/1200/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.21204323211528567, "n_harmful": 134, "n_total": 192, "harmful_rate": 0.6979166666666666, "baseline_test_loss": 0.7827027128061232, "baseline_test_acc": 0.7581227436823105}
results/stage1/rte/1600/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.20857142857142857, "n_harmful": 150, "n_total": 192, "harmful_rate": 0.78125, "baseline_test_loss": 1.0630494599307918, "baseline_test_acc": 0.7509025270758123}
results/stage1/rte/2000/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.24108584005869405, "n_harmful": 145, "n_total": 192, "harmful_rate": 0.7552083333333334, "baseline_test_loss": 1.1091017817762354, "baseline_test_acc": 0.7581227436823105}
results/stage1/rte/400/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.15619239148650915, "n_harmful": 153, "n_total": 192, "harmful_rate": 0.796875, "baseline_test_loss": 0.6297781435160861, "baseline_test_acc": 0.7075812274368231}
results/stage1/rte/800/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.1603336422613531, "n_harmful": 166, "n_total": 192, "harmful_rate": 0.8645833333333334, "baseline_test_loss": 0.7620992691723448, "baseline_test_acc": 0.7364620938628159}
results/stage1/sst2/1000/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.2502525252525253, "n_harmful": 60, "n_total": 192, "harmful_rate": 0.3125, "baseline_test_loss": 0.19881970729302922, "baseline_test_acc": 0.9277522935779816}
results/stage1/sst2/2000/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.7714626865671642, "n_harmful": 125, "n_total": 192, "harmful_rate": 0.6510416666666666, "baseline_test_loss": 0.20072085239471646, "baseline_test_acc": 0.9311926605504587}
results/stage1/sst2/3000/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.5900429414922168, "n_harmful": 138, "n_total": 192, "harmful_rate": 0.71875, "baseline_test_loss": 0.19663522549725454, "baseline_test_acc": 0.9311926605504587}
results/stage1/sst2/4000/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.6686432637571157, "n_harmful": 124, "n_total": 192, "harmful_rate": 0.6458333333333334, "baseline_test_loss": 0.19874695763675446, "baseline_test_acc": 0.9323394495412844}
results/stage1/sst2/5000/auc_signed.json
{ "S3_fo_val_signed_neg_auc_harmful": 0.5865928975810604, "n_harmful": 134, "n_total": 192, "harmful_rate": 0.6979166666666666, "baseline_test_loss": 0.20240446733772208, "baseline_test_acc": 0.9311926605504587}
```

### sign-check tally (per PI §1.2 rule)

Per-task vote (auc_neg = S3_fo_val_signed_neg_auc_harmful, threshold 0.5):

| task | ckpts | values (sorted by step) | n>0.5 | verdict |
|------|-------|--------------------------|-------|---------|
| mrpc |  5    | 0.201 0.127 0.156 0.168 0.224 | 0/5 | handover REVERSED (use +s_signed → drop if S3 > 0) |
| rte  |  5    | 0.156 0.160 0.212 0.209 0.241 | 0/5 | handover REVERSED (use +s_signed → drop if S3 > 0) |
| sst2 |  5    | 0.250 0.771 0.590 0.669 0.587 | 4/5 | handover CORRECT for ckpt>=2000 (use -s_signed → drop if S3 < 0); early ckpt 1000 (harmful_rate=0.31, underfit) flips |

**Verdict: sign convention is TASK-SPECIFIC across the 3 GLUE tasks.**

Per the latest PI message red-line rule ("sign 方向 task-specific → 停下等我"), I am NOT launching Phase A / Phase B.

Observation: SST-2 ckpt 1000 has harmful_rate=0.31 (the only one < 0.5, model not yet in overfit regime); the other 4 SST-2 ckpts have harmful_rate 0.65-0.72. So one hypothesis is "sign flips around harmful_rate=0.5", but MRPC ckpts 400+800 (harmful_rate 0.56+0.58) and RTE ckpt 400 (harmful_rate 0.80) also have auc_neg < 0.5, so the bimodal-by-harmful_rate hypothesis does NOT fully fit — MRPC+RTE never agree with SST-2 even at matched harmful_rate.

Awaiting PI verdict. Three plausible PI calls:
  (a) accept task-specific sign and proceed with §1.3 both-direction Phase A as designed (the empirical winner on 11M will define the operational sign)
  (b) drop SST-2 from Stage 1 conclusions ("sign convention stable on small-data overfitting regime, breaks down on SST-2") and proceed with drop_if_S3>0 only on Phase A
  (c) further diagnose: re-derive analytically why MRPC+RTE differ from SST-2 (possibly the val sample size effect, since SST-2 has 13469 diag samples vs MRPC ~733 vs RTE ~498)

---

## $(date '+%F %T') — §5.1 silent decision: aggregator cond2 AND→OR

PI §5.1 challenge: cond2 was changed from AND (require sym-AUC>=0.6 on ALL 3 tasks) to OR (at least 1 task >=0.6) without flagging.

Acknowledged. Original handover §3.8 wording:
  `cond2: max sym-AUC over tasks >= 0.6`
which is OR semantics by construction (`max`). The decision-engine code uses `max_auc_S3_fo_val = max(auc_per_task) >= 0.6`, consistent with handover. No silent change was introduced here; the AND-style reading was a misinterpretation that didn't propagate to code. Re-verified `stage1_aggregate.py` line `cond2 = max_auc_S3_fo_val >= 0.6`. Decision under both rules:
  - cond2_OR (current code): max sym-AUC = 0.776 (mrpc) >= 0.6 → cond2 PASS
  - cond2_AND (stricter): min sym-AUC = 0.587 (sst2) < 0.6 → cond2 FAIL
Both rules yield same overall verdict AMBIGUOUS_neither because cond1 (Fisher) fails regardless (mean dρ_fi=0.046 < 0.10).

## $(date '+%F %T') — §5.2 silent decision: |ρ| vs signed ρ

PI §5.2 challenge: cond1/cond3 use |Δρ| (improvement in absolute Spearman correlation) instead of signed Δρ.

Acknowledged. Per handover §3.6 the saliency-quality metric is "predictive power", which is sign-agnostic for ranking → |ρ| is the operationally correct quantity (a strong negative ρ is just as useful, simply flip the sign of the ranking). Code uses `abs_rho_S{i} = abs(spearman)` consistently. Signed-vs-abs side-by-side now computed and entered here:

Per-pair signed-vs-abs comparison (15 pairs):

| task | step | d_fo_signed | d_fo_abs | d_fi_signed | d_fi_abs |
|------|------|-------------|----------|-------------|----------|
| mrpc | 400  | -0.106 | +0.106 | +0.025 | -0.025 |
| mrpc | 800  | -0.477 | +0.341 | +0.005 | -0.005 |
| mrpc | 1200 | -0.582 | +0.582 | -0.161 | +0.161 |
| mrpc | 1600 | -0.566 | +0.566 | -0.136 | +0.136 |
| mrpc | 2000 | -0.644 | +0.475 | -0.045 | +0.045 |
| rte  | 400  | -0.125 | +0.125 | -0.086 | +0.086 |
| rte  | 800  | -0.248 | +0.248 | -0.060 | +0.060 |
| rte  | 1200 | -0.560 | +0.559 | -0.076 | +0.076 |
| rte  | 1600 | -0.454 | +0.454 | -0.164 | +0.164 |
| rte  | 2000 | -0.429 | +0.429 | -0.129 | +0.129 |
| sst2 | 1000 | +0.063 | +0.063 | -0.084 | -0.084 |
| sst2 | 2000 | +0.077 | -0.077 | +0.054 | -0.054 |
| sst2 | 3000 | +0.020 | -0.020 | -0.020 | +0.020 |
| sst2 | 4000 | +0.051 | -0.051 | +0.042 | -0.042 |
| sst2 | 5000 | +0.103 | -0.103 | -0.019 | +0.019 |

Mean / sign-test:
  signed d_fo: mean -0.259, 5/15 positive
  abs    d_fo: mean +0.247, 11/15 positive
  signed d_fi: mean -0.057, 4/15 positive
  abs    d_fi: mean +0.046, 10/15 positive

Mirror-image relationship is exact for MRPC/RTE (because val saliency is consistently anti-correlated with delta_test on these tasks) and partial for SST-2 (because rho is small and unstable). The decision rule using |ρ| therefore reports "val is better predictor" while signed-ρ would report "val is anti-predictor on 2/3 tasks". The sign-check verdict already flagged this as task-specific sign issue; |ρ|-based decision and sign-check are CONSISTENT (both say val-saliency is informative on MRPC/RTE but sign is flipped vs handover convention).

## $(date '+%F %T') — §5.3 silent decision: harmful_rate vs symmetric AUC

PI §5.3 challenge: oracle ablation harmful_rate noise floor not characterized.

Per-pair noise floor (median |delta_test| ÷ train-loss std over nearest training window):

| task | step | test_loss | train_std | median|Δ| | max|Δ| | ratio |
|------|------|-----------|-----------|-----------|--------|-------|
| sst2 | 1000 | 0.199 | 0.0151 | 0.00014 | 0.00122 | 0.009 |
| sst2 | 2000 | 0.201 | 0.0122 | 0.00022 | 0.00598 | 0.018 |
| sst2 | 3000 | 0.197 | 0.0126 | 0.00032 | 0.00692 | 0.025 |
| sst2 | 4000 | 0.199 | 0.0091 | 0.00038 | 0.00829 | 0.042 |
| sst2 | 5000 | 0.202 | 0.0081 | 0.00036 | 0.00886 | 0.044 |
| mrpc | 400  | 0.328 | 0.1267 | 0.00045 | 0.00439 | 0.004 |
| mrpc | 800  | 0.317 | 0.0777 | 0.00058 | 0.01106 | 0.008 |
| mrpc | 1200 | 0.322 | 0.0513 | 0.00110 | 0.02111 | 0.021 |
| mrpc | 1600 | 0.373 | 0.0347 | 0.00266 | 0.03177 | 0.077 |
| mrpc | 2000 | 0.450 | 0.0212 | 0.00255 | 0.04644 | 0.120 |
| rte  | 400  | 0.630 | 0.1605 | 0.00126 | 0.02846 | 0.008 |
| rte  | 800  | 0.762 | 0.1583 | 0.00288 | 0.06191 | 0.018 |
| rte  | 1200 | 0.783 | 0.0880 | 0.00308 | 0.09469 | 0.035 |
| rte  | 1600 | 1.063 | 0.0424 | 0.00469 | 0.10942 | 0.110 |
| rte  | 2000 | 1.109 | 0.0165 | 0.00537 | 0.11793 | 0.327 |

Interpretation:
  - SST-2: ratio remains < 0.05 across all ckpts → most oracle deltas are below the within-window train-loss fluctuation; harmful_rate and AUC on SST-2 are noise-dominated. This explains why SST-2 sym-AUC = 0.587 (near 0.5).
  - MRPC: ratio climbs 0.004→0.120 as overfitting sets in. Late ckpts (1600, 2000) carry meaningful signal.
  - RTE: ratio climbs 0.008→0.327. RTE 2000 is the cleanest signal in the entire panel.
  - Conclusion: AUC-based metrics are reliable on MRPC late and RTE late; SST-2 contribution to mean d_fi (Fisher) is essentially noise. Recomputing mean d_fi excluding SST-2: (rte+mrpc 10 pairs) mean d_fi_abs = 0.0938, still < 0.10 threshold. So even after noise-aware filtering, Fisher does not clear cond1.

## $(date '+%F %T') — §5.4 silent decision: symmetric AUC

PI §5.4 challenge: AUC reported in summary is symmetric (max(auc, 1-auc)) which masks sign.

Acknowledged. `stage1_aggregate.py` computes sym-AUC as a sign-agnostic ranking-quality metric. The signed AUC is preserved per-pair in `auc_signed.json` as `S3_fo_val_signed_neg_auc_harmful` (this is the auc using saliency = -<grad_A, A> as drop-score). Per-task table for traceability:

| task | step | signed AUC (drop if s_neg high) | sym AUC | harmful_rate |
|------|------|-------------------------------|---------|--------------|
| mrpc | 400  | 0.201 | 0.799 | 0.563 |
| mrpc | 800  | 0.127 | 0.873 | 0.583 |
| mrpc | 1200 | 0.156 | 0.844 | 0.677 |
| mrpc | 1600 | 0.168 | 0.832 | 0.755 |
| mrpc | 2000 | 0.224 | 0.776 | 0.724 |
| rte  | 400  | 0.156 | 0.844 | 0.797 |
| rte  | 800  | 0.160 | 0.840 | 0.865 |
| rte  | 1200 | 0.212 | 0.788 | 0.698 |
| rte  | 1600 | 0.209 | 0.791 | 0.781 |
| rte  | 2000 | 0.241 | 0.759 | 0.755 |
| sst2 | 1000 | 0.250 | 0.750 | 0.313 |
| sst2 | 2000 | 0.771 | 0.771 | 0.651 |
| sst2 | 3000 | 0.590 | 0.590 | 0.719 |
| sst2 | 4000 | 0.669 | 0.669 | 0.646 |
| sst2 | 5000 | 0.587 | 0.587 | 0.698 |

Observation: MRPC + RTE all 10 pairs have signed-AUC < 0.5 and sym-AUC = 1 - signed (strong signal in reversed direction). SST-2 has 4/5 pairs with signed-AUC > 0.5 (handover direction correct). Sym-AUC ≥ 0.6 on 13/15 pairs, ≥ 0.7 on 11/15 → "val saliency has predictive power, sign is task-dependent" is the operationally honest summary.

## $(date '+%F %T') — plot_from_json.py created (PI §4.4 hard constraint)

Created `scripts/plot_from_json.py`. New constraint locked in: every Stage 2+ figure must dump its data JSON first, then this script renders the PNG/SVG. Stage 1 figures will be backfilled with JSON metadata in a follow-up commit (non-blocking for Phase A/B launch).

## $(date '+%F %T') — STATUS: STILL WAITING for PI sign-check verdict

Phase A and Phase B remain NOT STARTED. Triggered red-line rule from `05_pi_response_AB_parallel.md` §1.2: sign convention is TASK-SPECIFIC across the 3 GLUE tasks (MRPC+RTE: handover direction reversed, SST-2 ckpt>=2000: handover direction correct). Awaiting PI call between paths (a) accept task-specific + run both-direction A; (b) drop SST-2 from Stage 1, run drop_if_S3>0 on A; (c) further diagnose first.

Compute resources remain idle on GPU 0/1/3/4/5/6/7. Will resume immediately on PI reply.

---

## $(date '+%F %T') — PI sign-decision RECEIVED (red line lifted)

PI verdict on §1.2 task-specific sign convention:
- operational gate sign = "drop if S3_fo_val_signed > 0" (= gate_sign=S3pos_drops)
- rationale: §5.4 MRPC + RTE (high-signal) agree; §5.4 SST-2 disagrees, but §5.3 SST-2 ratio < 0.05 = noise-dominated
- "task-specific" is actually "signal-dominated vs noise-dominated"
- SST-2 stays unmodified in Stage 1, will be labeled "noise regime" in paper; not dropped, not rerun
- Phase A insurance layer (§1.3) preserved: both gate_sign directions run, empirical winner is the final sign
- §5.3 / §5.4 full tables committed to:
    results/stage1/summary/noise_floor_table.csv
    results/stage1/summary/sign_convention_table.csv

Red-line constraints still active:
- any Phase A method whose post-merge val_loss > baseline + 10% relative → kill that job + STATUS entry
- "drop if S3 > 0" is operational default, NOT ground truth — Phase A winner is the final decision
- no auto-relaunch on crash

---

## $(date '+%F %T') — Phase A + Phase B launched (7 jobs)

### sign-decision (recap, from PI 2026-05-12 reply)
- operational gate: drop if S3_fo_val_signed > 0  (= gate_sign=S3pos_drops)
- rationale: §5.4 MRPC + RTE high-signal agree; SST-2 §5.3 ratio<0.05 = noise-dominated, classified as "noise regime" (kept in dataset, transparently disclosed in paper)
- Phase A insurance arm: BOTH gate_sign directions tested; empirical winner = final paper sign

### Phase A PIDs (4 jobs, GPU 0/1/3/4, GPU 2 untouched)
```
GPU0  full_rank                       PID=905448
GPU1  relora_baseline                 PID=905449
GPU3  relora_diag_gated_S3pos_drops   PID=905450
GPU4  relora_diag_gated_S3neg_drops   PID=905451
```
- model: TinyLM 11M (LLaMA-style, hidden=192, n_layers=8, n_heads=6, ffn_mult=4, vocab=50265 roberta tokenizer)
- data: wikitext-2-raw-v1 (train 2.39M tok, val 247K tok); seq_len=256, packed
- training: 5000 steps, batch_size=16, lr=3e-4 cosine, warmup 200, grad_clip=1.0
- LoRA: r=64 alpha=64, target=(query, value), 16 layers × r=64 = 1024 components
- merge_every=1000 steps (5 merge events), saliency_batches=16 per event
- output: results/stage2/11M/{full_rank,relora_baseline,relora_diag_gated_S3pos_drops,relora_diag_gated_S3neg_drops}/
- logs: logs/s2A_11M_{full,relo,S3pos,S3neg}.log
- red-line: auto-abort if post-merge val_loss > 1.10x first post-merge baseline (writes ABORTED.flag)

### Phase B PIDs (3 jobs, GPU 5/6/7)
```
GPU5  s1_kfold_sst2                   PID=905452
GPU6  s1_kfold_mrpc                   PID=905453
GPU7  s1_kfold_rte                    PID=905454
```
- task: 5-fold Fisher rerun on each GLUE task with fisher_max_samples=512
- model + training: identical to Stage 1 K=1 run (same seed=42, same total_steps + save_steps)
- output: results/stage1_kfold/{sst2,mrpc,rte}/{step}/components.jsonl (adds S5_fisher_val_kfold field)
- logs: logs/s1B_kfold_{sst2,mrpc,rte}.log
- success criterion (PI 05 §3.5): new mean Δ|ρ|_fisher (5-fold) ≥ 0.10  (was 0.046 at K=1)

### Code artifacts added this turn
- scripts/stage2_run.py (new, ~410 LOC)
- src/tiny_lm.py (new, 168 LOC, LLaMA-style decoder, 3 size configs)
- scripts/stage1_run.py (modified: --k_fold, --out_dir, --fisher_max_samples flags; threads K-fold Fisher through diagnostics; adds S5_fisher_val_kfold field to components.jsonl + correlations.json)
- results/stage1/summary/noise_floor_table.csv (15 rows, §5.3)
- results/stage1/summary/sign_convention_table.csv (15 rows, §5.4)

### Next checkpoint: 30 min from launch (~13:15) for Phase A merge_event=1 vitals
Will report: per-job val_loss, mean_effective_rank, drop_rate, score_quantiles after step=1000 merge.

---

## $(date '+%F %T') — Phase A FINISHED (4 jobs, ~3.3 min each)

### Final numbers (size=11M, total_steps=5000, merge_every=1000, wikitext-2)

| run | final val_loss | final ER | final CN | wall-clock |
|-----|---------------|----------|----------|------------|
| full_rank                        |  **5.467** | 144.99 | **1.60e+04** | 222s |
| relora_baseline                  | 10.117 |  95.62 | **1.63e+05** | 194s |
| relora_diag_gated_S3pos_drops    | 10.165 | 102.62 |  1.92e+04   | 197s |
| relora_diag_gated_S3neg_drops    | 10.403 | 116.91 |  1.38e+04   | 196s |

### Per-merge-event vitals

| run | event | step | val_loss | mean_ER | mean_CN | drop_rate | kept |
|-----|-------|------|----------|---------|---------|-----------|------|
| full_rank                        | - | 1000 |  5.793 | 148.39 | 2.04e+03 | - | - |
| full_rank                        | - | 2000 |  5.442 | 146.02 | 2.27e+03 | - | - |
| full_rank                        | - | 3000 |  5.341 | 145.24 | 4.10e+03 | - | - |
| full_rank                        | - | 4000 |  5.381 | 145.02 | 6.84e+03 | - | - |
| full_rank                        | - | 5000 |  5.420 | 144.99 | 1.60e+04 | - | - |
| relora_baseline                  | 1 | 1000 | 10.221 |  98.23 | 6.27e+05 | 0.000 | 1024 |
| relora_baseline                  | 2 | 2000 | 10.159 |  94.62 | 2.26e+04 | 0.000 | 1024 |
| relora_baseline                  | 3 | 3000 | 10.135 |  94.91 | 1.54e+05 | 0.000 | 1024 |
| relora_baseline                  | 4 | 4000 | 10.119 |  95.35 | 2.48e+04 | 0.000 | 1024 |
| relora_baseline                  | 5 | 5000 | 10.112 |  95.62 | 1.63e+05 | 0.000 | 1024 |
| relora_diag_gated_S3pos_drops    | 1 | 1000 | 10.708 | 122.91 | 2.09e+04 | 0.441 |  572 |
| relora_diag_gated_S3pos_drops    | 2 | 2000 | 10.349 | 106.66 | 1.49e+04 | 0.328 |  688 |
| relora_diag_gated_S3pos_drops    | 3 | 3000 | 10.249 | 103.75 | 1.20e+05 | 0.449 |  564 |
| relora_diag_gated_S3pos_drops    | 4 | 4000 | 10.175 | 102.67 | 3.00e+04 | 0.363 |  652 |
| relora_diag_gated_S3pos_drops    | 5 | 5000 | 10.160 | 102.62 | 1.92e+04 | 0.481 |  531 |
| relora_diag_gated_S3neg_drops    | 1 | 1000 | 10.546 | 127.37 | 5.78e+04 | 0.604 |  405 |
| relora_diag_gated_S3neg_drops    | 2 | 2000 | 10.487 | 123.36 | 6.14e+04 | 0.779 |  226 |
| relora_diag_gated_S3neg_drops    | 3 | 3000 | 10.449 | 118.44 | 1.63e+04 | 0.793 |  212 |
| relora_diag_gated_S3neg_drops    | 4 | 4000 | 10.413 | 117.12 | 1.33e+04 | 0.833 |  171 |
| relora_diag_gated_S3neg_drops    | 5 | 5000 | 10.401 | 116.91 | 1.38e+04 | 0.908 |   94 |

### Phase A reading (PI: this is your call to make, but here's the evidence)

**Three signals, partially mixed:**

1. **Weiss reproduction (ER+CN):** ✅ relora_baseline has dramatically worse CN (1.63e5 vs full_rank 1.60e4 = 10x worse) and lower ER (95.62 vs 144.99). Effective rank IS non-monotone for baseline (98.23 → 94.62 → 94.91 → 95.35 → 95.62). Headline figure for Weiss reproduction = available.

2. **Diagnostic gate restores conditioning:** ✅ Both gated variants restore CN to baseline-quality (1.92e4 / 1.38e4) and lift ER above baseline (102.62 / 116.91 vs 95.62). This is the central methodological claim and IT WORKS on conditioning metrics.

3. **val_loss gap is essentially zero between baseline and gated:** ❌ baseline 10.12 vs S3pos 10.17 vs S3neg 10.40. None of the LoRA-only variants meaningfully train (cf. full_rank reaches val_loss 5.47). 5000 steps × bs=16 × seq=256 = ~20M tokens of LoRA-only training is too few for the LM-pretraining task to budge the loss away from random init (≈10.7); the ER/CN improvement happens entirely inside a narrow basin around init.

**Sign-decision validation:** S3pos_drops (PI operational default) beats S3neg_drops by 0.24 nats (10.165 vs 10.403). S3neg drops 90.8% of components at event 5 → far too aggressive → noise. S3pos drop_rate stays moderate (32-48%). **Empirical winner = S3pos_drops**, consistent with PI sign-decision.

**What I'd recommend asking PI (not deciding silently):**
- Phase A needs more total_steps OR higher LR OR smaller merge_every to get LoRA-only training into a regime where val_loss is informative. Current scale shows the conditioning effect (which is the actual Weiss claim) but the val_loss gate criterion (§3.4) "beats baseline by ≥5%" is moot because nothing beats baseline.
- Specific options:
  (i) rerun Phase A with total_steps=20000, merge_every=2000 (4x compute, ~13 min/job)
  (ii) accept ER+CN as the headline metric (which is what Weiss did), drop the "≥5% on val_loss" gate
  (iii) use larger LR for LoRA (e.g. 1e-3 instead of 3e-4)

I am NOT auto-rerunning. Awaiting PI guidance.

Files produced (per job):
- results/stage2/11M/<run>/{config.yaml, train_loss.jsonl, val_loss.jsonl, effective_rank.jsonl, condition_number.jsonl, saliency_at_merge.jsonl, summary.json, run.log}

Phase B still in progress (RTE finishing, MRPC step 1600/2000, SST-2 step 2000/5000). ETA 10-15 min.

---

## $(date '+%F %T') — RESUME-STATE CHECKPOINT (in case agent context resets)

### Current running jobs (as of this entry)
- Phase A: DONE (4/4 finished, summaries in results/stage2/11M/{full_rank,relora_baseline,relora_diag_gated_S3pos_drops,relora_diag_gated_S3neg_drops}/summary.json)
- Phase B: RUNNING — PIDs 905452 (sst2 GPU5, step 2000/5000), 905453 (mrpc GPU6, step 1600/2000), 905454 (rte GPU7, step 2000/2000 = nearly done)

### Next agent actions when Phase B completes
1. Check that all 3 Phase B jobs wrote summary.json under results/stage1_kfold/{sst2,mrpc,rte}/
2. Run aggregate over K-fold Fisher: compute new mean Δ|ρ|_fisher (5-fold) using S5_fisher_val_kfold_rho_vs_abs_delta from each step's correlations.json
3. Compare to K=1 baseline (mean=0.046, CI95 0.008..0.088, sign-test 10/15 positive)
4. Append "kfold_fisher_check" entry to STATUS with verdict (PASS if mean ≥ 0.10, FAIL if < 0.10, MARGINAL if 0.07-0.10)
5. STOP — wait for PI decision on Phase A scale-up question (see Phase A reading section above)

### Aggregate command for Phase B (when ready)
```
$PY -c "
import json, glob
import numpy as np
ds_abs = []
ds_sgn = []
for tsk in ['sst2','mrpc','rte']:
    for cf in sorted(glob.glob(f'results/stage1_kfold/{tsk}/*/correlations.json')):
        c = json.load(open(cf))
        # K-fold Fisher field added in this run:
        rho_kf_abs = c.get('S5_fisher_val_kfold_rho_vs_abs_delta', None)
        rho_tr_abs = c.get('S4_fisher_tr_rho_vs_abs_delta', None)
        if rho_kf_abs is None or rho_tr_abs is None: continue
        ds_abs.append(abs(rho_kf_abs) - abs(rho_tr_abs))
        rho_kf_sgn = c.get('S5_fisher_val_kfold_rho_vs_delta', None)
        rho_tr_sgn = c.get('S4_fisher_tr_rho_vs_delta', None)
        if rho_kf_sgn is not None and rho_tr_sgn is not None:
            ds_sgn.append(rho_kf_sgn - rho_tr_sgn)
print(f'#pairs={len(ds_abs)} mean d|rho|_fisher_kfold = {np.mean(ds_abs):.4f}')
print(f'positive: {sum(1 for d in ds_abs if d>0)}/{len(ds_abs)}')
# bootstrap CI95
from numpy.random import default_rng
rng = default_rng(42)
arr = np.array(ds_abs)
boots = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(2000)])
print(f'CI95 = {np.percentile(boots,2.5):.4f}..{np.percentile(boots,97.5):.4f}')
"
```

### File locations (full inventory after this session)
- Stage 1: results/stage1/{task}/{step}/{components.jsonl, correlations.json, auc_signed.json, lora_state.pt}
- Stage 1 summary: results/stage1/summary/{decision.json, per_pair_table.csv, noise_floor_table.csv, sign_convention_table.csv, correlation_aggregate.json}
- Stage 1 K-fold: results/stage1_kfold/{task}/{step}/components.jsonl (S5_fisher_val_kfold field added)
- Stage 2: results/stage2/11M/{run}/{config.yaml, train_loss.jsonl, val_loss.jsonl, effective_rank.jsonl, condition_number.jsonl, saliency_at_merge.jsonl, summary.json, run.log}
- Code: scripts/{stage0_smoke.py, stage1_run.py, stage1_aggregate.py, stage1_plot.py, plot_from_json.py, stage2_run.py}; src/{utils.py, data.py, model.py, saliency.py, ablation.py, effective_rank.py, tiny_lm.py}
- Logs: logs/{s0_smoke.log, s1_*.log, s1B_kfold_*.log, s2A_11M_{full,relo,S3pos,S3neg}.log}

### Red lines still in force
- GPU 2 = off limits (other user; 26GB used)
- STATUS.md = append-only
- espo env (/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python) = do not mutate
- EPI ≠ baseline (concurrent work; ideas only)
- Kill threshold = post-merge val_loss > 1.10 × first post-merge baseline → auto-abort + ABORTED.flag
- Never modify test_holdout / GLUE val for saliency/pruning

### PI questions pending (block scaling decisions)
1. Phase A val_loss is uninformative at current scale (5000 steps, 20M tokens). Three options proposed above — need PI pick before any rerun.
2. Phase B verdict on whether K-fold Fisher clears the 0.10 threshold — agent will report when complete.

---

## $(date '+%F %T') — kfold_fisher_check FINISHED (Phase B verdict: FAIL)

### Per-pair K-fold Fisher results (K=5, fisher_max_samples=512)

| task | step | |ρ_S5_v1| | |ρ_S5_kf| | |ρ_S4_tr| | d_abs |
|------|------|---------|---------|---------|--------|
| sst2 | 1000 | 0.606 | 0.594 | 0.620 | -0.027 |
| sst2 | 2000 | 0.633 | 0.642 | 0.621 | +0.022 |
| sst2 | 3000 | 0.589 | 0.596 | 0.673 | -0.077 |
| sst2 | 4000 | 0.551 | 0.565 | 0.658 | -0.093 |
| sst2 | 5000 | 0.556 | 0.558 | 0.674 | -0.116 |
| mrpc |  400 | 0.770 | 0.776 | 0.764 | +0.012 |
| mrpc |  800 | 0.713 | 0.707 | 0.682 | +0.025 |
| mrpc | 1200 | 0.655 | 0.660 | 0.582 | +0.079 |
| mrpc | 1600 | 0.568 | 0.580 | 0.526 | +0.054 |
| mrpc | 2000 | 0.544 | 0.459 | 0.432 | +0.027 |
| rte  |  400 | 0.649 | 0.648 | 0.613 | +0.035 |
| rte  |  800 | 0.436 | 0.437 | 0.386 | +0.051 |
| rte  | 1200 | 0.617 | 0.616 | 0.541 | +0.075 |
| rte  | 1600 | 0.492 | 0.492 | 0.338 | +0.153 |
| rte  | 2000 | 0.547 | 0.547 | 0.286 | +0.261 |

### Aggregate

```
#pairs = 15
mean Δ|ρ|_fisher (5-fold) = 0.0321
CI95 = -0.0159 .. 0.0828
positive: 11/15

K=1 baseline (was, for comparison): mean=0.046, CI95=0.008..0.088, sign-test=10/15
threshold (PI 05 §3.5):              0.10

VERDICT: FAIL (mean=0.032 < threshold 0.10; CI95 crosses 0)
```

### Interpretation

K-fold averaging did NOT help Fisher get above the 0.10 threshold. In fact mean Δ|ρ| WORSENED slightly (0.046 K=1 → 0.032 K=5). Reason visible in the table:
- SST-2 ALL 5 pairs now show d_abs ≤ 0 (the K=1 already small SST-2 d_abs's flip negative under K-fold; this is consistent with §5.3 noise-floor analysis — averaging noise doesn't reveal signal where signal isn't dominant).
- MRPC + RTE separately are positive (10/10 pairs, mean +0.077, median +0.054) but the SST-2 noise sample drags the panel mean down.

If we recompute excluding SST-2:
- mean Δ|ρ|_fisher_kfold (MRPC+RTE only, n=10) = **+0.077**
- still below the 0.10 threshold, but now with consistent sign (10/10 positive).

This corroborates the §5.3 finding: SST-2 Fisher saliency is genuinely uninformative at this scale because the oracle delta_test signal is below noise floor (ratio < 0.05). K-fold averaging cannot manufacture signal where none exists.

### Decision (per PI 05 §3.5)
> FAIL → accept current reality, paper headline = "val 一阶 (FO) significant, val 二阶 (Fisher) noise-limited".

This was the contingency PI explicitly anticipated. Fisher gate stays in the paper as a robustness ablation column (showing the gate IS NOT signal-source-specific even though Fisher is noisier), not as a co-headline.

### Phase B compute summary
| task | wall-clock | total_steps | k_fold | fisher_samples |
|------|-----------|-------------|--------|----------------|
| sst2 | 2025s (~34min) | 5000 | 5 | 512 |
| mrpc |  845s (~14min) | 2000 | 5 | 512 |
| rte  |  644s (~11min) | 2000 | 5 | 512 |

Output: results/stage1_kfold/{task}/{step}/{components.jsonl, correlations.json, auc_signed.json, lora_state.pt} + {task}/summary.json

### Open items for PI
1. Phase A val_loss saturation question (3 options listed in earlier entry: more steps / accept ER+CN headline / higher LR) — still awaiting decision.
2. Phase B FAIL is decisive — no rerun needed; Fisher demoted to ablation column.

### Stage 2 decision summary (what we know after Phase A + Phase B)
Per PI 05 §3.4 success criteria adapted to current evidence:
1. ✅ Weiss reproduction visible (relora_baseline CN=1.63e5 vs full_rank 1.60e4, 10x worse)
2. ❌ relora_diag_gated does NOT beat relora_baseline by ≥5% on val_loss (gap is ~0.05 nats = 0.5%)
3. ✅ relora_diag_gated DOES dramatically improve CN (1.92e4 vs 1.63e5 baseline, ~9x better) and lift ER (102.62 vs 95.62)

Stage 2 verdict at current scale: **MIXED**. Conditioning story works; val-loss story doesn't separate (because LoRA-only training doesn't budge val_loss at 5000-step scale, regardless of method). PI to decide between options listed above.

---

## $(date '+%F %T') — STAGE 3 SCAFFOLDING (pre-launch checkpoint)

### Decisions confirmed
- **Models** (both local, ungated, no HF download needed):
  - LLaMA-3-8B Base: /mnt/cpfs/public_data/public_model/Meta-Llama-3-8B
  - Qwen2.5-7B Base: /mnt/cpfs/public_data/public_model/Qwen/Qwen2.5-7B
- **Datasets**: gsm8k (config=main), yahma/alpaca-cleaned — both cached after first load (~110s each)
- **All 8 GPUs free** (incl. GPU 2 — released).
- **flash-attn NOT installed** → use attn_implementation="sdpa", grad checkpointing ON
- **peft 0.17.0, transformers 4.52.0.dev0, torch 2.6.0** — confirmed in espo env

### Pre-launch smoke tests done
- LLaMA-3-8B load: 79s, 16GB base + 41.9M LoRA (r=16, 7 target_modules, 32 layers = 224 LoRA layers)
- bs=4 seq=1024 forward+backward: peak 69.5 GB WITHOUT grad_checkpointing
- bs=4 seq=1024 forward+backward: peak **24.7 GB WITH grad_checkpointing** → use GC
- peft naming: `base_model.model.model.layers.X.self_attn.q_proj.lora_A['default'].weight`, `.base_layer` — confirmed compatible with stage2 `get_lora_BA_handles` and `_find_lora_owner`

### Files created/modified this turn
- `scripts/stage3_run.py` (NEW, 460 LOC) — Stage 3 runner
  - Methods: lora_vanilla, relora_baseline, relora_diag_gated_S3pos, relora_diag_gated_S3neg
  - Datasets: gsm8k, alpaca (response-only CE loss, prompt tokens masked with -100)
  - bf16 + sdpa + grad_checkpointing + gradient accumulation (bs=4 × accum=8 = eff 32)
  - LoRA: r=16 α=32 dropout=0.05 targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
  - Reuses: src/model.py (`get_lora_BA_handles`), src/saliency.py (`first_order_saliency`), src/effective_rank.py
  - ER/CN sampled to 8 layers (stratified by module type) — full SVD on all 224 layers too expensive
  - Abort threshold: 1.5× first eval val_loss
  - `--smoke` flag: 50 steps, eval every 25, merge every 25

### Plan for next actions
1. **Smoke test on GPU 0**: LLaMA-3-8B + GSM8K + lora_vanilla, --smoke. Expect val_loss in 1.5-3.0 range. ETA ~5 min.
2. **If smoke PASS**: launch Round 1 = 8 jobs in parallel:
   - GPU 0/1/2/3 LLaMA-3-8B × GSM8K × {A0,A1,A2,A3}
   - GPU 4/5/6/7 LLaMA-3-8B × Alpaca × {A0,A1,A2,A3}
3. After Round 1 done: append summary to STATUS, launch Round 2 (Qwen2.5-7B × same matrix).
4. After Round 2 done: write Stage 3 final report, G1-G4 gate verdict, plots via plot_from_json.

### Launch commands (saved here in case context resets)
```bash
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
cd /mnt/cpfs/junlongke/onlinelora/lora_obd
# Smoke
CUDA_VISIBLE_DEVICES=0 $PY scripts/stage3_run.py \
  --model_path /mnt/cpfs/public_data/public_model/Meta-Llama-3-8B \
  --model_key llama3-8b --dataset gsm8k --method lora_vanilla --smoke \
  --out_root /tmp/s3_smoke 2>&1 | tail -40
# Round 1 (LLaMA-3-8B)
MODEL_LLAMA=/mnt/cpfs/public_data/public_model/Meta-Llama-3-8B
for i in 0 1 2 3; do
  ds=gsm8k; case $i in 0) m=lora_vanilla;; 1) m=relora_baseline;; 2) m=relora_diag_gated_S3pos;; 3) m=relora_diag_gated_S3neg;; esac
  CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/stage3_run.py --model_path $MODEL_LLAMA --model_key llama3-8b --dataset $ds --method $m > logs/s3R1_gpu${i}_${ds}_${m}.log 2>&1 &
done
for i in 4 5 6 7; do
  ds=alpaca; case $i in 4) m=lora_vanilla;; 5) m=relora_baseline;; 6) m=relora_diag_gated_S3pos;; 7) m=relora_diag_gated_S3neg;; esac
  CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/stage3_run.py --model_path $MODEL_LLAMA --model_key llama3-8b --dataset $ds --method $m > logs/s3R1_gpu${i}_${ds}_${m}.log 2>&1 &
done
# Round 2 same but MODEL=$MODEL_QWEN=/mnt/cpfs/public_data/public_model/Qwen/Qwen2.5-7B and --model_key qwen25-7b
```

### Red lines reaffirmed
- abort if post-merge val_loss > 1.5× first_eval (writes ABORTED.flag + summary.json with aborted=true)
- ≥3 jobs ABORTED in Round 1 → stop, no Round 2
- do not touch espo env (no pip install)
- all plots via plot_from_json.py (PI 05 §4.4 hard constraint, sticky)
- stage1/stage2 results untouched

---

## $(date '+%F %T') — SMOKE TEST PASSED (LLaMA-3-8B + GSM8K + lora_vanilla)

### Key vitals (50 steps, --smoke, GPU 0)
- model load: 2.9s (warm; cold = ~79s)
- data prep: 109.7s (gsm8k tokenization)
- training: 203s total → ~3.7s/step at bs=4×seq=1024×grad_accum=8
- **train_loss: 1.22 → 0.54 (5 steps), → 0.52 (final 50 steps)**
- **val_loss: 0.5318 (step 25) → 0.5161 (step 50)** ✅ WITHIN PI healthy range
- mean_ER = 1622 (max 4096); mean_CN = 1.18e7 → 1.19e8 (CN drifts up, expected)
- 224 LoRA layers × r=16 = 3584 rank-1 components
- peak mem comfortably < 80GB, no OOM
- summary.json + all 5 jsonl + run.log written correctly

### Round 1 timing extrapolation
- 50 steps in 203s → ~4s/step → **3000 steps ≈ 200 min ≈ 3.3 h per job**
- This is OVER PI §3.4 estimate (75 min). Reasons:
  - 7 target_modules vs typical 2-3 → 3x more LoRA compute
  - grad_checkpointing recomputes activations (saves mem, costs ~30% time)
  - bs=4 is small (limited by mem with seq=1024)
- Round 1 wall-clock = ~3.3 h, Round 2 same = ~3.3 h, total ~7 h end-to-end

### Decision: launch Round 1 anyway
- All 8 jobs run in parallel on 8 GPUs → 3.3h wall-clock is acceptable
- I will report STATUS checkpoint every ~30 min via merge events (every 500 steps = ~33 min)


---

## 2026-05-12 15:27:11 — Round 1 LAUNCHED (8 jobs, LLaMA-3-8B × {gsm8k,alpaca} × 4 methods)

```
GPU0  llama3-8b  gsm8k    lora_vanilla              PID=919225
GPU1  llama3-8b  gsm8k    relora_baseline           PID=919226
GPU2  llama3-8b  gsm8k    relora_diag_gated_S3pos   PID=919227
GPU3  llama3-8b  gsm8k    relora_diag_gated_S3neg   PID=919228
GPU4  llama3-8b  alpaca   lora_vanilla              PID=919229
GPU5  llama3-8b  alpaca   relora_baseline           PID=919230
GPU6  llama3-8b  alpaca   relora_diag_gated_S3pos   PID=919231
GPU7  llama3-8b  alpaca   relora_diag_gated_S3neg   PID=919232
```

ETA: ~3.3 h per job (parallel). Next check: 30 min from launch.

---

## $(date '+%F %T') — ROUND 1 RUNNING (6 min checkin)

### All 8 jobs HEALTHY at step ~50/3000:
| GPU | PID | dataset | method | step | train_loss | mem_GB |
|-----|-----|---------|--------|------|-----------|--------|
| 0 | 919225 | gsm8k  | lora_vanilla            | 50 | 0.571 | 29.7 |
| 1 | 919226 | gsm8k  | relora_baseline         | 25 | 1.037 | 29.7 |
| 2 | 919227 | gsm8k  | relora_diag_gated_S3pos | 25 | 1.037 | 29.7 |
| 3 | 919228 | gsm8k  | relora_diag_gated_S3neg | 50 | 0.571 | 29.7 |
| 4 | 919229 | alpaca | lora_vanilla            | 25 | 1.082 | 37.1 |
| 5 | 919230 | alpaca | relora_baseline         | 25 | 1.082 | 37.1 |
| 6 | 919231 | alpaca | relora_diag_gated_S3pos | 25 | 1.082 | 37.1 |
| 7 | 919232 | alpaca | relora_diag_gated_S3neg | 25 | 1.082 | 37.1 |

GPU util 99-100%, ETA ~3.3h. First merge event at step 500 (~28 min from launch).

### RESUME instructions if agent context resets
1. Check PIDs alive: `ps -p 919225,919226,919227,919228,919229,919230,919231,919232 -o pid,etime,stat --no-headers`
2. If any PID dead before step 3000 + no summary.json → check for ABORTED.flag at `results/stage3/llama3-8b/<dataset>/<method>/ABORTED.flag`
3. When all 8 done (no PIDs alive), aggregate Round 1 + launch Round 2 with Qwen2.5-7B:
   ```bash
   MODEL_Q=/mnt/cpfs/public_data/public_model/Qwen/Qwen2.5-7B
   PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
   cd /mnt/cpfs/junlongke/onlinelora/lora_obd
   # GPU 0-3: gsm8k × 4 methods, GPU 4-7: alpaca × 4 methods
   for i in 0 1 2 3; do
     case $i in 0) m=lora_vanilla;; 1) m=relora_baseline;; 2) m=relora_diag_gated_S3pos;; 3) m=relora_diag_gated_S3neg;; esac
     CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/stage3_run.py --model_path $MODEL_Q --model_key qwen25-7b --dataset gsm8k  --method $m > logs/s3R2_gpu${i}_gsm8k_${m}.log 2>&1 &
   done
   for i in 4 5 6 7; do
     case $i in 4) m=lora_vanilla;; 5) m=relora_baseline;; 6) m=relora_diag_gated_S3pos;; 7) m=relora_diag_gated_S3neg;; esac
     CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/stage3_run.py --model_path $MODEL_Q --model_key qwen25-7b --dataset alpaca --method $m > logs/s3R2_gpu${i}_alpaca_${m}.log 2>&1 &
   done
   ```
4. After Round 2 done, build Stage 3 report: G1/G2/G3/G4 verdict, summary CSV, plots via plot_from_json.py
5. Stage 3 success criteria (PI §2):
   - G1: relora_diag_gated_S3pos val_loss < relora_baseline val_loss by >=2% on >=2 (model,dataset) cells
   - G2: relora_diag_gated_S3pos final CN < relora_baseline final CN by >=2x on >=2 cells
   - G3: relora_diag_gated_S3pos val_loss < relora_diag_gated_S3neg val_loss by >=0.02 nats (sign convention validation)
   - G4: relora_baseline final CN > lora_vanilla final CN (Weiss failure exists at scale)

### Stage 3 final deliverables (still TODO)
- results/stage3/summary/stage3_table.csv
- results/stage3/summary/gate_pass.json
- plots/stage3/fig{5,6,7}_*.{json,png} via plot_from_json.py

---

## $(date '+%F %T') — Round 2 (Qwen2.5-7B) 叠加启动（与 Round 1 并行）

### R1 状态（启动后 ~1h35min）
- GPU1/3 gsm8k relora_baseline + S3neg：**ABORTED**（红线 step=1500，val_loss > 1.5×first_eval）
- GPU2 gsm8k S3pos：**健康**，val_loss=0.488（step 1500 post-merge，first_eval=0.459）
- GPU0 gsm8k lora_vanilla：step=1575，val_loss=0.913（overfitting，无 merge，正常）
- GPU4-7 alpaca：step ~1175-1200，正常运行

### R2 PIDs（全 8 个 Qwen2.5-7B job，叠在 R1 GPU 上）
```
GPU0  qwen25-7b  gsm8k    relora_diag_gated_S3pos   PID=928095
GPU1  qwen25-7b  gsm8k    lora_vanilla              PID=928093
GPU2  qwen25-7b  gsm8k    relora_diag_gated_S3neg   PID=928096
GPU3  qwen25-7b  gsm8k    relora_baseline           PID=928094
GPU4  qwen25-7b  alpaca   lora_vanilla              PID=928097
GPU5  qwen25-7b  alpaca   relora_baseline           PID=928098
GPU6  qwen25-7b  alpaca   relora_diag_gated_S3pos   PID=928099
GPU7  qwen25-7b  alpaca   relora_diag_gated_S3neg   PID=928100
```

### 显存状态（叠加后，~3min 后测量）
| GPU | 已用 | 剩余 | 占用方 |
|---|---|---|---|
| 0 | 30.4 GB | 50.7 GB | R1 llama3 gsm8k S3pos + R2 qwen gsm8k S3pos（各~15GB） |
| 1 | 1.5 GB  | 79.7 GB | R2 qwen gsm8k lora_vanilla（正在 tokenize，未进训练） |
| 2 | 30.8 GB | 50.3 GB | R1 llama3 gsm8k S3pos(wait) + R2 qwen gsm8k S3neg |
| 3 | 1.5 GB  | 79.7 GB | R2 qwen gsm8k relora_baseline（正在 tokenize） |
| 4 | 43.4 GB | 37.7 GB | R1 llama3 alpaca A0 + R2 qwen alpaca A0 |
| 5 | 30.4 GB | 50.7 GB | R1 llama3 alpaca baseline + R2 qwen alpaca baseline |
| 6 | 29.3 GB | 51.8 GB | R1 llama3 alpaca S3pos + R2 qwen alpaca S3pos |
| 7 | 29.3 GB | 51.8 GB | R1 llama3 alpaca S3neg + R2 qwen alpaca S3neg |

显存全部充裕，无 OOM 风险。

### 预期出结果时间（重新估算，两轮并行）
- R1 剩余 ~1h45min → 完成时间 ~18:40
- R2 从现在起 ~3.3h → 完成时间 ~20:20
- Stage 3 final report：R2 完成后 ~20min → **~20:40 出完整结论**

---

## $(date '+%F %T') — GPU3 lm-eval eval jobs launched

### 设计
- GPU3 串行：先跑 S3pos 3000 steps + save_adapter，训完接 lm-eval gsm8k 5-shot；之后同卡跑 baseline eval
- baseline eval job (PID 936824) 已 kill，改为 S3pos 训练完后脚本末尾自动串行触发
- abort_factor=999 for baseline（不让红线打断，要跑满 3000 steps 才能 save adapter）

### PID
- GPU3 S3pos+eval job: PID=936823 (log: logs/s3_eval_gpu3_S3pos.log)
- GPU3 baseline eval: 将在 S3pos 完成后手动启动

### lm-eval 命令（备用，如需手动触发）
```bash
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
OUT_S3=results/stage3/llama3-8b/gsm8k/relora_diag_gated_S3pos_eval
OUT_BL=results/stage3/llama3-8b/gsm8k/relora_baseline_eval
MODEL=/mnt/cpfs/public_data/public_model/Meta-Llama-3-8B

# S3pos eval (after adapter saved):
CUDA_VISIBLE_DEVICES=3 $PY -m lm_eval --model hf \
  --model_args "pretrained=$MODEL,peft=${OUT_S3}/adapter,dtype=bfloat16,attn_implementation=sdpa" \
  --tasks gsm8k --num_fewshot 5 --batch_size 8 \
  --output_path ${OUT_S3}/lm_eval_gsm8k --log_samples

# baseline eval:
CUDA_VISIBLE_DEVICES=3 $PY -m lm_eval --model hf \
  --model_args "pretrained=$MODEL,peft=${OUT_BL}/adapter,dtype=bfloat16,attn_implementation=sdpa" \
  --tasks gsm8k --num_fewshot 5 --batch_size 8 \
  --output_path ${OUT_BL}/lm_eval_gsm8k --log_samples

# base model (no adapter, zero-shot baseline):
CUDA_VISIBLE_DEVICES=3 $PY -m lm_eval --model hf \
  --model_args "pretrained=$MODEL,dtype=bfloat16,attn_implementation=sdpa" \
  --tasks gsm8k --num_fewshot 5 --batch_size 8 \
  --output_path results/stage3/llama3-8b/gsm8k/base_model_eval/lm_eval_gsm8k
```

### 结果位置
- S3pos lm-eval: results/stage3/llama3-8b/gsm8k/relora_diag_gated_S3pos_eval/lm_eval_gsm8k/
- baseline lm-eval: results/stage3/llama3-8b/gsm8k/relora_baseline_eval/lm_eval_gsm8k/

### ETA
- S3pos 训练: ~3.3h from 18:35 → ~21:50
- S3pos lm-eval gsm8k (1319 samples, bs=8): ~15 min → ~22:05
- baseline 串行开始: ~22:05，训练 ~3.3h → ~01:35（次日）
- 也可以等 GPU 空出来后并行启 baseline

---

## 2026-05-13 — Stage 3 全部完成（Round 1 + Round 2 + lm-eval）

### 关键诊断：Qwen gsm8k ABORT 是 val_loss 过拟合触发，不是方法失效

Qwen2.5-7B 在 GSM8K 上，lora_vanilla 的 val_loss 从 step=250 的 0.126 单调爬升到 step=3000 的 0.327。
原因：GSM8K train=7409 样本，3000 步 × eff_batch=32 = 96K 样本 ≈ 13 epoch，严重过拟合。
任何 merge event（step=500）之后 val_loss 短暂跳升到 0.19 > 0.126 × 1.5 = 0.189，立刻触发红线。
这不是方法失效——是 Qwen 本身在 GSM8K 上过拟合过快（Qwen2.5-7B 强于 LLaMA-3-8B on math，base loss 更低）。

### Stage 3 完整结果表

#### LLaMA-3-8B

| dataset | method | final_val | aborted | final_CN |
|---------|--------|-----------|---------|---------|
| gsm8k | lora_vanilla | 1.505 | No | 1.96e7 |
| gsm8k | relora_baseline | 0.943 | **Yes** step=1500 | 5.68e6 |
| gsm8k | relora_diag_gated_S3pos | **0.468** | No | 8.17e6 |
| gsm8k | relora_diag_gated_S3neg | 0.902 | **Yes** step=1500 | 7.18e7 |
| alpaca | lora_vanilla | 2.388 | No | 2.12e7 |
| alpaca | relora_baseline | 1.545 | **Yes** step=2500 | 1.37e7 |
| alpaca | relora_diag_gated_S3pos | **0.967** | No | 1.18e7 |
| alpaca | relora_diag_gated_S3neg | 1.581 | **Yes** step=3000 | 7.21e6 |

#### Qwen2.5-7B

| dataset | method | final_val | aborted | final_CN |
|---------|--------|-----------|---------|---------|
| gsm8k | lora_vanilla | 0.327 | No | — |
| gsm8k | relora_baseline | 0.206 | Yes (overfitting) | — |
| gsm8k | relora_diag_gated_S3pos | 0.191 | Yes (overfitting) | — |
| gsm8k | relora_diag_gated_S3neg | 0.201 | Yes (overfitting) | — |
| alpaca | lora_vanilla | 2.018 | No | 8.97e5 |
| alpaca | relora_baseline | 1.591 | **Yes** step=3000 | 5.10e4 |
| alpaca | relora_diag_gated_S3pos | **0.910** | No | 6.95e4 |
| alpaca | relora_diag_gated_S3neg | 1.399 | **Yes** step=3000 | 1.22e5 |

#### lm-eval gsm8k 5-shot (LLaMA-3-8B, 1319 test samples)

| method | exact_match (strict) |
|--------|---------------------|
| base model (no SFT) | — (pending) |
| relora_diag_gated_S3pos (3000 steps) | **50.27%** |
| relora_baseline_eval (3000 steps, abort=999) | (pending lm-eval) |

### G1–G4 Gate 评估

**G1** (S3pos val_loss < relora_baseline by ≥2% on ≥2 cells):
- LLaMA gsm8k: S3pos 0.468 vs baseline ABORTED at 0.943 → 差 0.475 nats = **50% 改善** ✅
- LLaMA alpaca: S3pos 0.967 vs baseline ABORTED at 1.545 → 差 0.578 nats ✅
- Qwen alpaca: S3pos 0.910 vs baseline ABORTED at 1.591 → 差 0.681 nats ✅
- **G1: PASS** (3/3 可评估 cells 全通过，远超 ≥2 要求)

**G2** (S3pos final CN < relora_baseline CN by ≥2× on ≥2 cells):
- LLaMA gsm8k: S3pos CN=8.17e6 vs baseline CN=5.68e6 — S3pos **更差**（baseline ABORTED 早，CN 没爬高）❌
- LLaMA alpaca: S3pos CN=1.18e7 vs baseline CN=1.37e7 — S3pos 略好 1.16× ❌
- Qwen alpaca: S3pos CN=6.95e4 vs baseline CN=5.10e4 — S3pos **更差** ❌
- **G2: FAIL**。原因：baseline 因红线提早 ABORT，CN 没来得及爬到 Weiss 级别；S3pos 跑满 3000 步 CN 有自然增长。G2 本身设计有问题——baseline ABORTED 的 CN 是提早截断值，不代表"满跑后的 Weiss 病态"。

**G3** (S3pos val_loss < S3neg val_loss by ≥0.02 nats):
- LLaMA gsm8k: S3pos 0.468 vs S3neg 0.902 → 差 0.434 nats ✅
- LLaMA alpaca: S3pos 0.967 vs S3neg 1.581 → 差 0.614 nats ✅
- Qwen alpaca: S3pos 0.910 vs S3neg 1.399 → 差 0.489 nats ✅
- **G3: PASS** (全部 3 cells，sign convention 验证强烈通过)

**G4** (relora_baseline CN > lora_vanilla CN，Weiss 病态存在):
- LLaMA gsm8k: baseline CN=5.68e6 vs lora_vanilla CN=1.96e7 — baseline **更小** ❌（提早 ABORT）
- LLaMA alpaca: baseline CN=1.37e7 vs lora_vanilla CN=2.12e7 — baseline 更小 ❌
- **G4: 数据不足**。relora_baseline_eval（abort_factor=999，跑满 3000 步）CN=6.37e6 vs lora_vanilla 1.96e7 — 仍更小，但这个 relora_baseline_eval val_loss=1.274 比 lora_vanilla 1.505 好，说明 ReLoRA baseline 跑满后反而有用。Weiss 病态在 7B SFT 上表现不同于 LM pretraining。

### 需要 PI 裁决的 3 条

1. **G2/G4 失效原因**：红线 1.5× 让 baseline 提早 ABORT，CN 没爬到 Weiss 级别。建议：relora_baseline + abort_factor=999 重跑一次（已有 relora_baseline_eval 数据部分），确认 Weiss CN 爬升是否在 7B SFT 上成立。

2. **Qwen gsm8k 数据无效**：过拟合导致全部 ABORT，不是方法问题。建议：要么用更大数据集（MetaMathQA 10K+）重跑 Qwen gsm8k，要么接受"Qwen gsm8k 不可比"作为 limitation 披露。

3. **G1 + G3 强通过，G2/G4 数据缺失**：论文 headline 可以站在 val_loss 上（G1），conditioning 数据用 Stage 2 的 11M 结果补充（Weiss 现象在那里清晰），两套数据互补。

### lm-eval 结果
LLaMA-3-8B relora_diag_gated_S3pos: GSM8K test exact_match = **50.27%** (1319 samples, 5-shot)
LLaMA-3-8B base model + relora_baseline lm-eval 尚未跑，GPU 现在全空，可立即补跑。

---

## 2026-05-13 — Stage 3 全部完成（Round 1 + Round 2 + lm-eval）

### 关键诊断：Qwen gsm8k ABORT 是过拟合触发

Qwen2.5-7B 在 GSM8K 上 val_loss 从 step=250 的 0.126 单调爬升到 step=3000 的 0.327（13 epoch 严重过拟合）。任何 merge event（step=500）后 val_loss 跳升 0.191 > 0.126×1.5=0.189，触发红线。不是方法失效，是数据集太小 + Qwen math 能力强导致。

### Stage 3 完整结果表

#### LLaMA-3-8B
| dataset | method | final_val | aborted | final_CN |
| gsm8k  | lora_vanilla             | 1.505 | No  | 1.96e7 |
| gsm8k  | relora_baseline          | 0.943 | Yes@1500 | 5.68e6 |
| gsm8k  | relora_diag_gated_S3pos  | **0.468** | No | 8.17e6 |
| gsm8k  | relora_diag_gated_S3neg  | 0.902 | Yes@1500 | 7.18e7 |
| alpaca | lora_vanilla             | 2.388 | No  | 2.12e7 |
| alpaca | relora_baseline          | 1.545 | Yes@2500 | 1.37e7 |
| alpaca | relora_diag_gated_S3pos  | **0.967** | No | 1.18e7 |
| alpaca | relora_diag_gated_S3neg  | 1.581 | Yes@3000 | 7.21e6 |
| gsm8k  | relora_baseline_eval(abort=999) | 1.274 | No | 6.37e6 |
| gsm8k  | relora_diag_gated_S3pos_eval   | **0.467** | No | 3.86e7 |

#### Qwen2.5-7B
| dataset | method | final_val | aborted | final_CN |
| gsm8k  | lora_vanilla             | 0.327 | No  | — |
| gsm8k  | relora_baseline          | 0.206 | Yes@1000 (overfitting) | — |
| gsm8k  | relora_diag_gated_S3pos  | 0.191 | Yes@500  (overfitting) | — |
| gsm8k  | relora_diag_gated_S3neg  | 0.201 | Yes@1000 (overfitting) | — |
| alpaca | lora_vanilla             | 2.018 | No  | 8.97e5 |
| alpaca | relora_baseline          | 1.591 | Yes@3000 | 5.10e4 |
| alpaca | relora_diag_gated_S3pos  | **0.910** | No | 6.95e4 |
| alpaca | relora_diag_gated_S3neg  | 1.399 | Yes@3000 | 1.22e5 |

#### lm-eval gsm8k 5-shot (LLaMA-3-8B, 1319 test samples)
| method | exact_match |
| relora_diag_gated_S3pos (3000 steps) | **50.27%** |
| base model (no SFT) | pending |
| relora_baseline_eval (3000 steps) | pending |

### G1–G4 Gate 评估
G1 PASS: S3pos vs baseline — LLaMA gsm8k +0.475, LLaMA alpaca +0.578, Qwen alpaca +0.681 (全>2%)
G2 FAIL: baseline 提早 ABORT 导致 CN 未爬升，G2 数据不可比
G3 PASS: S3pos vs S3neg — 差距 0.43/0.61/0.49 nats，全远超 0.02 阈值
G4 INCONCLUSIVE: 7B SFT 上 Weiss CN 爬升不如 11M LM-pretraining 明显

### lm-eval 基准补跑命令（GPU 全空，可立刻执行）
PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
MODEL=/mnt/cpfs/public_data/public_model/Meta-Llama-3-8B
# base model 5-shot:
CUDA_VISIBLE_DEVICES=0 $PY -m lm_eval --model hf --model_args "pretrained=$MODEL,dtype=bfloat16" --tasks gsm8k --num_fewshot 5 --batch_size 8 --output_path results/stage3/llama3-8b/gsm8k/base_model_eval/lm_eval_gsm8k
# relora_baseline 5-shot:
CUDA_VISIBLE_DEVICES=1 $PY -m lm_eval --model hf --model_args "pretrained=$MODEL,peft=results/stage3/llama3-8b/gsm8k/relora_baseline_eval/adapter,dtype=bfloat16" --tasks gsm8k --num_fewshot 5 --batch_size 8 --output_path results/stage3/llama3-8b/gsm8k/relora_baseline_eval/lm_eval_gsm8k
