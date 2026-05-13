# Stage 1 → Stage 2 决策回复（PI → Cloud Agent）

> **READ FIRST.** 本文件是 `04_stage2_directive_path_A.md` 的修订与补充。冲突处以本文件为准。所有 silently 做出的决定必须先在 STATUS.md 登记后再行动。
>
> **Owner**: PI. **Status**: APPROVED. **Date**: 2026-05-12.

---

## 0. TL;DR

- **Path A 为主、Path B 并行**。7 张可用 A100-80G 全部跑满：A 占 4 卡（GPU 0/1/3/4），B 占 3 卡（GPU 5/6/7）。
- **GPU 2 是别人的，绝对不要碰**（snapshot 已确认）。
- **Stage 2 启动前先解决一个致命决断点：sign convention**——见 §1。10 分钟以内可定结论。
- Path A 中**两个 gate 方向同时跑**（`S3 > 0` drop vs `S3 < 0` drop），赢的就是真符号；这一步顺便把方法学的 sub-contribution 拿到了。
- **所有图必须先 dump 一份 JSON 元数据**，再用 `scripts/plot_from_json.py` 从 JSON 出图。详见 §4。这条是硬约束，方便后期美化与论文复现。
- aggregator 里 silently 做的另外两个决定（`cond2` AND→OR、`cond1` 用 |ρ| 而非 signed ρ）以及 `harmful_rate` 的噪声归因，必须按 §5 各自回报一次。

---

## 1. 致命决断点：sign convention 必须先钉死

### 1.1 问题

`scripts/stage1_aggregate.py` L270-275 注释承认：

> "original handover says 'auc(-s_signed)' predicts harmful, but the aggregator originally used naive positive direction. We compute the symmetric AUC = max(auc(+), auc(-)) per (task, ckpt) ..."

这等于把"`S3_fo_val_signed` 的哪个方向预测 harmful"这个问题用 `max(+, −)` 给抹掉了。Stage 2 的 gate predicate 完全依赖这个方向；如果反了，gate 会把有益 component 当 harmful 删掉、把 harmful component 当 helpful 合并——Stage 2 一定崩。

具体冲突链（已校对源码）：

| 来源 | 公式 | 推论 |
|---|---|---|
| `tests/test_saliency.py:99` 推导 | `s_handover = −<grad_A, A> = −per_comp`；shrink helpful 让 L 升 | `s_handover < 0` ⟺ harmful |
| `src/saliency.py:94` 存盘 | `S3_fo_val_signed = +<grad_A, A> = +per_comp` | 存盘符号与 `s_handover` 相反 |
| `scripts/stage1_run.py:137` | `auc_signed = _auc(-s_signed, labels)` | 与 handover 一致 |
| `stage1_aggregate.py` | `auc_sym = max(auc(+), auc(-))` | **抹掉方向** |

### 1.2 你必须先回报的 10 行（最高优先级，≤ 10min）

把以下命令的输出贴到 STATUS.md 最新条目下面，**然后等 PI 看完再启动 Stage 2**：

```bash
cd /mnt/cpfs/junlongke/onlinelora/lora_obd
echo "## sign_check $(date '+%F %T')" >> STATUS.md
for f in $(ls results/stage1/*/*/auc_signed.json | sort); do
    echo "$f : $(cat $f)" >> STATUS.md
done
```

解读规则（PI 这边会判，但你也可以同步判断）：

- 多数 `S3_fo_val_signed_neg_auc_harmful` > 0.5（比如 ≥ 0.6）→ **handover 方向正确**，gate predicate = "drop if `S3_fo_val_signed < 0`"
- 多数 < 0.5（比如 ≤ 0.4）→ **handover 方向反**，gate predicate = "drop if `S3_fo_val_signed > 0`"
- 跨任务方向不一致（比如 SST-2 偏 < 0.5，MRPC > 0.5）→ **信号 task-specific**，停下，等 PI 回复，不要自作主张做 per-task gate

### 1.3 保险层：Phase A 同时跑两个 gate 方向

无论 §1.2 结论是什么，Phase A 都**两个方向都跑**——这把"哪个方向才对"做成了一个经验论据，论文里直接当 sub-contribution 写。

```python
# scripts/stage2_run.py 里加一个 sign 参数
p.add_argument("--gate_sign", choices=["S3pos_drops", "S3neg_drops"],
               default="S3pos_drops",
               help="S3pos_drops: drop if S3_fo_val_signed > 0; "
                    "S3neg_drops: drop if S3_fo_val_signed < 0.")
```

```python
def keep_component_for_merge(s_i: float, gate_signal: str, gate_sign: str) -> bool:
    if gate_signal == "none":
        return True
    if gate_signal == "S5_fisher_val":
        return s_i > fisher_layer_threshold
    # S3_fo_val_signed
    if gate_sign == "S3pos_drops":
        return s_i < 0.0   # drop if S3 >= 0
    elif gate_sign == "S3neg_drops":
        return s_i > 0.0   # drop if S3 <= 0
    raise ValueError(gate_sign)
```

---

## 2. Path A 详细做法（4 卡：GPU 0/1/3/4）

### 2.1 资源分配

| GPU | Method | gate_signal | gate_sign | 目的 |
|---|---|---|---|---|
| 0 | `full_rank` | none | n/a | oracle ceiling（不做低秩，不做 ReLoRA） |
| 1 | `relora_baseline` | none | n/a | replicate Weiss 失败（effective rank 单调降） |
| 3 | `relora_diag_gated` | S3_fo_val_signed | **S3pos_drops** | 假设 1：drop if S3 > 0 |
| 4 | `relora_diag_gated` | S3_fo_val_signed | **S3neg_drops** | 假设 2：drop if S3 < 0 |

胜者准则（Path A GO/NO-GO）：
- `relora_diag_gated` 的赢家在 val loss 上比 `relora_baseline` 好 ≥ **5% relative** → Path A GO，gate_sign 钉死。
- 两个方向都赢 → 怀疑实验有 bug 或 baseline 太弱，停下报。
- 两个方向都输 → Path A 失败，但 Path B（K-fold Fisher）可能还能救。
- 一个赢一个输 → 完美剧本，赢的那个就是真方向。

### 2.2 启动模板

```bash
export PY=/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
export HF_HOME=/mnt/cpfs/junlongke/hf_cache
cd /mnt/cpfs/junlongke/onlinelora/lora_obd
mkdir -p logs results/stage2/11M plots/stage2 results/stage2/plots/json

# Pre-flight
nvidia-smi --query-gpu=index,memory.free --format=csv  # 确认 GPU 0/1/3/4 都空闲

# Path A: 11M × 4 jobs on GPU 0/1/3/4
CUDA_VISIBLE_DEVICES=0 $PY scripts/stage2_run.py --size 11M --method full_rank                                                > logs/s2A_11M_full.log     2>&1 &
echo $! >> .stage2_pids
CUDA_VISIBLE_DEVICES=1 $PY scripts/stage2_run.py --size 11M --method relora_baseline                                          > logs/s2A_11M_relo.log     2>&1 &
echo $! >> .stage2_pids
CUDA_VISIBLE_DEVICES=3 $PY scripts/stage2_run.py --size 11M --method relora_diag_gated --gate_signal S3_fo_val_signed --gate_sign S3pos_drops > logs/s2A_11M_S3pos.log 2>&1 &
echo $! >> .stage2_pids
CUDA_VISIBLE_DEVICES=4 $PY scripts/stage2_run.py --size 11M --method relora_diag_gated --gate_signal S3_fo_val_signed --gate_sign S3neg_drops > logs/s2A_11M_S3neg.log 2>&1 &
echo $! >> .stage2_pids
```

不 `wait` —— Path A 和 Path B 同步起。

### 2.3 Fisher ablation arm 怎么办

`relora_diag_gated_fisher` （Fisher gate）暂时**不在 Phase A 跑**。原因：

1. Path A 已经占 4 卡，要留 3 卡给 Path B 做 Fisher 的根本性修复。
2. Phase A 一旦确定了赢家方向，输的那个方向的 GPU 立刻释放，那张卡用来跑 Fisher ablation arm。
3. 如果 Path B 的 K-fold 把 Fisher 救回来了（mean Δ|ρ|_fisher ≥ 0.10），那 Fisher gate 也用 K-fold 版本来跑，更扎实。

---

## 3. Path B 详细做法（3 卡：GPU 5/6/7）

### 3.1 目的

把 Stage 1 的 Fisher 部分用 K-fold 重跑一遍。当前 Fisher mean Δ|ρ| = 0.046（差 0.054 才达标 0.10）；猜测是 256 样本的方差太大。如果 5-fold 平均把方差压下来后 Fisher 也能过门，方法学故事就完整了——"val 二阶视角"也能用，不只是 first-order 视角。

### 3.2 资源分配

3 张卡 × 3 个 GLUE 任务 = 完美 1:1 并行。

| GPU | Task | K | val_samples 每 fold |
|---|---|---|---|
| 5 | sst2 | 5 | ~2700 |
| 6 | mrpc | 5 | ~150 |
| 7 | rte  | 5 | ~50 |

### 3.3 实现

不要写新的 trainer——复用 `stage1_run.py` 的 saliency 计算路径，加一个 `--k_fold` 参数。具体改动：

```python
# scripts/stage1_run.py 顶部 args 加：
p.add_argument("--k_fold", type=int, default=1,
               help="K-fold cross-validation for S5_fisher_val. Each fold averages "
                    "fisher saliency over its hold-out chunk and writes a separate "
                    "S5_fisher_val_kfold field. K=1 reproduces Stage 1 behavior.")
```

```python
# saliency 计算位置：原来一次性算 S5_fisher_val。改成：
if args.k_fold > 1:
    fold_size = len(diag_loader.dataset) // args.k_fold
    fisher_kfold_per_layer = defaultdict(list)
    for fold in range(args.k_fold):
        fold_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
        fold_subset = Subset(diag_loader.dataset, fold_indices)
        fold_loader = DataLoader(fold_subset, batch_size=diag_loader.batch_size)
        s5_fold = fisher_saliency(model, fold_loader, signed=False, max_samples=args.fisher_max_samples)
        for layer, vec in s5_fold.items():
            fisher_kfold_per_layer[layer].append(vec)
    # average across folds
    s5_kfold = {layer: torch.stack(vecs).mean(dim=0) for layer, vecs in fisher_kfold_per_layer.items()}
    # 写一份 S5_fisher_val_kfold 到 components.jsonl
```

输出位置：`results/stage1_kfold/<task>/<step>/components.jsonl`（与 Stage 1 隔离）。

### 3.4 启动模板

```bash
# Path B: K-fold Fisher rerun on GPU 5/6/7
CUDA_VISIBLE_DEVICES=5 $PY scripts/stage1_run.py --config configs/stage1_sst2.yaml --k_fold 5 --out_dir results/stage1_kfold/sst2 --fisher_max_samples 512 > logs/s1B_kfold_sst2.log 2>&1 &
echo $! >> .stage2_pids
CUDA_VISIBLE_DEVICES=6 $PY scripts/stage1_run.py --config configs/stage1_mrpc.yaml --k_fold 5 --out_dir results/stage1_kfold/mrpc --fisher_max_samples 512 > logs/s1B_kfold_mrpc.log 2>&1 &
echo $! >> .stage2_pids
CUDA_VISIBLE_DEVICES=7 $PY scripts/stage1_run.py --config configs/stage1_rte.yaml  --k_fold 5 --out_dir results/stage1_kfold/rte  --fisher_max_samples 512 > logs/s1B_kfold_rte.log  2>&1 &
echo $! >> .stage2_pids
```

跑完后调用 `stage1_aggregate.py --kfold_dir results/stage1_kfold/summary`（你可能需要加个 `--input_root` 参数复用 aggregator），看新 mean Δ|ρ|_fisher 是否 ≥ 0.10：

- 是 → Fisher gate 进 Stage 2 作为第三 method arm，论文叙事改成"val 视角的 saliency 整体（FO + Fisher）都比 train 强"。
- 否 → 接受现实，把贡献叙事定到 "val 一阶（FO）显著、val 二阶（Fisher）噪声受限"。

### 3.5 Path B 决策门

5-fold Fisher 跑完，写一行结论到 STATUS：

```
## kfold_fisher_check <date>
mean Δ|ρ|_fisher (5-fold) = <X> (CI95 <a>..<b>) — was 0.046 in K=1.
verdict: <PASS / FAIL / MARGINAL>
```

---

## 4. 绘图改造：JSON 元数据 + 二级 plot 脚本（硬约束）

### 4.1 为什么

我们现在的图是 matplotlib 直出 PNG。论文阶段要换风格、加 annotation、改字号、出 vector PDF——重跑实验代价太高。所以**所有数值必须先固化成 JSON，PNG 只是 JSON 的当前渲染**。

### 4.2 强制规则

1. **`scripts/stage{1,2}_plot.py` 每画一张 figN，必须先 dump `results/stage{1,2}/plots/json/figN_data.json`。**
2. **必须提供 `scripts/plot_from_json.py`，命令行接 JSON 出 PNG**：
   ```bash
   $PY scripts/plot_from_json.py results/stage2/plots/json/fig5_effective_rank_curves.json \
       --out plots/stage2/fig5_effective_rank_curves.png \
       --style publication      # 或 default
   ```
3. **图与 JSON 必须 1:1 对应**。如果 fig 改了，JSON 必须先改。

### 4.3 JSON schema（每张图都按这个写）

```json
{
  "figure_id": "fig5_effective_rank_curves",
  "title": "Effective Rank during ReLoRA Training (11M)",
  "schema_version": 1,
  "generated_at": "2026-05-12T11:30:00",
  "source_paths": [
    "results/stage2/11M/full_rank/effective_rank.jsonl",
    "results/stage2/11M/relora_baseline/effective_rank.jsonl",
    "results/stage2/11M/relora_diag_gated_S3pos/effective_rank.jsonl",
    "results/stage2/11M/relora_diag_gated_S3neg/effective_rank.jsonl"
  ],
  "axes": {
    "x": {"label": "training step", "unit": "step", "scale": "linear"},
    "y": {"label": "effective rank (Roy–Vetterli)", "unit": "rank", "scale": "linear"}
  },
  "series": [
    {
      "id": "full_rank",
      "label": "full_rank (oracle ceiling)",
      "color_hint": "#444444",
      "linestyle_hint": "--",
      "x": [0, 1000, 2000, ...],
      "y": [8.0, 8.0, 8.0, ...],
      "y_ci_low": [...],          // 可选，bootstrap CI
      "y_ci_high": [...],
      "merge_event_marks": []     // 可选，垂直虚线位置
    },
    {
      "id": "relora_baseline",
      "label": "relora_baseline (vanilla)",
      "color_hint": "#cc3333",
      "linestyle_hint": "-",
      "x": [...],
      "y": [...],
      "merge_event_marks": [5000, 10000, 15000]
    },
    {
      "id": "relora_diag_gated_S3pos",
      "label": "relora_diag_gated (drop if S3>0)",
      "color_hint": "#3366cc",
      "linestyle_hint": "-",
      "x": [...],
      "y": [...]
    },
    {
      "id": "relora_diag_gated_S3neg",
      "label": "relora_diag_gated (drop if S3<0)",
      "color_hint": "#33aa33",
      "linestyle_hint": "-",
      "x": [...],
      "y": [...]
    }
  ],
  "annotations": [
    {"type": "vline", "x": 5000, "label": "merge event 1"},
    {"type": "hline", "y": 8.0,  "label": "full rank ceiling"}
  ],
  "notes": "rendered from stage2/11M ; gate sign sub-experiment"
}
```

### 4.4 `plot_from_json.py` 的最小行为

```python
"""scripts/plot_from_json.py

Read a fig*_data.json (schema_version=1) and render to PNG/PDF.
Usage: python plot_from_json.py path/to/figN_data.json --out path/to/figN.png [--style {default,publication}]
"""
import argparse, json, matplotlib.pyplot as plt

def render(data, out, style="default"):
    if style == "publication":
        plt.rcParams.update({
            "font.family": "serif", "font.size": 10,
            "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        })
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in data["series"]:
        ax.plot(s["x"], s["y"], label=s["label"],
                color=s.get("color_hint"), linestyle=s.get("linestyle_hint", "-"))
        if s.get("y_ci_low"):
            ax.fill_between(s["x"], s["y_ci_low"], s["y_ci_high"], alpha=0.18,
                            color=s.get("color_hint"))
    ax.set_xlabel(data["axes"]["x"]["label"])
    ax.set_ylabel(data["axes"]["y"]["label"])
    ax.set_title(data["title"])
    for a in data.get("annotations", []):
        if a["type"] == "vline": ax.axvline(a["x"], color="gray", linestyle=":", linewidth=0.7)
        elif a["type"] == "hline": ax.axhline(a["y"], color="gray", linestyle=":", linewidth=0.7)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="default", choices=["default", "publication"])
    args = ap.parse_args()
    with open(args.data_json) as f:
        data = json.load(f)
    render(data, args.out, args.style)
```

### 4.5 既有 Stage 1 的 4 张图也要回填 JSON

`plots/stage1/fig{1,2,3,4}_*.png` 在跑 Stage 2 之前先回填一遍 JSON。如果你已经丢了底数据，从 `results/stage1/{task}/{step}/components.jsonl` 重算一次。这个 backlog 不阻塞 Stage 2，但**不要拖到 Stage 2 写报告**。

---

## 5. 其他 silently 做出的决定，必须正面回应

每一条独立写进 STATUS.md。这些不阻塞 Stage 2，但下次跑 aggregator 时必须有人决定到底用哪个版本。

### 5.1 `cond2`：sign test AND vs OR

**问题**：`stage1_aggregate.py:279`

```python
cond2 = (n_pos_fi >= 10) or (n_pos_fo >= 10)
```

handover §3.8 表述为 "Positive on ≥ 10 of 15 (task, step) pairs"，**没明确**是 Fisher 还是 FO 还是两者都。云 agent 选 OR，更宽松。这次没影响（两者都 ≥ 10），但下次会有事。

**要做**：在 STATUS.md 写一条决策记录：
```
## 2026-05-12 — aggregator decision: cond2 = OR
Reason: handover §3.8 ambiguous; OR chosen to allow signal-source-specific PASS.
Per-task numbers this run: fisher 10/15, fo 11/15 — both pass either logic.
Going forward: keep OR but report both numbers in every aggregate.
```

### 5.2 `cond1`：用 |ρ| 而非 signed ρ

**问题**：`stage1_aggregate.py:175-190` 用 `|rho(val)| - |rho(train)|`。如果 train 和 val 的 ρ 都很大但符号相反，signed delta_rho 会很小甚至负；|ρ| 形式记成正向收益。

**要做**：复算一次 signed 版本，把对比写进 STATUS：

```bash
cd /mnt/cpfs/junlongke/onlinelora/lora_obd
$PY -c "
import json, glob, numpy as np
deltas_fo_abs, deltas_fo_signed = [], []
deltas_fi_abs, deltas_fi_signed = [], []
for cf in sorted(glob.glob('results/stage1/*/*/correlations.json')):
    c = json.load(open(cf))
    # 你的 correlations.json 里到底是 |rho| 还是 signed rho？请你确认字段命名
    # 然后补这里
    pass
print('check correlations.json field names first')
"
```

如果 signed mean Δρ_fo 仍 ≥ 0.05 且仍 11/15 正，方法学叙事不动；如果不是，论文 headline 数字得改。

### 5.3 `harmful_rate` 65-86% 是过拟合还是噪声？

**问题**：晚期 ckpt 有 65-86% 的 component 被标 harmful。云 agent 解读为"过拟合 regime 验证"。**另一种可能**：晚期 baseline test loss 已接近 noise floor，oracle ablation 的 `delta_test` 在噪声里随机正负，`harmful_flag` 退化为 ~50/50 标签噪声但被恰好偏 negative 的 noise 漂移成 65-86%。

**要做**：

```bash
$PY -c "
import json, glob, numpy as np
for tsk in ['sst2', 'mrpc', 'rte']:
    base = f'results/stage1/{tsk}'
    # baseline test loss 时间序列
    tl = [json.loads(l) for l in open(f'{base}/test_loss.jsonl')]
    # 每个 ckpt 的 delta_test 分布
    for stepdir in sorted(glob.glob(f'{base}/*/components.jsonl')):
        deltas = [json.loads(l)['delta_test'] for l in open(stepdir)]
        med = float(np.median(np.abs(deltas)))
        std_te = float(np.std([x['loss'] for x in tl[-50:]]))   # 取末段 std
        print(f'{stepdir}: median|delta_test|={med:.5f}, recent_test_loss_std={std_te:.5f}, ratio={med/max(std_te, 1e-9):.2f}')
"
```

若 `median|delta_test| / recent_test_loss_std < 1`，证据指向噪声主导，AUC 数字虚高。把这条结论写进 STATUS。

---

## 6. 报告 & 暂停规则

- **Phase A 启动后 30 min 内**：贴 §1.2 的 raw AUC 数据 + 4 个 jobs 的 `merge_event=1` 日志 → STATUS。
- **每个 merge event（每 5000 步）后**：贴 4 个 jobs 的 val_loss、effective_rank、drop_rate → STATUS。每个 job 一行。
- **任何一个 job 崩了**：不要 auto-relaunch，写 STATUS，等 PI 看。
- **§1.2 发现符号 task-specific**：暂停所有 Stage 2，等 PI 回复。
- **§3.5 K-fold Fisher 跑完**：贴决策行到 STATUS。

---

## 7. 红线（不变）

复述自 `03_handover` §9 + `04_directive` §6，外加本文件新增项：

1. 不在 diagnostic / test_holdout 上训练。
2. 不 auto-skip stage。
3. 不默认 merge stable updates。
4. 永远 log seed = 42。
5. 永远保存 per-checkpoint state_dict。
6. 永远报 effect size + bootstrap CI。
7. 不 silently 改 saliency 公式。
8. 不用官方 GLUE val 做 saliency / pruning（仍是 test_holdout，密封）。
9. EPI (arXiv:2604.14010) 是 concurrent，**不作 baseline**，思想可借鉴。
10. 不 silently 切换默认 gate signal/sign。任何 default 修改先写 STATUS 等 PI 确认。
11. **（新）** 不动 `espo` 环境；用 `/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python`。
12. **（新）** 不碰 GPU 2（别人在用）。
13. **（新）** 所有图必须先 dump JSON 再渲染 PNG（§4）。
14. **（新）** STATUS.md append-only；不覆盖、不重排。任何 silently 决定必须先在 STATUS 登记后再行动。

---

## 8. 文件与位置速查

| 你需要的东西 | 位置 |
|---|---|
| Stage 1 完整报告 | `lora_obd/results/stage1/report.md` |
| Stage 1 决策 JSON | `lora_obd/results/stage1/summary/decision.json` |
| Stage 1 4 张图 | `lora_obd/plots/stage1/fig{1,2,3,4}_*.png` |
| 15 个 ckpt 原始数据 | `lora_obd/results/stage1/{task}/{step}/components.jsonl` |
| 本文件 | `lora_obd_package/05_pi_response_AB_parallel.md`（云端落到 `/mnt/cpfs/junlongke/onlinelora/`） |
| 上一份指令 | `04_stage2_directive_path_A.md`（被本文件 §1.3 / §2 / §3 部分覆盖） |
| 总 handover | `03_handover_for_gpu_agent.md` §4 |

---

**完。请按 §1 → §5 → §2 + §3 并行 → §6 报告节奏推进。任何 ambiguity 写 STATUS 等 PI。**
