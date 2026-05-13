# 缺失实验清单 v1 — 从"hypothesis 已验证"到"投会级证据链"

> **范围**：基于 STATUS.md 已完成的 Stage 0/1/2/3，对照 `02_research_v2_baselines_theory.md` §4-5 锁定的 baseline 集和 `03_handover_for_gpu_agent.md` §3-5 的计划，识别为形成一篇有完整证据脉络的论文还需补哪些实验、baseline、ablation、可视化。
>
> **状态**：DRAFT，待 PI 决议后定版。
>
> **核心判断**：核心 hypothesis（val saliency > train saliency for predicting test-loss-change）已通过 Stage 1 验证；Stage 3 G1（val_loss）+ G3（sign convention）已强 PASS。当前缺口集中在 **(a) baseline 缺 5 个**、**(b) Stage 2 scale 不够（只 11M，没复现 Weiss 失败）**、**(c) ablation arm 缺 6 个**、**(d) 可视化 9 张应出未出**、**(e) lm-eval benchmark 缺 4 个**。

---

## 0. TL;DR — 三批次执行计划

| Batch | 目标 | 估算 GPU-h | 关键产物 |
|---|---|---|---|
| **B1（最小代价 main table）** | 在 LLaMA-3-8B × Alpaca 上把 4 个 Tier-1 baseline + 关键 ablation 跑齐，补 lm-eval 全 suite | ~120 | Stage 3 main table + fig9 (active-vs-cumulative rank) + fig11 (GSM8K bars) + fig12 (ablation grid) |
| **B2（Stage 2 完整 Weiss 复现）** | 11M/33M/66M × {full/relora/ours} × 5B tokens on C4 子集 | ~250 | fig5/fig6/fig7/fig8 + Paloma PPL 表 |
| **B3（顶会拉满）** | + Llama-3.2-3B 做 hyperparameter ablation grid + COLA/PrunedLoRA + Tulu-3/MetaMathQA + rank 扫描 | ~180 | 完整 ablation 矩阵 + cross-model generalization 证据 |

**Total ≈ 550 GPU-h**。8×A100-80G 满载并行约 **3 天**（B1），**~6 天**（B1+B2），**~10 天**（B1+B2+B3）。

---

## 1. 现状盘点（已做 vs 计划）

| 维度 | 计划（handover §2-5） | 实际完成 | Δ |
|---|---|---|---|
| Stage 1 任务数 | 3 GLUE (SST-2/MRPC/RTE) | ✓ 3 个 + K-fold Fisher 补充 | OK |
| Stage 1 saliency 变体 | S1~S5（5 种） | ✓ 5 种 | OK |
| Stage 1 决策 | GO / STOP / AMBIGUOUS | AMBIGUOUS → PI Path A | OK，FO 信号确认 |
| **Stage 2 模型规模** | 11M / 33M / 66M | **只 11M** | **缺 33M / 66M** |
| **Stage 2 训练数据** | SlimPajama-6B 或 C4 子集，**5B tokens** | wikitext-2，**~20M tokens** | **少 250×** → val_loss 无法分开 |
| **Stage 2 验证指标** | Paloma PPL（Weiss 主指标） | 自家 wikitext val_loss | **缺 Paloma** |
| Stage 2 phase | A(11M) + B(33M+66M) | 只 Phase A | **缺 Phase B** |
| **Stage 3 模型** | Llama-3.1-8B | Llama-3-8B + Qwen2.5-7B | 模型版本旧；可接受 |
| **Stage 3 baseline 数** | 7 个（LoRA/DoRA/AdaLoRA/SensitivityLoRA/CTRLoRA/ReLoRA/COLA）+ ours | 1 个 LoRA + 1 个 ReLoRA + ours(S3pos) + ours(S3neg) | **缺 5 个 baseline** |
| **Stage 3 数据** | Tulu-3 SFT mixture (140k) 或 Alpaca + GSM8K | Alpaca + GSM8K（独立） | 量级 OK；Qwen-GSM8K 13 epoch 过拟合无效 |
| **Stage 3 评估 benchmark** | GSM8K / MMLU / BBH / IFEval | 仅 GSM8K 5-shot × 1 method | **缺 3 benchmark + 缺 3 method 的 GSM8K eval** |

---

## 2. 缺失 baseline（按优先级 + 官方实现来源）

### 2.1 Tier-1：必须有，否则 main table 不成立

| Baseline | 在论文中的作用 | 官方实现 | 实现工作量 |
|---|---|---|---|
| **DoRA** | PEFT 当前最强 single-LoRA；任何 LoRA 工作必比 | `peft.LoraConfig(use_dora=True)` 内置 | < 1h |
| **AdaLoRA** | "动态 rank + train sensitivity" 经典 | `peft.AdaLoraConfig` 内置 | < 1h |
| **Sensitivity-LoRA** ★★★★★ | **直接占"二阶 + LoRA"格点**，v2 §1.1 标 kill-or-seal | arXiv:2509.09119；**官方代码暂未发布** → 自实现：train-Fisher 对角作 sensitivity → matrix-level rank budget 再分配 | ~200 LOC，1 天 |
| **CTR-LoRA** ★★★★☆ | 直接占"曲率 + trust region + LoRA"格点 | arXiv:2510.15962；**官方代码状态待查** → 最低限度跑 ablation arm：trust-region 正则项 $\lambda\sum_\ell \|A_\ell B_\ell^\top\|_M^2$ 加到 lora_vanilla | ~300 LOC，1-2 天 |
| **COLA (Chain of LoRA)** | merge-restart 的 Frank-Wolfe 视角，**本方法的理论框架来源**，不比对会被审稿人质疑 | arXiv:2401.04151；GitHub 上 Wenhan-Tan/COLA（待 verify 可用性） | 改 `stage3_run.py` 加 method=cola：不重置 optimizer，旧 LoRA 冻结叠加新 LoRA。~100 LOC |

**不做 Tier-1 的后果**：审稿人 standard objection — "你只对比 LoRA 和 ReLoRA，无法证明改善不是来自任意一个 PEFT trick。"

### 2.2 Tier-2：强故事支撑

| Baseline | 角色 | 来源 |
|---|---|---|
| **PrunedLoRA** | 间接竞争 + 免费 theoretical lemma（gradient pruning > activation pruning） | arXiv:2510.00192 |
| **Random-drop（同剪枝率）** | 证明"诊断信号有效"而不是"剪枝本身有效" | 自实现，10 LOC（saliency 函数返 `torch.rand`） |
| **Train-saliency-gated ReLoRA** | **核心 ablation**：直接证明 val > train 卖点 | 改 `stage3_run.py` 一个 `--gate_source train|val` flag |

### 2.3 Tier-3：appendix-only

- L1RA (arXiv:2509.04884)
- Flexi-LoRA (ICML 2025)
- Adaptive LoRA Merge (Miyano & Arase, ACL-F 2025)
- StelLA (NeurIPS 2025)

### 2.4 不做的（v2 已明确）

- **EPI** (arXiv:2604.14010)：concurrent work，2026-04，距本工作 ≤ 1 个月，handover §9 rule 9 显式禁止当 baseline。
- 任何 train-loss-only 的 PEFT 方法（不在 white-space 同一格）。

---

## 3. 缺失 ablation arm（审稿人查的清单）

### 3.1 必做（决定论文是否站得住）

| Ablation | 验证什么 | 实现方式 |
|---|---|---|
| **A1. Random drop vs diagnostic drop（同剪枝率）** | 信号有效 vs 剪枝本身有效 | saliency 改 `torch.rand`，drop_rate 强制等同 S3pos 的实际值 |
| **A2. Train-saliency vs val-saliency（pipeline 内同 setup）** | 卖点核心 | `--gate_source train` 对照 `--gate_source val` 全 stage3 重跑 |
| **A3. Signed vs unsigned gating** | "filter harmful directions" 必要性 | `|S3|` 阈值排序 vs signed S3 阈值 |
| **A4. FO 一阶 vs Fisher 二阶 gate** | 验证 Path A 决议（FO 主、Fisher ablation） | 两个 method arm 跑同 setup |

### 3.2 强烈建议（hyperparameter sensitivity）

| Ablation | 范围 | 算力 |
|---|---|---|
| **A5. Merge frequency 扫描** | $T \in \{500, 1000, 2000, 4000\}$ | 4× 1 method × 1 dataset，可在 3B 上做 |
| **A6. Drop ratio 扫描** | 固定 25% / 50% / 75% drop_rate | 3× |
| **A7. LoRA rank 扫描** | $r \in \{8, 16, 32, 64\}$ | 4×，**输出 fig9 narrative 核心图** |
| **A8. Saliency batch 数扫描** | $n_{\text{batches}} \in \{4, 8, 16, 32\}$ | 4×，证明信号 robust 到估计噪声 |

### 3.3 Stage 1 升级 ablation

| Ablation | 内容 |
|---|---|
| **A9. Stage 1 扩到大数据集** | +QNLI(105k) 或 +MNLI(393k)，验证"SST-2 噪声"假说 = 大数据 + mild overfit 下 oracle 信号弱（vs LoRA 失效） |
| **A10. Stage 1 重跑于 SFT setting** | 在 LLaMA-3-8B GSM8K 单 checkpoint 做 oracle ablation，确认 val saliency 优势在 7B scale 也成立（Stage 1 当前只在 RoBERTa-base 上做） |

---

## 4. 缺失可视化清单

`plots/` 当前只有 `plots/stage1/` 4 张。**Stage 2/Stage 3 几乎全空**。`plot_from_json.py` 已就绪，按 PI §4.4 硬约束每图先写 json 再渲染。

| 编号 | 图名 | 内容 | 数据现状 | 优先级 |
|---|---|---|---|---|
| fig1 | correlation_grid | 3×5 scatter, x=Δtest y=saliency | ✓ Stage 1 已有 | done |
| fig2 | rho_over_time | ρ vs step, faceted by task | ✓ | done |
| fig3 | train_vs_val_paired | **Stage 1 headline** | ✓ | done |
| fig4 | harmful_auc | AUC bars per task per saliency | ✓ | done |
| **fig5** | **effective_rank_curves** | x=tokens, y=ER, 3 lines × 3 model sizes | 11M jsonl 有，**33M/66M 缺数据** | **P0** |
| **fig6** | **condition_number_curves** | log10(CN) | 同上 | **P0** |
| **fig7** | **paloma_perplexity** | Weiss 主指标，跨 model size | **Paloma 完全没跑** | **P0** |
| **fig8** | **saliency_dist_at_merges** | per-merge violin + drop_rate 柱 | Stage 2 11M 有 jsonl，**没出图** | **P0** |
| **fig9** | **active_vs_cumulative_rank** | active rank = r 固定线；cumulative rank(Δ_stable) 单调上升 | **完全没数据**（cumulative rank 没保存） | **P0 narrative 核心** |
| **fig10** | **stage3_main_table_heatmap** | 7 baseline × 2 model × 5 benchmark 热力图 | 缺 5 baseline + 4 benchmark | **P0** |
| **fig11** | **lm_eval_gsm8k_bars** | 5-shot GSM8K 4 method × 2 model | 只 1 个 method | **P0** |
| **fig12** | **ablation_grid** | random/train-sal/val-sal/val-sal-signed 4 行对比 | **没数据** | **P0** |
| fig13 | hyperparam_heatmap | merge_every × drop_ratio | 没 | P1 |
| fig14 | rank_scaling_curve | r ∈ {8,16,32,64} val_loss & ER | 没 | P1 |
| fig15 | cross_model_generalization | bar chart over Llama/Qwen/Mistral | 缺 Mistral | P2 |

---

## 5. Model × Dataset 推荐矩阵

### 5.1 Stage 1（信号验证）

| 当前 | 补充 |
|---|---|
| SST-2 (67k) / MRPC (3.7k) / RTE (2.5k) | **+QNLI (105k)** 或 **+MNLI (393k)**：验证 SST-2 噪声不是单纯数据大小问题 |
| RoBERTa-base | **+ Llama-3-8B 单 checkpoint oracle**：scale-up 验证 |

### 5.2 Stage 2（Weiss 复现）

**必须升级到三档**：

| Size | 当前 | 目标 |
|---|---|---|
| 11M | ✓ wikitext-2 20M tok / 5000 steps | **改 C4 子集 1B tok / 50000 steps** |
| 33M | **未做** | C4 子集 1B tok / 30000 steps |
| 66M | **未做** | C4 子集 1B tok / 20000 steps |

每档 3 method：full_rank / relora_baseline / relora_diag_gated_S3pos。共 9 runs。

### 5.3 Stage 3（SFT scale-up）

**模型轴**：

| 当前 | 推荐补充 | 角色 |
|---|---|---|
| Llama-3-8B | **Llama-3.2-3B** | 便宜模型，做 hyperparameter ablation grid 的主战场 |
| Qwen2.5-7B | （可选）Mistral-7B-v0.3 | 跨 family generalization 第 3 个 anchor |

**数据轴**：

| 当前 | 推荐补充 | 理由 |
|---|---|---|
| Alpaca-cleaned (52k) | **Tulu-3 SFT mixture (140k)** | Alpaca 已过时（2023），Tulu-3 是 2024 标准 SFT 数据集 |
| GSM8K (7.4k train) | **MetaMathQA-10k subset** | 替换 Qwen-GSM8K 过拟合无效问题；10× 数据 |

**评估 benchmark**（lm-eval-harness 一条命令搞定）：

| Benchmark | 当前 | 必加 | 角色 |
|---|---|---|---|
| GSM8K 5-shot | 1 method only | **全 method** | 数学推理 |
| **MMLU 5-shot** | 没 | **必加** | 通识 |
| **BBH 3-shot** | 没 | **必加** | 复杂推理 |
| **IFEval** | 没 | **必加** | 指令跟随 |
| HumanEval | 没 | 可选 | 代码（如选 Tulu-3 含 code） |
| AlpacaEval 2 | 没 | 可选 | 生成质量（需要 GPT-4 judge） |

### 5.4 最小集 vs 推荐集 vs 顶会集

| 版本 | Stage 1 | Stage 2 | Stage 3 model | Stage 3 data | Stage 3 eval | 估算 GPU-h |
|---|---|---|---|---|---|---|
| **最小（通信级）** | 现有 | 现有 11M | Llama-3-8B | Alpaca + GSM8K | GSM8K + MMLU + IFEval | ~120 |
| **推荐（顶会下限）** | + 1 大数据 | + 33M | + Qwen2.5-7B | + Tulu-3 | + BBH | ~400 |
| **顶会拉满** | + Llama-8B oracle | + 66M, 5B tok, Paloma | + Llama-3.2-3B + Mistral | + MetaMathQA + Tulu-3 | + HumanEval + AlpacaEval2 | ~700 |

---

## 6. 必须保存的"证据"清单（reviewer 复现需要）

| 证据 | 当前状态 | 整改 |
|---|---|---|
| 每 step train_loss / val_loss | ✓ jsonl | 保持 |
| 每 merge event ER / CN | Stage 2 11M ✓；Stage 3 sample 8 layers | **Stage 3 改全层 SVD 或 sample 16 层** |
| **per-component saliency 分布** | Stage 1/2 ✓；**Stage 3 没保存** | 加 `saliency_at_merge.jsonl`（layer, comp, S_train, S_val, decision） |
| **每 merge 的 dropped components 索引** | Stage 2 只有 kept count | 改 `dropped_components.jsonl`（layer, comp, score, decision, threshold） |
| **每 method 的 final adapter** | 只 S3pos 存了 | 全 method save_adapter，便于 lm-eval 与外部复用 |
| **lm-eval 完整 output_path** | 1 个 | 完整 method × model × benchmark 矩阵 |
| **cumulative rank trajectory** rank(Δ_stable) | **没保存** | 每 merge 后 SVD merged update，存 jsonl |
| **wall-clock + peak_mem 表** | STATUS.md 散点 | 抽 `results/cost_table.csv` |
| **完整 config yaml** | Stage 1 有 | 补 Stage 2/3 完整 yaml（含 seed/abort_factor/method 全字段） |
| **reproduce_all.sh** | 没 | 一键复跑脚本 |

---

## 7. 三批次执行计划（细节）

### 7.1 Batch 1：Stage 3 main table 补全（~120 GPU-h，~3 天）

**目标**：在 LLaMA-3-8B × Alpaca 上把 Tier-1 baseline + 关键 ablation 跑齐，出 main table。

**工作**：

| # | 任务 | GPU-h | 依赖 |
|---|---|---|---|
| 1 | 加 DoRA / AdaLoRA 到 `stage3_run.py` | <1（写代码） | — |
| 2 | 实现 Sensitivity-LoRA / CTR-LoRA（自复现） | <1（写代码） | — |
| 3 | 实现 COLA arm（freeze 旧 LoRA + 叠新 LoRA） | <1 | — |
| 4 | 实现 random-drop / train-saliency-gated 两 ablation | <1 | — |
| 5 | 重跑 Llama-3-8B × Alpaca × 9 methods (4 Tier-1 + 3 ablation + lora_vanilla + S3pos) | 30 (8 GPU 并行约 4h) | 1-4 |
| 6 | 重跑 Llama-3-8B × GSM8K × 同 9 methods | 30 | 5 |
| 7 | lm-eval all 9 methods × 4 benchmark (GSM8K/MMLU/IFEval/BBH) on Llama-3-8B | 20 | 5,6 |
| 8 | 重跑 Qwen2.5-7B × Alpaca × 9 methods（保留过拟合 cell 当 limitation） | 30 | 5 |
| 9 | lm-eval Qwen × 4 benchmark | 10 | 8 |
| 10 | 出图 fig9 / fig10 / fig11 / fig12 | <1 | 5-9 |

**产物**：
- `results/stage3_v2/<model>/<dataset>/<method>/{adapter, train_loss.jsonl, val_loss.jsonl, ER.jsonl, CN.jsonl, saliency_at_merge.jsonl, dropped_components.jsonl, summary.json}`
- `results/stage3_v2/summary/main_table.csv`（7 baseline + ours + ablation × 2 model × 2 dataset × 4 benchmark）
- `plots/stage3/fig9_active_vs_cumulative_rank.png`
- `plots/stage3/fig10_main_table_heatmap.png`
- `plots/stage3/fig11_lm_eval_gsm8k_bars.png`
- `plots/stage3/fig12_ablation_grid.png`

### 7.2 Batch 2：Stage 2 Weiss 复现升级（~250 GPU-h，~5 天）

**目标**：让 fig5/6/7/8 真正可用，复现 Weiss 失败模式 + 修复。

**工作**：

| # | 任务 | GPU-h |
|---|---|---|
| 1 | 切 wikitext-2 → C4 子集 (5B tok 子采样) | <1（数据准备） |
| 2 | 加 Paloma eval（HuggingFace `Paloma` benchmark） | <1 |
| 3 | 加 cumulative rank tracking（SVD merged Δ_stable） | <1 |
| 4 | 11M × 5B tok × {full, relora, ours} on 3 GPU | 30 |
| 5 | 33M × 5B tok × 3 methods on 3 GPU | 90 |
| 6 | 66M × 5B tok × 3 methods on 3 GPU | 130 |
| 7 | 出 fig5 / fig6 / fig7 / fig8 + fig9 数据补 11M/33M/66M | <1 |

**产物**：
- `results/stage2_v2/{11M,33M,66M}/<method>/...`（含 Paloma PPL）
- `plots/stage2/fig5_effective_rank_curves.png`
- `plots/stage2/fig6_condition_number_curves.png`
- `plots/stage2/fig7_paloma_perplexity.png`
- `plots/stage2/fig8_saliency_dist_at_merges.png`

### 7.3 Batch 3：顶会拉满（~180 GPU-h，~4 天，可选）

| # | 任务 | GPU-h |
|---|---|---|
| 1 | + Llama-3.2-3B（便宜模型）做 hyperparameter ablation grid | 60 |
| 2 | A5 merge_every 扫描 ($T \in \{500,1000,2000,4000\}$) | 30 |
| 3 | A6 drop_ratio 扫描 (25/50/75%) | 25 |
| 4 | A7 rank 扫描 ($r \in \{8,16,32,64\}$) | 40 |
| 5 | + Tulu-3 SFT 主表重跑 1 model × 4 method | 30 |
| 6 | + MetaMathQA-10k 替 Qwen-GSM8K | 15 |
| 7 | （可选）+ Mistral-7B-v0.3 验证 cross-family | 30 |
| 8 | A9/A10 Stage 1 升级（QNLI + Llama oracle 单 ckpt） | 20 |
| 9 | 出 fig13/fig14/fig15 | <1 |

---

## 8. PI 决议（已锁定 2026-05-14）

**Q1 — Sensitivity-LoRA / CTR-LoRA 复现策略**：
- ✅ **决议 (a)**：严格自实现，main table 标 "our re-implementation"
- 实现差异（lr / target_modules / batch / seed）写进论文 appendix
- 与原文报告数字偏差 > 5% 时，在 footnote 说明并 cite 原文作 reference

**Q2 — Stage 2 scale 升级（B2）**：
- ✅ **决议：做**
- 11M / 33M / 66M × 3 method × 5B tokens on C4 子集
- 加 Paloma PPL 评估
- 加 cumulative rank tracking（SVD merged Δ_stable）

**Q3 — 评估 benchmark**：
- ✅ **决议**：B1 必做 5 个 — **GSM8K / MMLU / IFEval / BBH / HumanEval**
- 顺序：GSM8K（数学）→ MMLU（通识）→ IFEval（指令）→ BBH（推理）→ HumanEval（代码）
- lm-eval-harness 单 commit hash 锁定
- 注：**2026 视角建议升级版** 见 §12（MMLU-Pro / MATH-500 / HumanEval+）

**Q4 — Qwen GSM8K 处理**：
- ✅ **决议 (a)**：换 **MetaMathQA-10k** 重跑（10× 数据，避免 13 epoch 过拟合）
- 旧 GSM8K cell 作 limitation 在 appendix 披露
- MetaMathQA 训练 cell 进 main table

**Q5 — SFT 数据**：
- ✅ **决议：Tulu-3 SFT mixture (140k) 替换 Alpaca**
- B1 起就用 Tulu-3，Alpaca 不再保留
- 单 epoch on 140k 在 8B 上 ~6-8h/method
- 注意：Tulu-3 含 code 数据 → HumanEval 评测对位

---

## 9. 风险与红线

继承自 handover §9：

1. **不**改 espo 环境（pip install 一律 user-local）
2. **不**修改 test_holdout / GLUE val 用于 saliency
3. **不**把 EPI 当 baseline（concurrent work）
4. abort_factor 红线：post-merge val_loss > 1.5× first_eval → ABORTED.flag（B1/B2/B3 全继承）
5. STATUS.md append-only，所有运行 PID + GPU 记录在册
6. seed=42 全程，每 run 保存 config.yaml

新增风险：

7. **Tier-1 baseline 复现 vs 原 paper 数字偏差**：自实现的 Sensitivity-LoRA / CTR-LoRA 若与原文报告偏差 > 5%，需要在论文里说明实现差异（lr / target_modules / seed），并 cite 原文为指导而非 SOTA。
8. **lm-eval-harness 版本锁定**：所有 benchmark 用同一 commit hash 跑，避免分数不可比。
9. **Adapter checkpoint 体积**：B1 全 method save_adapter，Llama-3-8B r=16 单 adapter ~80MB，9 method × 2 model × 2 dataset = 36 adapter ≈ 3GB，磁盘 OK。

---

## 10. 验收标准（Definition of Done）

按 Batch 验收：

**B1 完成 = 同时满足**：
- [ ] Tier-1 5 baseline 全跑通（DoRA / AdaLoRA / Sensitivity / CTR / COLA）
- [ ] 4 ablation arm 跑通（random-drop / train-gate / val-gate / val-signed-gate）
- [ ] lm-eval 4 benchmark × 9 method × 2 model 全完成
- [ ] fig9 / fig10 / fig11 / fig12 出图
- [ ] `results/stage3_v2/summary/main_table.csv` 生成
- [ ] STATUS.md 写完 B1 总结 entry

**B2 完成 = 同时满足**：
- [ ] 11M / 33M / 66M × 3 method × 5B tokens 全跑完
- [ ] Paloma PPL 评估完成
- [ ] cumulative rank 数据保存
- [ ] fig5 / fig6 / fig7 / fig8 出图
- [ ] STATUS.md 写完 B2 总结 entry

**B3 完成 = 同时满足**：
- [ ] A5-A10 ablation 全跑完
- [ ] Llama-3.2-3B + MetaMathQA + Tulu-3 数据齐
- [ ] fig13 / fig14 / fig15 出图
- [ ] reproduce_all.sh 可一键复跑全套
- [ ] STATUS.md 最终 entry

**论文级整体 DoD**：
- [ ] main table（fig10）信号一致：ours 在 ≥ 4/6 cells 上击败最强 baseline ≥ 1.0 average point
- [ ] fig3（Stage 1 headline）+ fig5/fig9（Stage 2/3 narrative）+ fig12（ablation）三组图自洽
- [ ] sign convention task-specific 现象在新数据上得到一致解释（信号-噪声框架）
- [ ] 所有 G1/G3 在 B1 完成后仍然 PASS

---

## 11. 文档关联

- 总章程：`03_handover_for_gpu_agent.md` §3-5
- baseline 锁定：`02_research_v2_baselines_theory.md` §4
- 当前 STATUS：`STATUS.md`（append-only）
- Path A 决议：`04_stage2_directive_path_A.md`
- PI 反馈：`05_pi_response_AB_parallel.md`

本文档是 v6 / v7 之后的 v8 级补丁，**修改本文档时同步在 STATUS.md 加 append entry**。

— end of missing-experiments list v1 —

---

## 12. 2026 视角调整（DRAFT 2026-05-14）

> 项目立项时间窗看起来是 2025 末或 2026 初；当前是 **2026 年 5 月**。从今天的 model / benchmark / data landscape 重审，发现 **3 处需要升级，2 处可以加新故事，1 处必须重新检索 concurrent work**。

### 12.1 模型 landscape 已经迭代两次（必须升级）

| 类别 | 项目当前用 | 2026-05 状态 | 建议 |
|---|---|---|---|
| Llama 7-8B | Llama-3-8B (2024-04) | **Llama-3.1-8B / 3.3-8B-Instruct** 是当前标准；Llama-4 (2025-04) 是 MoE 109B+，不适合 LoRA 论文 | **切到 Llama-3.1-8B-Instruct**，保留 Llama-3-8B 作对照（已有数据） |
| Qwen 7B | Qwen2.5-7B (2024-09) | **Qwen3-8B** (2025-04) 已是主流，原生 thinking mode | **加 Qwen3-8B** 作第二 anchor；Qwen2.5-7B 保留为兼容 baseline |
| 小模型 | （未选）Llama-3.2-3B | **Gemma-3-4B** / **Qwen3-1.7B** / **Phi-4-mini-3.8B** 2025-Q1 后均强 | B3 选 **Qwen3-1.7B** 或 **Gemma-3-4B**——两者都比 Llama-3.2-3B 新 1 代 |
| **Reasoning 模型** | （未选） | **DeepSeek-R1-Distill-Qwen-7B** (2025-01) 后已成"小推理模型"标准 | **强烈建议加为第 3 个 anchor**（见 §12.4） |

**核心建议**：B1 main anchor 从 `Llama-3-8B + Qwen2.5-7B` → **`Llama-3.1-8B-Instruct + Qwen3-8B`**；B3 加 **`DeepSeek-R1-Distill-Qwen-7B`** 做 reasoning 故事。

### 12.2 Benchmark 大半已饱和（必须升级 + 加新）

| 原 benchmark | 2026-05 状态 | 替换/补充 |
|---|---|---|
| **MMLU** 5-shot | 7B+ 模型普遍 60-75%，已饱和 | **MMLU-Pro** 5-shot（更难，10 选 1，CoT）<br>—— 保留 MMLU 作 reference，主表用 MMLU-Pro |
| **GSM8K** 5-shot | 7B 模型 50-85%，partially saturated | + **MATH-500** 0-shot（竞赛题）<br>+ **AIME-2024** 0-shot（限 R1-distill anchor） |
| **HumanEval** | 已饱和（70%+ common） | **HumanEval+** + **MBPP+**（EvalPlus 2024）<br>顶会推荐 **LiveCodeBench**（contamination-free）|
| **BBH** | 仍有效 | 保留 |
| **IFEval** | 仍是 IF 黄金标准 | 保留 |
| **(新)** | — | + **GPQA-Diamond** 0-shot（graduate-level 科学，难，未饱和） |
| **(新 for reasoning anchor)** | — | + **MUSR**（multi-step reasoning） |

**Q3 升级版决议**（建议）：
- B1 必跑：**MMLU-Pro / GSM8K / IFEval / BBH / HumanEval+**（5 个）
- B3 加跑：**MATH-500 / GPQA-Diamond**（2 个）
- Reasoning anchor 专属：**AIME-2024 / MUSR**

### 12.3 SFT 数据集状态（Tulu-3 仍 SOTA，但加 1 个 2025 数据）

| 数据 | 当前 | 2026 视角 |
|---|---|---|
| Tulu-3 SFT mixture (140k) | ✓ Q5 已选 | **保留**，2024-Q4 至今仍是学术 SFT 标准 |
| MetaMathQA | ✓ Q4 已选 | 保留；2026 替代是 **OpenMathInstruct-2 (NVIDIA 2024-Q3)** 14M 合成样本，取 100k 子集 → **B3 可选**升级 |
| **(新)** | — | **OpenThoughts-114k** (2025-02，社区蒸馏 R1 CoT) → **R1-Distill anchor 专用**（§12.4） |
| **(新)** | — | **Llama-Nemotron-Post-Training-Dataset** (NVIDIA 2025-Q1, 30M) → too big, B3 appendix only |

**建议**：B1 用 Tulu-3 + MetaMathQA-10k；B3 加 OpenThoughts-114k（如果走 R1 anchor）。

### 12.4 新故事线（强烈建议加）：Reasoning-Model LoRA SFT

**为什么 2026 必须有这条线**：
- 2025-01 DeepSeek-R1 发布后，**"小推理模型 + LoRA 继续训"** 成了 PEFT 论文标配
- 你的方法天然适配——reasoning model 的 SFT 容易破坏 reasoning 能力（catastrophic forgetting CoT pattern），**val-saliency gating 应该能"过滤掉破坏 reasoning 的方向"**
- 与 ours sign convention（drop S3 > 0）的"剪 harmful"框架天然契合

**最小新实验**：
- 模型：**DeepSeek-R1-Distill-Qwen-7B**
- 数据：**OpenThoughts-114k**（保 CoT 长度）+ **MetaMathQA-10k**（短答案）混合
- 评测：**AIME-2024** + **MATH-500** + **MUSR**（reasoning bench triad）
- **新指标**：**reasoning trace length retention**——SFT 前后平均 CoT 长度比例（< 0.8 说明破坏了 reasoning），ours vs 其它 method 比较
- 配 main table 一行 + 单独 1 张 fig17

**估算**：~80 GPU-h（B3 范围）。

### 12.5 Concurrent work 重新检索（必做）

handover §9 rule 9 锁的 EPI (arXiv:2604.14010, 2026-04) 距今 1 个月，仍是 boundary。但需要在投稿前重新跑一次：

| 检索方向 | 目标 |
|---|---|
| arXiv 2025-09 至 2026-05 | "diagnostic LoRA"、"val-gated PEFT"、"saliency-based LoRA pruning" |
| Google Scholar 6 个月 cite Sensitivity-LoRA / CTR-LoRA / ReLoRA | 看是否有人已用 val signal 改它们 |
| OpenReview NeurIPS 2026 / ICLR 2026 已 desk-rejected 但 public 的稿 | 避免撞同思路 |
| paper-with-code "PEFT 2026 leaderboard" | 排名前 5 有无未列基线 |

**红线触发**：若发现 2025-09 至 2026-04 间有论文用 **held-out / diagnostic val + LoRA component pruning** 三件套，必须：
- (a) 立刻在 STATUS.md 标 concurrent work
- (b) 调整 framing（"first to combine X + Y" 改成 "first to combine X + Y + Z"）
- (c) 跑对比实验（如代码可用）

### 12.6 评测基础设施升级（2026 标准）

| 项 | 2024 做法 | 2026 标准 |
|---|---|---|
| 推理引擎 | HuggingFace transformers `generate` | **vLLM** 0.6+（10× 快） |
| Eval harness | lm-eval-harness 任意 | **lm-eval-harness pinned commit**（在 `requirements.txt` 锁 commit hash） |
| 数学题判定 | 字符串 match | **Math-Verify**（HuggingFace 2025）或 **MathRuler** sympy-based |
| Code 评测 | EvalPlus | EvalPlus + **LiveCodeBench**（contamination-free） |
| AlpacaEval | GPT-4 judge | **Arena-Hard-v2** + **AlpacaEval-2-LC**（length-controlled，2024-Q3 后标准） |

**操作**：B1 写 `scripts/eval_lm_harness.sh` 时即用 vLLM backend + pinned harness commit，写入 STATUS.md。

### 12.7 投稿目标与时间窗（2026 视角）

| 会议 | 提交窗 | 适合度 |
|---|---|---|
| **EMNLP 2026** | 通常 6 月 | 时间紧；只够 B1 出 main table，B2/B3 进 rebuttal |
| **NeurIPS 2026** | 已过（5 月） | 不可达 |
| **ICLR 2027** | 9-10 月（2026） | **最佳目标**：B1+B2+B3 完整数据，3-4 个月 polish |
| **ICML 2027** | 1-2 月（2027） | 备用 |
| **TMLR** | 滚动 | 始终可投，PEFT 类工作常去 |

**建议主投 ICLR 2027**，备 EMNLP 2026 short paper（只 main table + Stage 1 信号验证）。

### 12.8 12.x 调整后的最终模型 × 数据 × benchmark 矩阵

| 阶段 | 模型 | 数据 | Benchmark |
|---|---|---|---|
| **B1 (核心)** | Llama-3.1-8B-Instruct + Qwen3-8B | Tulu-3 SFT (140k) + MetaMathQA-10k | MMLU-Pro / GSM8K / IFEval / BBH / HumanEval+ |
| **B2 (Weiss 复现)** | 11M / 33M / 66M LLaMA-style | C4 5B tokens | Paloma PPL + 自家 val PPL |
| **B3 (拉满)** | + Qwen3-1.7B (小模型 ablation grid) + **DeepSeek-R1-Distill-Qwen-7B** (reasoning anchor) | + OpenThoughts-114k + OpenMathInstruct-2 (可选) + Tulu-3 (cross) | + MATH-500 + GPQA-Diamond + AIME-2024 + MUSR + reasoning_trace_length |

### 12.9 2026 调整对 GPU-h 预算的影响

| Batch | 原估算 | 2026 调整后 | 增量来源 |
|---|---|---|---|
| B1 | ~120 GPU-h | **~150 GPU-h** | +Qwen3-8B vs Qwen2.5-7B 略快；+MMLU-Pro/HumanEval+/BBH 评测 (+30h) |
| B2 | ~250 GPU-h | ~250 GPU-h | 不变 |
| B3 | ~180 GPU-h | **~280 GPU-h** | + R1-Distill anchor SFT (~80h) + AIME/MUSR/GPQA eval (~20h) |
| **Total** | ~550 | **~680** | +130 GPU-h（≈ 8 GPU × 16h，1 个晚上） |

代价可接受。**强烈建议采纳 §12.1 / §12.2 / §12.4 三处升级**。

### 12.10 一句话总结

**2026 投稿要点**：模型升一代（Llama-3.1 / Qwen3）、benchmark 反饱和（MMLU-Pro / HumanEval+ / MATH-500）、加 reasoning 故事线（R1-Distill anchor + AIME），投 ICLR 2027。

— end of 2026 perspective addendum —
