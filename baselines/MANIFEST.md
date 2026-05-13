# Baselines Manifest

**目的**：固定 baseline 实现来源 + commit hash，保证后续所有 ablation 可复现。

**两阶段策略**：
- **Phase B1（当下执行）**：只跑有官方代码的 baseline（4 个 cloned），降低实现风险
- **Phase B1.5（rebuttal / camera-ready 前）**：自复现 4 个无官方代码的 baseline（COLA / Sensitivity-LoRA / CTR-LoRA / PrunedLoRA）

---

## Phase B1 — 已克隆官方代码（4 个）

| # | 目录 | 论文 | arxiv | 会议 | Repo | Commit | License | 角色 |
|---|---|---|---|---|---|---|---|---|
| 1 | `DoRA_official/` | DoRA: Weight-Decomposed Low-Rank Adaptation | 2402.09353 | ICML 2024 (Oral) | https://github.com/NVlabs/DoRA | `7e2f10ab` | NVIDIA Research License | **Tier-1 baseline**（main table） |
| 2 | `AdaLoRA_official/` | AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | 2303.10512 | ICLR 2023 | https://github.com/QingruZhang/AdaLoRA | `d10f5ebe` | MIT | **Tier-1 baseline**（main table） |
| 3 | `ReLoRA_official/` | ReLoRA: High-Rank Training Through Low-Rank Updates | 2307.05695 | ICLR 2024 | https://github.com/Guitaricet/relora | `176f3763` | Apache-2.0 | **Tier-1 baseline**（已用，作锚） |
| 4 | `LoRAPrune_reference/` | LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning | 2305.18403 | ACL 2024 Findings | https://github.com/aim-uofa/LoRAPrune | `4da52721` | Apache-2.0 | **Tier-2 reference**（PrunedLoRA 自复现的参照） |

### 推荐使用方式

- **DoRA**：直接走 PEFT 内置（`peft.LoraConfig(use_dora=True)`），与 NVlabs 实现等价，省一份代码维护。`DoRA_official/` 保留作 sanity-check 数字对照。
- **AdaLoRA**：直接走 PEFT 内置（`peft.AdaLoraConfig`），AdaLoRA_official 保留作 hyperparameter 来源。
- **ReLoRA**：项目当前 `stage3_run.py` 已实现 ReLoRA-style merge，参考官方实现确认 optimizer reset 协议。
- **LoRAPrune**：作 Phase B1.5 复现 PrunedLoRA 的参考（结构类似：用 LoRA 权重 + 梯度做 importance estimation）。

---

## Phase B1.5 — 仅 PDF，自复现（4 个，DEFERRED）

| # | 目录 | 论文 | arxiv | 状态 | 角色 |
|---|---|---|---|---|---|
| 5 | `COLA_reimpl/` | Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning | 2401.04151 | ICML 2024，**无官方代码** | Tier-1 baseline，**自复现** |
| 6 | `Sensitivity_LoRA_reimpl/` | Sensitivity-LoRA: Low-Load Sensitivity-Based Fine-Tuning for Large Language Models | 2509.09119 | 2025-09 arxiv only | **Tier-1 must-beat**，自复现（用 train-Fisher 替代 ours 的 val-FO 信号即可） |
| 7 | `CTR_LoRA_reimpl/` | CTR-LoRA: Curvature-Aware and Trust-Region Guided Low-Rank Adaptation | 2510.15962 | 2025-10 arxiv only | Tier-1 baseline，自复现（在 lora_vanilla 加 trust-region 正则） |
| 8 | `PrunedLoRA_reimpl/` | PrunedLoRA: Robust Gradient-Based Structured Pruning for Low-rank Adaptation | 2510.00192 | ICLR 2026 anonymous submission，**无代码** | Tier-2 baseline，自复现（参考 `LoRAPrune_reference/`） |

每个 reimpl 目录已下载 `PAPER.pdf`。Phase B1.5 启动时先生成 `IMPLEMENTATION_NOTES.md`（算法 box + 超参表 + plug-in 点），再写代码。

---

## 与 PEFT library 的对应

| Method | PEFT 内置 | 官方源码 | 推荐使用 |
|---|---|---|---|
| LoRA | ✅ `LoraConfig` | microsoft/LoRA | PEFT |
| DoRA | ✅ `LoraConfig(use_dora=True)` | NVlabs/DoRA | PEFT（与官方等价） |
| AdaLoRA | ✅ `AdaLoraConfig` | QingruZhang/AdaLoRA | PEFT |
| ReLoRA | ❌ 项目自写 | Guitaricet/relora | 项目内 `stage3_run.py` |

PEFT 版本锁定：`peft>=0.14.0`（含 DoRA + AdaLoRA + bf16 修复），放进 Phase B1 的 `requirements_b1.txt`。

---

## 命名与挂载约定

- 官方代码：`<Name>_official/`
- 参考实现（非直接 baseline）：`<Name>_reference/`
- 自复现：`<Name>_reimpl/`

所有目录的 `.git` 不进当前 repo（已 add 到 `.gitignore`，避免 submodule 嵌套）。如需 reviewer 复现，提供本 manifest 中的 commit hash + 上游 URL。

---

## 后续维护

- 每加一个新 baseline，**先更新本文件，再写代码**。
- 任何 commit hash 升级要 STATUS.md append entry。
- License 不兼容（NVIDIA Research License = 仅研究用途）的实现：仅作 reference，**不**把代码拷进我们的 module，仅 cite + 提供 commit hash。

— end of manifest —
