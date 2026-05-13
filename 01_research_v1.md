# OBD/诊断驱动 LoRA 剪枝、合并与 Rank 回收：调研备忘录

## 0. 结论先行

这个想法**有研究意义**，但需要把创新点从“改 LoRA 结构”明确转成：

> 在训练过程中，用独立 diagnostic set 周期性评估 LoRA rank component / adapter / expert 的泛化贡献；剪掉低贡献或有害更新，合并高贡献更新，并把释放出的 rank 重新用于学习新方向，从而在固定活跃 rank 预算下实现动态容量增长。

相对现有 LoRA / AdaLoRA / DyLoRA / SoRA / MoE-LoRA / LoRA merging 方法，潜在差异是：

1. **目标不同**：不是只做 rank 分配、结构路由或静态合并，而是做“训练中的泛化诊断”。
2. **信号不同**：不是只依赖训练 loss、参数范数、门控稀疏或路由频率，而是显式使用 held-out diagnostic loss / gradient / curvature。
3. **机制不同**：不是一次性剪枝或训练后合并，而是周期性 `diagnose → prune → merge → recycle rank`。
4. **容量解释不同**：单次 LoRA rank 不变，但多轮有效更新可被合并进 stable adapter/base weight，使累计更新 rank 理论上超过活跃 rank。

建议把方法暂命名为：

- **DVR-LoRA**: Diagnostic Validation-guided Rank-Recycling LoRA
- **OBD-LoRA**: Optimal-Brain-Damage-guided LoRA Recycling
- **ReLoRA-Diag**: Diagnosed Rank-Recycling LoRA

其中最稳妥的论文定位是 **diagnostic validation-guided rank recycling for LoRA**，OBD 是理论/打分工具，而不是唯一核心。

---

## 1. 你的方法可以形式化成什么

给定冻结基座权重 \(W_0\)，LoRA 更新为：

\[
W = W_0 + \Delta W,\quad \Delta W = BA
\]

若 rank 为 \(r\)，可以拆成 rank component：

\[
\Delta W = \sum_{i=1}^{r} b_i a_i^\top
\]

训练时维护两类更新：

\[
W = W_0 + \Delta W_{\text{stable}} + \Delta W_{\text{active}}
\]

- \(\Delta W_{\text{active}}\)：当前可训练 LoRA，rank 固定为 \(r\)。
- \(\Delta W_{\text{stable}}\)：通过诊断后保留下来的历史有效更新，可以是 merged LoRA bank，也可以直接 merge 到 base。

周期性执行：

```text
train active LoRA
every T steps:
    compute diagnostic contribution score for each component/expert/adapter
    prune bottom p% or below-threshold components
    merge high-confidence components into stable adapter/base
    orthogonalize/rotate remaining active directions
    reinitialize freed rank slots
```

关键是：**active rank 一直是 r，但 stable 更新不断吸收有效方向**。如果每轮吸收一个 rank-\(r'\) 更新，累计更新：

\[
\Delta W_{\text{total}} = \sum_{t=1}^{K} \Delta W_t
\]

其 rank 上界可达：

\[
\text{rank}(\Delta W_{\text{total}}) \leq \sum_{t=1}^{K} \text{rank}(\Delta W_t)
\]

因此它不是靠单次“旋转”突破 rank 限制，而是靠 **merge + rank recycling** 突破活跃 rank 预算限制。

---

## 2. 理论支持

### 2.1 OBD / Taylor pruning 支持“删掉低 saliency 参数”

Optimal Brain Damage 的基本思想是用二阶泰勒近似估计删除参数的 loss 增量：

\[
\Delta L \approx \frac{1}{2} h_i \theta_i^2
\]

其中 \(h_i\) 是 Hessian 对角项。其假设包括：

- 当前点接近局部极小，梯度项较小；
- loss 在局部近似二次；
- 忽略 Hessian 非对角项。

迁移到 LoRA 后，不一定要对单个 scalar 参数打分，可以对 rank component 打分：

\[
\Delta W_i = b_i a_i^\top
\]

删除第 \(i\) 个 component 的影响：

\[
s_i \approx L_{\text{diag}}(W) - L_{\text{diag}}(W - \Delta W_i)
\]

或者用 Taylor / Fisher 近似：

\[
\Delta L_i \approx -g^\top \delta_i + \frac{1}{2}\delta_i^\top H\delta_i
\]

这里 \(\delta_i\) 表示移除该 LoRA component 对参数造成的变化。

可行近似：

- **直接消融**：最可信但贵，逐个临时置零 component，看 diagnostic loss 变化。
- **一阶 Taylor**：\(|g^\top \theta|\) 或 \(-g^\top \delta\)。
- **Fisher / 梯度平方**：用 \(\mathbb{E}[g^2]\) 近似 Hessian 对角。
- **Hutchinson Hessian diagonal**：更强二阶近似，但实现复杂度更高。

理论上合理，因为神经网络剪枝领域长期使用 magnitude、gradient、Taylor、Fisher、Hessian saliency 估计参数/结构重要性。你的方法的新意是把它放到 **LoRA rank component 的在线泛化诊断** 上。

### 2.2 Held-out diagnostic set 支持“合理更新”的泛化判断

训练 loss 下降并不代表更新合理，尤其小数据微调、指令微调、RLHF/SFT 场景容易过拟合或学到 spurious direction。用 diagnostic set 判断 component 是否有泛化贡献，本质上是：

- validation-guided model selection；
- validation-guided pruning；
- bilevel / meta-objective 的简化形式。

但要注意：如果频繁用真正 validation set 做训练决策，会 validation leakage。建议论文中明确三分：

```text
train set: 更新 active LoRA
diagnostic set: 周期性诊断、剪枝、merge
test / final validation: 只做最终报告
```

文中最好不要把 diagnostic set 直接叫 validation set。

### 2.3 旋转 / 正交化的理论位置

LoRA 分解不唯一：

\[
BA = (BQ)(Q^{-1}A)
\]

因此只旋转 \(A,B\) 不改变 \(\Delta W\)，也不改变 rank。它的作用不是直接增加表达 rank，而是：

1. **让 component 更可分解**：通过 SVD 或近似 SVD，把更新变到主奇异方向，便于 component-level pruning。
2. **降低冗余**：正交化 active directions，减少多个 rank 学同一方向。
3. **辅助 rank recycling**：merge 后清空/重启 rank slot，学习新方向。

所以文中应避免说“旋转本身突破 rank”，而应说：

> rotation/orthogonalization makes rank components more disentangled and pruneable; merge-and-reinitialize accumulates multiple low-rank updates and thereby surpasses the active-rank bottleneck over time.

---

## 3. 需要比较的基线

下面按“必须比较 / 强相关 / 可选”分组。

### 3.1 必须比较的核心基线

| 类别 | 基线 | 为什么必须比 | 需要控制 |
|---|---|---|---|
| Full fine-tuning | Full FT | 上界或强参考 | 如果资源允许，小模型上做 |
| 标准 PEFT | LoRA | 最基本 baseline | 同 trainable parameter budget 或同 active rank |
| 量化 LoRA | QLoRA | LLM 微调常用强 baseline | 同 base model、同数据、同 rank |
| 动态 rank | AdaLoRA | 直接竞争：按重要性动态分配 rank | 同总参数预算 |
| 动态 rank | DyLoRA | 直接竞争：一次训练支持多 rank | 比 rank robustness / rank budget |
| 稀疏 rank | SoRA | 直接竞争：通过 gate 学动态 rank | 同初始 rank、同最终稀疏度 |
| LoRA 结构增强 | DoRA | 近年强 LoRA 变体 | 同参数量或报告额外参数 |

#### LoRA

论文：**LoRA: Low-Rank Adaptation of Large Language Models**  
核心：冻结基座，在 Transformer 线性层上插入低秩更新 \(BA\)，显著减少训练参数和显存，没有额外推理延迟。

你的方法相对 LoRA 的比较点：

- 同 active rank 下是否更好；
- 同最终参数量下是否更好；
- 训练中是否能识别并删除有害 rank component；
- merge/recycle 后是否提高低 rank LoRA 的容量。

#### AdaLoRA

论文：**Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**  
核心：用 SVD-like 参数化和重要性打分，在不同权重矩阵之间动态分配 rank budget，剪掉不重要 singular values。

和你的方法最接近，必须重点比较。差异：

- AdaLoRA 主要解决 **rank budget allocation across matrices**；
- 你的方法解决 **online diagnostic contribution of updates**；
- AdaLoRA 主要基于训练过程的重要性估计；
- 你的方法引入 diagnostic set 上的泛化贡献；
- AdaLoRA 剪掉后通常不是“merge good directions + recycle freed rank repeatedly”的机制。

#### DyLoRA

论文：**DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation**  
核心：训练一个能在多个 rank 下工作的 LoRA，避免为不同 rank 重新搜索/训练。

差异：

- DyLoRA 强调 deployment-time rank flexibility；
- 你的方法强调 training-time diagnosis and capacity recycling；
- DyLoRA 不主动判断某个 rank direction 对 diagnostic set 是否有害。

#### SoRA

论文：**Sparse Low-rank Adaptation of Pre-trained Language Models**  
核心：在 LoRA rank 上加 gate，用 proximal gradient 形成稀疏 rank，动态调整有效 rank。

差异：

- SoRA 是 regularization/gate-induced sparsity；
- 你的方法是 held-out diagnostic contribution-induced pruning；
- SoRA 可以作为强对照：如果你的 diagnostic pruning 比纯稀疏门控更稳，说明诊断信号有价值。

#### DoRA

论文：**DoRA: Weight-Decomposed Low-Rank Adaptation**  
核心：把权重更新分解为 magnitude 和 direction，LoRA 负责 direction，提升学习能力和稳定性。

DoRA 不是动态 rank 方法，但经常是强 PEFT baseline。建议至少在主实验或附录比较。

---

### 3.2 与 MoE-LoRA / 多专家相关的基线

如果你的方法扩展到 “MoE LoRA / 旧有 LoRA bank / expert pool”，需要比较：

| 类别 | 基线 | 相关点 |
|---|---|---|
| LoRA experts | MoELoRA | LoRA as MoE，router + contrastive specialization |
| LoRA experts | MoLE / Mixture of LoRA Experts | 多 LoRA 融合与分支选择 |
| LoRA experts | MixLoRA | LoRA-based MoE、多任务场景 |
| MoE model PEFT | DR-LoRA | MoE 模型中按 expert saliency 动态扩 rank |
| Layer-wise expert allocation | MoLA / MoE-LoRA variants | 层级专家分配 |

#### MoELoRA

论文：**MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models**  
核心：把多个 LoRA 当专家，通过 router 组合，并用 contrastive learning 促使专家差异化。

和你的差异：

- MoELoRA 侧重结构路由和专家多样性；
- 你的方法侧重专家/adapter 在 diagnostic set 上的贡献判断；
- 可扩展为 “diagnostic-gated MoE-LoRA”：周期性删除低贡献 expert，merge 高贡献 expert。

#### MoLE / Mixture of LoRA Experts

论文：**Mixture of LoRA Experts**  
核心：用层级控制和更灵活的分支选择组合多个 LoRA，优于直接 arithmetic merging。

你的方法可比较：

- 固定 expert mixture vs diagnostic pruning/merge；
- router-only selection vs held-out contribution selection；
- inference-time mixture vs training-time recycle。

#### DR-LoRA

论文：**DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models**  
核心：针对 MoE 模型，不同 expert 的 LoRA rank 不应统一；用 routing frequency + rank importance 等 saliency 动态增长 expert ranks。

如果你做 MoE 模型，这是非常直接的 baseline。差异：

- DR-LoRA 的 saliency 主要来自 expert 路由和 rank 重要性；
- 你的方法可用 diagnostic loss / OBD saliency 判断 expert update 是否真的泛化；
- DR-LoRA 是 grow ranks，你的方法是 prune/merge/recycle ranks。

---

### 3.3 与 LoRA 剪枝 / 稀疏化相关的基线

| 基线 | 核心 | 和你的关系 |
|---|---|---|
| LoRAPrune | 用 LoRA 权重和梯度估计结构重要性，剪基座结构 | 不是直接剪 LoRA rank，但 saliency 思路相关 |
| PrunedLoRA | gradient-based structured pruning for LoRA | 直接相关，若可复现应加入 |
| Magnitude pruning | 按 \(\|\Delta W_i\|\)、\(\|a_i\|\|b_i\|\) 删 | 简单消融 baseline |
| Random pruning | 随机删同等比例 rank | 证明诊断不是靠正则/随机重启 |
| Train-loss pruning | 用训练集 loss/gradient 打分 | 证明 diagnostic set 的必要性 |

#### LoRAPrune

论文：**LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning**  
核心：用 LoRA weights and gradients 做重要性估计，进行结构化剪枝，避免直接使用基座权重梯度带来的大显存开销。

差异：

- LoRAPrune 主要目标是压缩 LLM 结构；
- 你的方法主要目标是提升/稳定 LoRA 微调的泛化与容量；
- 但其 “LoRA weights + gradients can guide pruning” 可以作为理论旁证。

---

### 3.4 与 LoRA 合并 / 旧 LoRA 兼容相关的基线

如果你强调“兼容旧有 LoRA”，需要比较 adapter composition / merging：

| 基线 | 核心 | 相关实验 |
|---|---|---|
| Linear / average merge | 直接加权平均 LoRA | 最简单 merge baseline |
| Task arithmetic | 合并 task vectors | 多任务/技能组合 |
| TIES merging | 解决符号冲突，合并任务向量 | 多 LoRA 冲突处理 |
| DARE | 随机丢弃+重标定 delta | 合并时去冗余 |
| LoraHub | few-shot gradient-free 动态组合多个 LoRA | 旧 LoRA bank 选择 |
| LoRA Soups / CAT | concatenation + optimal weighting | 技能组合强 baseline |

#### LoraHub

论文：**LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition**  
核心：给定多个已训练 LoRA，用少量新任务样本无梯度地优化组合权重，实现 cross-task generalization。

差异：

- LoraHub 是 inference/adaptation-time composition；
- 你的方法是 training-time diagnose, prune, merge, recycle；
- 但如果你维护 old LoRA bank，则 LoraHub 是必须比较的强 baseline。

#### LoRA Soups

论文：**LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks**  
核心：多个单技能 LoRA 通过 concatenation/weighted composition 做技能组合，报告 CAT 优于多种 model/data merging。

你的方法可比较：

- 对旧 LoRA 的静态最优组合 vs 周期性 diagnostic selection；
- 是否能删除负迁移 LoRA；
- 是否能在训练中把有用旧 LoRA 融入 stable adapter。

---

## 4. 方法设计建议

### 4.1 最小可行版本

第一篇实验不建议一上来做太复杂的 MoE-LoRA。最小版本：

```text
Base model + LoRA(active rank r)
Diagnostic set D_diag

每 T steps:
    对每层每个 rank component 计算 score
    剪掉 bottom p% component
    将 top/high-confidence component merge 到 stable LoRA
    被剪掉和被 merge 的 active rank slots 重新初始化
```

为了避免直接 merge base 不可逆，建议先维护：

\[
\Delta W_{\text{stable}} = B_s A_s
\]

或维护一组 frozen stable components。推理时：

\[
W_0 + \Delta W_{\text{stable}} + \Delta W_{\text{active}}
\]

最终再决定是否 merge 到 base。

### 4.2 打分函数候选

#### Direct ablation score

\[
s_i = L_{\text{diag}}(W - \Delta W_i) - L_{\text{diag}}(W)
\]

- \(s_i > 0\)：删除会变差，component 有用。
- \(s_i < 0\)：删除会变好，component 有害。

优点：最直观。  
缺点：每个 component 都 forward 太贵。

可优化：

- layer-wise batch ablation；
- group ablation；
- 只对候选低分 component 做精确 ablation；
- 每次只诊断部分层。

#### First-order score

删除 component 对参数变化是 \(\delta_i = -\Delta W_i\)：

\[
\Delta L_i \approx g^\top \delta_i
\]

可用：

\[
s_i = -g^\top \Delta W_i
\]

这里 \(g=\nabla_{\Delta W} L_{\text{diag}}\)。如果删除会让 loss 上升，则 component 有用。

#### Fisher / OBD-like score

\[
s_i \approx \frac{1}{2} \sum_j F_j \Delta W_{i,j}^2
\]

其中：

\[
F_j \approx \mathbb{E}_{x\in D_{\text{diag}}}[g_j^2]
\]

这是 OBD 二阶思想的可扩展近似。

### 4.3 合并策略

不要一开始就 hard merge 到 base。推荐顺序：

1. **stable LoRA bank**：把高分 component freeze 到 bank。
2. **soft merge**：\(\Delta W_{\text{stable}} \leftarrow \Delta W_{\text{stable}} + \lambda \Delta W_{\text{good}}\)，\(\lambda \in [0.1, 1.0]\)。
3. **confidence merge**：只有连续 \(k\) 次高分才 merge。
4. **final merge**：训练结束后才把 stable + active merge 到 base。

### 4.4 剪枝策略

推荐：

- warmup 前 \(N\) steps 不剪；
- 每次最多剪 \(p=10\%-30\%\)，不建议一开始 50%；
- 低分连续 \(k=2\) 或 \(3\) 次才删除；
- 每层设置最小 rank，避免 layer collapse；
- 对 query/value/up/down projection 分别统计，避免某类层被清空。

### 4.5 Rank slot 重初始化

被释放的 rank slot 可用：

- Gaussian init；
- PiSSA/EVA-like data-driven init；
- 与已有 active/stable directions 正交的随机方向；
- 从旧 LoRA bank 中高相似 component 初始化。

其中“正交随机重启”最符合你说的“旋转一下”的直觉。

---

## 5. 实验设计

### 5.1 主任务选择

建议至少覆盖三类：

1. **NLU / classification**：GLUE、SuperGLUE 子集，便宜、稳定。
2. **Instruction / reasoning**：GSM8K、SVAMP、MATH 子集、BBH 子集。
3. **Domain adaptation / low-resource generation**：医学、法律、代码、摘要等。

如果目标是 LLM LoRA，推荐：

- 小模型快速验证：Llama-3.2-1B/3B、Qwen2.5-1.5B/3B。
- 中等模型主实验：Llama-3.1-8B、Mistral-7B、Qwen2.5-7B。

### 5.2 公平性控制

必须报告三种预算：

1. **Active trainable params**：每一步实际训练多少参数。
2. **Total stored adapter params**：stable bank + active LoRA 总共存多少。
3. **Inference params/latency**：推理时是否多 adapter 叠加，是否已 merge。

否则审稿人会质疑：你的方法是不是只是总 rank 越积越大。

推荐主表使用两种公平设置：

#### Setting A: same active rank

所有方法 active rank 相同，比较训练稳定性和最终性能。

#### Setting B: same final adapter budget

限制 stable + active 总 rank，不允许无限增长，比较容量利用效率。

如果你的方法在 A 下更强，但 B 下持平，也仍然可以说明它是一种训练时容量扩展机制。

### 5.3 主比较表

建议表格列：

| Method | Active rank | Final rank/storage | Trainable params | Diagnostic overhead | Accuracy/EM | Loss/PPL | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full FT | - | full | full | - | | | |
| LoRA | r | r | x | 1.0x | | | |
| QLoRA | r | r | x | 1.0x | | | |
| AdaLoRA | budget | learned | x | | | | |
| DyLoRA | max r | variable | x | | | | |
| SoRA | max r | sparse r | x | | | | |
| DoRA | r | r + mag | x | | | | |
| Ours | r | stable + active | x | | | | |

### 5.4 必做消融

| 消融 | 目的 |
|---|---|
| no diagnostic, random prune | 证明不是随机重启带来的 |
| train-set score instead of diagnostic score | 证明 diagnostic 泛化信号有用 |
| magnitude score | 对比简单范数剪枝 |
| first-order score | 看是否需要二阶 |
| Fisher/OBD score | 验证理论打分 |
| no merge, only prune/reinit | 证明 merge 对突破 rank 有用 |
| merge without prune | 证明剪枝/诊断有用 |
| no rotation/orthogonalization | 证明旋转只作为辅助 |
| different prune ratio | 10/20/30/50% |
| diagnostic frequency T | 100/500/1000 steps |
| diagnostic set size | 32/128/512 examples |

### 5.5 关键图

1. **Component score distribution over time**：显示一部分 rank direction 变有用，一部分长期有害/冗余。
2. **Rank recycling curve**：active rank 固定，stable rank 增长，性能持续提升。
3. **Ablation heatmap**：不同剪枝比例和诊断频率的性能。
4. **Rank utilization per layer**：和 AdaLoRA/SoRA 对比，展示不同层容量分配。
5. **Diagnostic vs final test correlation**：component score 与最终 test improvement 的相关性。

---

## 6. 可能的审稿质疑与对应策略

### 质疑 1：这是不是 AdaLoRA/SoRA 的变体？

回应：

- AdaLoRA/SoRA 主要动态调 rank 或稀疏化；
- 你的方法核心是 held-out diagnostic contribution；
- 你的方法有 merge-and-recycle，使 active rank budget 与 cumulative update capacity 解耦；
- 实验中加入 AdaLoRA/SoRA 并做 same-budget 比较。

### 质疑 2：是不是用了 validation set 训练，导致过拟合？

回应：

- 明确 diagnostic set 与 final validation/test 分离；
- 做 diagnostic set size 和 resampling 消融；
- 做 cross-domain diagnostic：用少量 held-out 混合数据诊断，在不同 test benchmark 上评估。

### 质疑 3：累计 stable LoRA 越来越大，不公平

回应：

- 同时报告 active params、stored params、inference latency；
- 做固定 final adapter budget 实验；
- 做 periodic compression：stable bank 超过预算时用 SVD recompress 到 fixed rank。

### 质疑 4：直接消融太贵

回应：

- 主方法用 first-order/Fisher approximation；
- direct ablation 只作为小模型 oracle 或校准；
- 报告 diagnostic overhead，例如每 500 steps 多 3%-8% 训练成本。

### 质疑 5：merge 不可逆，错误会污染模型

回应：

- 默认 merge 到 frozen stable adapter，不直接改 base；
- 高分连续 \(k\) 次才 merge；
- 使用 soft merge coefficient；
- 可回滚 stable bank component。

---

## 7. 推荐论文叙事

### Motivation

现有 LoRA 及其变体多关注结构：

- 固定低 rank；
- 动态 rank allocation；
- 稀疏 rank gate；
- 多 LoRA experts；
- 多 adapter merging。

但它们通常没有回答一个问题：

> 当前 LoRA 学到的每个 rank direction，是否真的改善了 held-out generalization？

训练过程中，LoRA 可能出现：

- redundant directions；
- harmful directions；
- task-specific overfitting directions；
- old LoRA 与新任务冲突；
- low-rank capacity 被早期错误方向占据。

### Method

提出 diagnostic validation-guided LoRA rank recycling：

1. 用 diagnostic set 估计每个 component 的泛化贡献；
2. 删除低贡献/负贡献 component；
3. 合并高置信 component；
4. 重新初始化释放 rank；
5. 可选地旋转/正交化，使 component 更可分。

### Claim

在相同 active rank 下，方法可以：

- 更高效使用 rank；
- 降低有害更新；
- 兼容旧 LoRA；
- 在多轮训练中累积超过单次 rank 的有效更新；
- 提供可解释诊断信号。

---

## 8. 推荐实现路线

### Stage 1：最小实验

- Base：RoBERTa-base 或 Llama/Qwen 小模型。
- Method：LoRA rank 8/16。
- Score：direct ablation + first-order。
- Dataset：GLUE 或小型 instruction benchmark。
- 目标：证明 diagnostic score 能预测 component usefulness。

### Stage 2：rank recycling

- 加 stable LoRA bank。
- 加 prune + merge + reinit。
- 和 LoRA、random prune、magnitude prune、AdaLoRA、SoRA 比。

### Stage 3：旧 LoRA 兼容

- 准备多个 task LoRA。
- 新任务上用 diagnostic set 选择/merge 旧 LoRA。
- 和 LoraHub、LoRA Soups、linear merge、TIES/DARE 比。

### Stage 4：MoE-LoRA 扩展

- 多 expert LoRA。
- diagnostic score 同时用于 expert pruning/rank growing。
- 和 MoELoRA、MoLE、MixLoRA、DR-LoRA 比。

---

## 9. 参考基线清单

### LoRA / PEFT

- LoRA: Low-Rank Adaptation of Large Language Models, 2021.
- QLoRA: Efficient Finetuning of Quantized LLMs.
- DoRA: Weight-Decomposed Low-Rank Adaptation, 2024.
- VeRA: Vector-based Random Matrix Adaptation, 2024.
- LoRA-FA: Efficient and Effective Low Rank Representation Fine-tuning.

### Dynamic / sparse rank

- AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning, ICLR 2023.
- DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation, EACL 2023.
- SoRA: Sparse Low-rank Adaptation of Pre-trained Language Models, EMNLP 2023.
- ARD-LoRA / L1RA / IncreLoRA / DR-LoRA：可作为后续扩展基线，视投稿时间和可复现性选择。

### LoRA pruning

- LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning, ACL Findings 2024.
- PrunedLoRA: Robust Gradient-Based structured pruning for Low-rank Adaptation in Fine-tuning.
- Magnitude / first-order / Fisher pruning baselines.

### MoE-LoRA

- MoELoRA: Contrastive Learning Guided Mixture of Experts on PEFT for LLMs, 2024.
- MoLE: Mixture of LoRA Experts, 2024.
- MixLoRA: LoRA-based Mixture of Experts.
- MoLA: MoE LoRA with Layer-wise Expert Allocation.
- DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models.

### LoRA composition / merging

- LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition, COLM 2024.
- LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks, 2024/2025.
- Task Arithmetic.
- Model Soups.
- TIES-Merging.
- DARE.

### Theory

- Optimal Brain Damage.
- Optimal Brain Surgeon.
- Taylor / Fisher / Hessian saliency pruning.
- Scalable second-order approximations such as Empirical Fisher and Hutchinson diagonal.

---

## 10. 最终判断

这个方向**值得做**，但要避免把卖点写成“LoRA 旋转突破 rank 限制”。更强、更严谨的卖点是：

> Existing LoRA variants allocate or route low-rank capacity, but they rarely diagnose whether learned low-rank directions improve held-out generalization. We propose a diagnostic validation-guided rank recycling framework that periodically identifies beneficial, redundant, and harmful LoRA directions; merges high-confidence directions into a stable adapter; prunes harmful directions; and reinitializes freed rank slots. This decouples active trainable rank from cumulative adaptation capacity while retaining PEFT efficiency.

最重要的实验不是单纯刷分，而是证明三件事：

1. diagnostic score 与最终泛化收益相关；
2. prune/merge/recycle 比 random/magnitude/train-loss pruning 更好；
3. 在相同 active rank 或相同 final budget 下，优于 LoRA、AdaLoRA、DyLoRA、SoRA 等强基线。
