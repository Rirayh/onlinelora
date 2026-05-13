# LoRA OBD-Recycling 调研补充 (v2)：最新基线锁定、White-space 与理论叙事

> 这是对 `lora_obd_rank_recycling_research.md` (v1, 712 行) 的补充。v1 的方法形式化、设计建议、实验框架已经完备。本 v2 聚焦三件事：
>
> 1. **补齐近 6 个月 (2025-09 ~ 2026-01) 才出现的强相关工作**——这些是 v1 还没覆盖、但你立项时必须正面回答的 baseline；
> 2. 用一张二维表把剩余 **white-space 精确锁定**到一格；
> 3. 把"这个想法是否有理论支持"从一段散文升级成 **三个互补的理论视角**，并明确每个视角的引用源头。

---

## 0. 一句话结论 (更新版)

> v1 已说清楚"做什么"。v2 的结论是：**这个想法仍有 white-space 但门正在关上**。半年内出现了 Sensitivity-LoRA (EMNLP-F'25) 和 CTR-LoRA (Oct 2025) 两篇直接占据"二阶 / 曲率信号 + LoRA"的工作，把卖点从"用 Hessian 诊断 LoRA"压缩到了"**用 val-set 上的 Hessian 信号驱动 prune+merge+rotate 的循环**"——剩下这一格是干净的，但方法叙事必须立刻收紧到这一格上，否则会被读者指认为既有方法的拼装。

---

## 1. 半年内必须正面回应的 5 篇新工作

按"威胁等级"排序。威胁等级 = 与你想说的卖点的重叠面积。

### 1.1 ★★★★★ Sensitivity-LoRA (Zhang et al., EMNLP Findings 2025, arXiv 2509.09119)

**做了什么**：

- 直接用 **Hessian 对角项** 作为权重矩阵的 sensitivity；
- 用 Taylor 展开 $\Delta E \approx \tfrac{1}{2}\sum_i h_{ii}\delta w_i^2$ 推导分数；
- 定义 global trace 和 local top-k / effective rank 两类指标；
- 按比例**动态分配 rank**到不同权重矩阵。

**为什么是最大威胁**：

> v1 文档 §2.1 把 OBD/Hessian 思想当作主要理论卖点，但 Sensitivity-LoRA 已经把**完全相同的二阶 saliency** 用在 LoRA rank 分配上了，并且声明"使用 Hessian 对角项以避免 SVD 计算开销"——这套话术你不能再原样用。

**你和它的真实差异(必须在论文 introduction 里点明)**：

| 维度 | Sensitivity-LoRA | 你的方法 |
|---|---|---|
| Hessian 估计在哪个集合上 | **训练集** | **诊断集 (held-out)** |
| 决策粒度 | 整个权重矩阵的 rank budget | 每个 rank-1 component |
| 动作 | 一次性分配 rank | 周期性 prune + merge + recycle |
| 是否区分有害方向 | 否（只看 saliency 大小） | 是（带符号，删有害） |
| 是否突破 active rank | 否 | 是（through merge bank） |

**实验必须做的对比**：把 Sensitivity-LoRA 跑在你的同一 setting 下，证明 **train-Hessian 不如 val-Hessian 预测 final test improvement**——这是 kill-or-seal 的核心实验。

### 1.2 ★★★★☆ CTR-LoRA (Oct 2025, arXiv 2510.15962)

**做了什么**：

- 用 **K-FAC + Hutchinson** 估计曲率（这是真二阶，比 Sensitivity-LoRA 的对角更强）；
- "Curvature-Aware Rank Scheduling"：用 whitened gradient 给 rank-1 方向打分；
- "Trust-Region 正则项"：$\lambda\sum_\ell \|A_\ell B_\ell^\top\|_M^2$，把更新约束在低曲率区域；
- 主要 baseline 是 **Sensitivity-LoRA**，自报提升 +1.2~1.7 点。

**为什么威胁很大**：

> 它已经做了"曲率信号 + rank 分配 + 稳定性约束"的三件套，并且**直接说自己 beat Sensitivity-LoRA**。你的方法如果只跑赢 LoRA/AdaLoRA 但跑不赢 CTR-LoRA，故事不成立。

**你和它的真实差异**：

- CTR-LoRA 的曲率仍在**训练梯度**上；它做的是 "训练时不要走进高曲率"，本质是**正则**而不是**诊断**。
- 它没有 merge / restart / rank recycling，单 LoRA 训完就结束。
- 它不区分"删冗余"和"删有害"。

**实验**：必须放进主表。如果跑不动 K-FAC，至少跑它的 ablation 中"只加 trust-region 正则"那一支。

### 1.3 ★★★★☆ DR-LoRA (arXiv 2601.04823, 2026)

**做了什么**：

- MoE 模型上的 LoRA；
- 用 **expert saliency = routing 频率 + gradient-based rank importance** 周期性**扩展**关键 expert 的 rank；
- 形成异构 rank 分布。

**和你 v1 §3.2 已经识别的关系一致**，但要补充一点：

> DR-LoRA 是 **monotone grow rank**（只加不减），你是 **prune + merge + recycle**（有删有增）。这意味着你需要一个"删了之后被 merge 掉的 capacity 不会丢"的 narrative，DR-LoRA 不需要这个。

如果你 v1 §5 stage 4 的 MoE 扩展要做，DR-LoRA 是 the must-beat。

### 1.4 ★★★☆☆ PrunedLoRA (Oct 2025, arXiv 2510.00192)

**做了什么**：

- LoRA + 结构化剪枝；
- **首个理论分析**：在权重扰动下，**gradient-based pruning 比 activation-based 更鲁棒**；
- 给出 Proposition 1：activation-based 的 loss inflation 依赖 module magnitude，gradient-based 不依赖。

**对你的影响**：

> 这是你**用得上的** theoretical lemma。它直接支持你 v1 §4.2 把 first-order/Fisher 打分作为主方法的理由。引用它 = 拿到一个免费的 theoretical 引理。

但它也是**间接竞争**：它的"剪 LoRA 内部结构"思路和你的"剪 component"重叠。差异在于它在 base model 结构上做剪枝，你在 LoRA 自身的 rank 方向上做剪枝。

### 1.5 ★★☆☆☆ Investigating ReLoRA (Sep 2025, arXiv 2509.12960)

**做了什么**：

> 在 11M-66M 小模型上系统评估 ReLoRA，发现 ReLoRA **不仅没有累积高秩，反而放大已有 rank deficiency，merge 后引入 ill-conditioned 更新**。

**为什么这是你的 motivation 金矿**：

这篇文章是**对 ReLoRA 一脉的实证打脸**。你的 narrative 可以变成：

> "ReLoRA 的 merge-restart 在大模型上 work，在小模型上反而退化（Weiss 2025）。我们认为根因是 **merge 是无差别的**——好方向和坏方向被一起吸进 base，坏方向通过 base 累积放大就变成 ill-conditioning。我们的 diagnostic-gated merge 正好回应这个失败模式。"

这就把"你为什么要在 merge 前加诊断"从"研究者直觉"升级成"有实证失败案例需要解决"。

### 1.6 三篇值得知道但威胁等级低的

- **L1RA** (Sep 2025, 2509.04884)：L1 正则诱导稀疏 rank，可作为 SoRA 类 baseline 的 modern replacement。
- **Flexi-LoRA** (ICML 2025)：input-adaptive dynamic ranks，每个输入选不同 rank，跟你的方向正交（你是 training-time，它是 inference-time）。
- **A Stronger Mixture of Low-Rank Experts** (ICML 2025, MoELoRA Riemannian)：MoE-LoRA 的几何细化，如果做 MoE 扩展时引用。
- **StelLA** (NeurIPS 2025)：Stiefel 流形上的 LoRA 子空间学习，**和你"旋转/正交化"卖点正交**——StelLA 把单个 LoRA 的 $A$ 约束在 Stiefel manifold 上，你是不同 LoRA 之间正交。
- **Cross-regularization** (ICML 2025)：通过 **validation gradient** 直接优化正则化超参，与你"用 val signal 做训练决策"是同一个 idea family，**必须引用作为理论先例**。
- **The LLM Surgeon** (NeurIPS 2024, 2312.17244)：K-FAC 曲率做 OBD-style LLM 剪枝，是你 v1 §2.1 OBD 思想在 LLM 上的现代版引用。

---

### 1.7 同期工作 (Concurrent Work) —— 不构成 prior art 但有可借鉴的经验

#### EPI: Evolving Parameter Isolation (April 2026, arXiv:2604.14010)

**arxiv ID 2604 = 2026 年 4 月，距本提案约 1 个月**。按学术惯例（concurrent window 通常 ≤ 3 个月），**不构成 prior art**，无需在 introduction 防御性对比，也不需要作为 baseline。本节单独列出，仅供方法设计借鉴。

**EPI 的核心论点**：

- 在 SFT / multi-task / continual learning 场景下，**参数重要性是随时间漂移的**——以前用静态 mask 选 "critical params" 然后冻结的做法（如 Lottery Ticket、SparseFT）忽略了这一点。
- 提出 **evolving isolation mask**：用 Fisher 信号在线估计参数重要性，**周期性更新**哪些参数被保护、哪些可以被释放回可训练池。
- 关键机制：**"plasticity recovery"**——已被冻结但重要性下降的参数可以重新进入训练，腾出位置给新涌现的关键参数。
- 主要为了缓解 catastrophic forgetting 和 task interference。

**与我们的关系（必须在 paper related work 一段话点清楚）**：

| 维度 | EPI | 我们 |
|---|---|---|
| **共享 premise** | 重要性随训练漂移，需要周期性重估 | ✓ 同一前提 |
| **信号来源** | **训练集** Fisher | **诊断集 (held-out)** 二阶 |
| **决策对象** | base model 的参数子集（决定哪些 freeze） | LoRA 的 rank-1 component（决定哪些 prune / merge / rotate） |
| **结构改动** | 不改 LoRA 结构，靠 mask 操作 base | 周期性重组 LoRA + 合并入 stable bank |
| **目标** | 防止遗忘、降低 task interference | 解耦 active rank 与累计 capacity，过滤有害方向 |
| **应用域** | 通用 SFT / continual learning | LoRA 微调的容量 recycling |

**可借鉴的经验（写代码时直接采纳）**：

1. **Plasticity recovery 的二段式判定**：EPI 不是"一次 Fisher 低就立刻释放"，而是连续 $k$ 次低于阈值才释放——这正好对应 v1 §4.4 "低分连续 $k$ 次才删除"的建议，EPI 的实证支持加强了这个设计。
2. **Mask 更新频率的实证**：EPI 报告每 N 步更新一次 mask 比每步更新更稳定。这给 v1 §5.4 "diagnostic frequency T = 100/500/1000 steps" 消融提供了一个先验范围。
3. **重要性归一化**：EPI 用层内归一化的 Fisher 分数避免跨层不可比——我们在 §3.4 的 5 个 saliency 变体上也应该层内归一化后再做 ranking。
4. **温和释放优于硬释放**：EPI 发现 "soft unfreeze with ramp-up LR" 比直接释放更稳。对应到我们的方法：rank slot 重初始化后的 warmup LR schedule 值得做一个消融。

**在论文中的正确写法（建议进 related work 的最后半段）**：

> *Concurrent with our work, EPI (Anonymous, 2026) independently observes that parameter importance is temporally non-stationary and proposes evolving isolation masks based on training-set Fisher scoring for catastrophic-forgetting mitigation in SFT. While we share the temporal-non-stationarity premise, EPI operates on base-model parameters and decides which to freeze, whereas we operate on LoRA rank-1 components and decide which to prune, merge, or rotate; further, our diagnostic signal is computed on a held-out validation set rather than the training set. The two approaches are orthogonal and can plausibly be composed (EPI's mask on the base, our recycling on the adapter).*

**对实验设计的影响**：

- **不需要把 EPI 加入 §4 的主表 baseline**（它解决的是不同问题），但可以在附录消融"我们的方法 + EPI-style base masking"看是否互补。
- **EPI 不威胁 §2 white-space 表的格点**——它的格点在 "训练 Fisher × base mask 更新"，跟我们的 "val 二阶 × LoRA 剪+merge+rotate" 在两个完全不同的维度上。
- **可以在 STATUS.md 里把 EPI 加入 "concurrent work to monitor" 列表**——如果 EPI 后续扩展到 LoRA 场景，需要及时调整 framing。

---

## 2. White-space 二维锁定

把所有方法按 "**信号源 × 决策动作**" 排开：

| 信号 \ 动作 | 静态 rank 分配 | 动态扩 rank | 单次剪枝 | 剪+合并 (无诊断) | 剪+合并 + 旋转 + 诊断 |
|---|---|---|---|---|---|
| 范数/magnitude | LoRA, DoRA | — | LoRA-drop | — | — |
| 训练 loss | — | DyLoRA | — | ReLoRA, COLA | — |
| 训练梯度/sensitivity | — | AdaLoRA, L1RA | LoRAPrune, PrunedLoRA | — | — |
| 训练 Hessian/曲率 | Sensitivity-LoRA | DR-LoRA | LLM Surgeon | CTR-LoRA | — |
| Routing 频率 | — | MoE-Sieve, DR-LoRA | — | — | — |
| **Val gradient** | Cross-reg (only on hyperparams) | — | — | — | — |
| **Val Hessian / 二阶** | — | — | — | — | **← 这一格** |

**关键观察**：

1. 所有"动态 / 剪枝 / 合并"列里，**没有任何一种方法的信号来自 val 集的二阶量**；
2. **最右下角整列**（"剪+合并+旋转+诊断"）整个是空的，不止 val-Hessian；
3. Cross-regularization 占住了"val gradient → 调超参"，但没占"val signal → 调 LoRA structure"。

**你的真正卖点（必须改写到 abstract 第一句）**：

> *"We are the first to use **held-out validation second-order signals** to drive a **train→prune→merge→rotate** cycle on LoRA components, decoupling active rank from cumulative adaptation capacity while filtering harmful (rather than just redundant) updates."*

**v1 文档需要修订的地方**：

- v1 §2.3 标题"旋转 / 正交化的理论位置"和 §10 都说"merge + rank recycling 突破 rank"——这点没问题；但要补一句**与 ReLoRA / COLA 的区别在于 merge 是 diagnostic-gated 的**，否则在二维表上你和 ReLoRA 同一格。
- v1 §3.1 没有 Sensitivity-LoRA 和 CTR-LoRA，要加进 must-compare。
- v1 §3.4 LoRA composition / merging 列表里漏了 **"Adaptive LoRA Merge with Parameter Pruning" (Miyano & Arase, ACL Findings 2025)**——它实际上做了 "merge + 剪 + 用 val 集评估"，**虽然是多 LoRA 后期合并而非训练中循环**，但读者会问你和它的差别。

---

## 3. 三个互补的理论视角

v1 §2 把理论分成三段（OBD / val 集 / 旋转），但写得偏散。建议在论文里改成下面这三视角，每个有 explicit 引用源：

### 3.1 视角 A：Val-Hessian-gated update 的泛化保证

**借用源**：Cross-regularization (ICML 2025)。

**核心论点**：在 val 上做二阶 Taylor 展开，

$$\Delta L_{\text{val}}(\Delta W_i) \approx g_v^\top \Delta W_i + \tfrac{1}{2}\Delta W_i^\top H_v \Delta W_i$$

只 merge 满足 $\Delta L_{\text{val}}(\Delta W_i) < 0$ 的方向，**等价于在 val loss 上做 Frank-Wolfe 一阶下降**（$g_v$ 项主导时）或**带二阶修正的 line search**（$H_v$ 项不可忽略时）。

Cross-regularization 已经证明：在凸条件下，用 val gradient 调整模型复杂度参数收敛到 cross-validation 最优点。把它扩展到 LoRA component 选择只需要把"超参"换成"是否保留 component 的 0/1 指示变量"。

**这给你提供的论点**：

> 我们的 merge gate 不是启发式，而是 val-loss 单调下降算法的一步；任何被 merge 的 component 都伴随 $\Delta L_{\text{val}} \leq 0$ 的可证保证。

### 3.2 视角 B：Merge-Rotate 循环的 Frank-Wolfe 收敛

**借用源**：COLA (Chain of LoRA, ICML 2024)。

COLA 已经证明：每一轮 LoRA = base 上做一次 Frank-Wolfe 的线性优化，整个 chain 收敛到 nonconvex 平稳点。

**你的扩展**：把每一轮 LoRA 分两步——

- 训练阶段：standard Frank-Wolfe step；
- 诊断阶段：在 base 加进 LoRA **之前**，先用 val-Hessian 做 line search-style 的 component-wise 校正。

**等价于**：每一轮的 vertex 不是直接用 trained $BA$，而是用 $\sum_{i\in\mathcal{S}} b_i a_i^\top$，其中 $\mathcal{S} = \{i: \Delta L_{\text{val}}(\Delta W_i) < 0\}$。

这是 **constrained Frank-Wolfe with side information**，理论上 step 长度比 vanilla COLA 更稳，因为坏方向被排除。

### 3.3 视角 C：ReLoRA 失败案例作为反向 motivation

**借用源**：Investigating ReLoRA (Sep 2025)。

那篇的实证结论：

- ReLoRA 在 SLM 上 **降低** effective rank（不是升）；
- merge 后 condition number 急剧恶化。

**你的论点**：

> 这是无差别 merge 的症状。如果坏方向被 merge 进 base，base 的奇异谱被噪声方向占据，后续 LoRA 在残差子空间训练时面对 ill-conditioned residual，导致优化退化。Diagnostic gate 能阻止这条 failure path——这是一个**可测试的 hypothesis**。

**对应的实验**（必做之一）：

- Reproduce ReLoRA 的 SLM 失败 setting；
- 跑你的 diagnostic-gated 版本；
- 测量每轮 merge 后 base + stable 的有效秩和 condition number；
- 如果你的方法能让 effective rank 单调上升、condition number 不爆炸，你就**实证回答了 ReLoRA 的 open problem**。

这个实验比刷 GLUE 强 1 个点重要 10 倍。

---

## 4. 锁定后的最小 baseline 集

v1 §3 列了一长串。建议**主表**只留下面 7 个，剩下进 appendix：

| Baseline | 不能省的理由 |
|---|---|
| LoRA | 任何 PEFT paper 的 floor |
| DoRA | 当前 PEFT 最强 single-LoRA |
| AdaLoRA | "动态 rank + 重要性" 经典 |
| **Sensitivity-LoRA** | 直接占"二阶+LoRA"的 EMNLP 2025 工作 |
| **CTR-LoRA** | 直接占"曲率+trust region+LoRA"的 Oct 2025 工作 |
| ReLoRA | merge-restart 鼻祖，正向打 |
| COLA | merge-restart 的 SOTA + 你的理论框架来源 |

如果做 MoE 扩展：再加 **DR-LoRA** 和 **MoELoRA Riemannian (ICML 2025)**。

如果做"兼容旧 LoRA"扩展：再加 **LoraHub** 和 **Adaptive LoRA Merge (ACL-F 2025)**。

---

## 5. 两个 kill-or-seal 实验

如果只能做两个实验决定这个 paper 立不立，做这俩：

### 实验 1：Predictive Validity of the Diagnostic Score

**问题**：你声称 val-Hessian saliency 比 train-Hessian saliency 更预测 final test improvement。

**做法**：

1. Train LoRA $T$ steps；
2. 对每个 rank-1 component 同时算 train-Hessian saliency $s^t_i$ 和 val-Hessian saliency $s^v_i$；
3. 同时算 oracle: 真的把这个 component 删掉测 final test loss 的变化 $\Delta_i^{\text{test}}$；
4. 报告 Spearman correlation $\rho(s^t, \Delta^{\text{test}})$ 和 $\rho(s^v, \Delta^{\text{test}})$。

**预期结果**：$\rho(s^v, \Delta^{\text{test}}) > \rho(s^t, \Delta^{\text{test}})$，且差距随训练后期变大（过拟合区）。

**为什么是 kill-or-seal**：如果这个相关性差距不显著，你的"用 val 而不是 train"的卖点崩塌；反过来，如果差距显著，你拿到了**整篇论文的核心 figure**，胜过任何刷分。

### 实验 2：ReLoRA 失败模式的修复

**问题**：你能否在 ReLoRA 失败的 SLM regime 下让 merge 真的累积高秩。

**做法**：复现 Weiss 2025 的 11M-66M LM pretrain setting，画三条线：

- ReLoRA：effective rank 随 merge 轮数下降；
- 你的方法：effective rank 单调上升；
- Full-rank training：上界。

**为什么是 kill-or-seal**：如果你能让 ReLoRA 失败的曲线翻过来，就直接拿到了 ICLR-tier 的 narrative：**"diagnostic gating fixes the central failure mode of merge-restart methods."**

这两个实验合起来覆盖：信号选对了 (实验 1) + 整个 pipeline work (实验 2)。其他刷分实验都是支撑。

---

## 6. 与 v1 的对接：建议的修订列表

| v1 位置 | 建议改动 |
|---|---|
| §0 (结论先行) | 卖点改写为"val 二阶诊断驱动的 prune-merge-rotate 循环"；强调 vs Sensitivity-LoRA 和 CTR-LoRA |
| §2.1 OBD 推导 | 明确写出 Sensitivity-LoRA 用了相同公式但在 train 集，并给出 train→val 的迁移论证 |
| §2.2 Held-out diagnostic | 加 Cross-regularization 作为先例引用 |
| §2.3 旋转 | 加一段"与 ReLoRA 失败模式 (Weiss 2025) 的对应"，把旋转/重启从"理论增强"变成"修复已知 failure" |
| §3.1 必须比较 | 加入 Sensitivity-LoRA 和 CTR-LoRA |
| §3.4 merging | 加入 Adaptive LoRA Merge (ACL-F 2025) |
| §5.4 消融 | 把"train-set score vs diagnostic score"提到必做 #1（这就是实验 1） |
| §5.5 关键图 | 加入 effective rank / condition number 随 merge 轮数的曲线 (实验 2) |
| §10 最终判断 | 删掉"突破 rank 限制"措辞，改为"diagnostic-gated rank accumulation surpasses active-rank bottleneck and avoids the ill-conditioning failure of unconditioned merge" |

---

## 7. 对"是否值得做"的更新判断

|  | v1 (712 行那版) | v2 (本文档) |
|---|---|---|
| 想法是否有理论支持 | 有，OBD + val pruning + Frank-Wolfe-like | 有且更强，三视角各自有 ICML/NeurIPS 引用 |
| 是否有 white-space | 有，宽 | 有，但已被压缩到一格 (val 二阶 × 剪+merge+rotate) |
| 是否有强 baseline | 列了一堆 | 锁定到 7 个核心 baseline |
| 是否有 kill-or-seal 实验 | 没明说 | 实验 1 (相关性) + 实验 2 (ReLoRA 修复) |
| 立项风险 | 中等 | 中—低，前提是动作要快（再 6 个月这一格也可能被占） |

**结论**：

> 想法值得做，但 white-space 在快速收紧。**在动手做实验之前，要先把 introduction 和 method section 的 framing 锁死到 "val-Hessian-gated diagnostic pipeline" 这一格上**——任何宽泛的"diagnose LoRA updates"措辞都会让你被读者投射到 Sensitivity-LoRA / CTR-LoRA / AdaLoRA 上。先做实验 1，48 小时之内能给出 go/no-go 信号。

---

## 8. 引用清单 (v1 之外新增)

### 必引

- Zhang et al., **Sensitivity-LoRA: Low-Load Sensitivity-Based Fine-Tuning for Large Language Models**, EMNLP Findings 2025, arXiv:2509.09119.
- **CTR-LoRA: Curvature-Aware and Trust-Region Guided Low-Rank Adaptation**, arXiv:2510.15962, Oct 2025.
- Yu et al., **PrunedLoRA: Robust Gradient-Based Structured Pruning for Low-rank Adaptation**, arXiv:2510.00192, Oct 2025.
- Weiss et al., **Investigating ReLoRA: Effects on the Learning Dynamics of Small Language Models**, BlackBoxNLP 2025 / arXiv:2509.12960.
- Stein Brito, **Cross-regularization: Adaptive Model Complexity through Validation Gradients**, ICML 2025.
- van der Ouderaa et al., **The LLM Surgeon**, NeurIPS 2024 / arXiv:2312.17244.
- Xia et al., **Chain of LoRA (COLA)**, ICML 2024, arXiv:2401.04151.

### 强相关

- **DR-LoRA**, arXiv:2601.04823 (2026).
- **L1RA**, arXiv:2509.04884.
- **Flexi-LoRA**, ICML 2025.
- **A Stronger Mixture of Low-Rank Experts**, ICML 2025 (MoELoRA Riemannian).
- **StelLA**, NeurIPS 2025.
- Miyano & Arase, **Adaptive LoRA Merge with Parameter Pruning for Low-Resource Generation**, ACL Findings 2025, arXiv:2505.24174.
- **LoGo: Instance-level Dynamic LoRA Selection and Merging**, arXiv:2511.07129, Nov 2025.

### 远相关但 framing 用得上

- **SoLoRA: LoRA Meets Second-Order Optimization**, ICLR 2026 submission.
- **TsqLoRA**, arXiv:2509.18585 (sensitivity + data quality combined).
- **MoE-Sieve**, arXiv:2603.24044.
- **Hybrid Routing for a Mixture of LoRA Experts (HotMoE)**, AAAI.
- **RoRA: Rotational Rank Adaptation**, SSRN 6101568, Jan 2026 (避免 "rotation" 一词被读者错位)。

### 同期工作 (Concurrent, ≤ 3 个月, 不构成 prior art)

- **EPI: Evolving Parameter Isolation**, arXiv:2604.14010, Apr 2026. 处理 SFT 中参数重要性时变漂移，提出周期更新的 isolation mask；信号来自训练 Fisher、对象是 base 参数。与本方法正交（见 §1.7）。

---

## 附录 A：一句话定位每篇核心 paper

| Paper | 一句话 |
|---|---|
| AdaLoRA | SVD-form LoRA + train sensitivity 分配 rank |
| Sensitivity-LoRA | Train Hessian 对角 → matrix-level rank budget |
| CTR-LoRA | Train K-FAC + 曲率正则 + 曲率感知 rank scheduling |
| DR-LoRA | MoE 上按 expert saliency 单调扩 rank |
| PrunedLoRA | 用 LoRA 梯度做结构剪枝，证明梯度法 > 激活法 |
| ReLoRA | Merge-restart 累积高秩（在小模型上失败） |
| COLA | Merge-restart 的 Frank-Wolfe 视角 + 收敛证明 |
| Cross-regularization | Val gradient 调超参（你扩展到调 LoRA 结构） |
| The LLM Surgeon | K-FAC OBD 给 LLM 做剪枝 |
| **You** | **Val Hessian 驱动 prune+merge+rotate 循环 of LoRA components** |
