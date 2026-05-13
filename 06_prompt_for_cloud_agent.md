你好。Stage 1 我看完了，决策如下。完整指令在/mnt/cpfs/junlongke/onlinelora/lora_obd/05_pi_response_AB_parallel.md，请先读它，本消息只是导读 + 启动顺序。

【决策】Path A 为主 + Path B 并行。7 张可用 A100 全部跑满（GPU 2 不要碰，是别人的）：
- Path A 占 GPU 0/1/3/4：full_rank / relora_baseline / relora_diag_gated(drop if S3>0) / relora_diag_gated(drop if S3<0)
- Path B 占 GPU 5/6/7：5-fold Fisher 重跑 stage1 saliency on sst2/mrpc/rte

【启动前必须先做的一件事】
你在 stage1_aggregate.py L270-275 把符号歧义用「symmetric AUC = max(auc(+), auc(-))」绕过去了。Stage 2 的 gate predicate 完全依赖符号，所以这件事必须先钉死。具体见 05 §1。10 分钟内做完：

cd /mnt/cpfs/junlongke/onlinelora/lora_obd
echo "## sign_check $(date '+%F %T')" >> STATUS.md
for f in $(ls results/stage1/*/*/auc_signed.json | sort); do
    echo "$f : $(cat $f)" >> STATUS.md
done

把 15 行 raw AUC 贴回 STATUS.md，然后判方向：
- 多数 S3_fo_val_signed_neg_auc_harmful > 0.5  → handover 方向对，drop if S3<0
- 多数 < 0.5                                   → 方向反，drop if S3>0
- 跨任务不一致                                 → 停下等我，不要做 per-task gate

不管 §1.2 结论是什么，Phase A 还是两个方向并行跑——赢的那个就是真符号，论文里这是一个 sub-contribution。

【绘图硬约束】
从今往后，所有图必须先 dump 一份 JSON 元数据（schema 见 05 §4.3），再用 scripts/plot_from_json.py 渲染 PNG。Stage 1 现有的 4 张图也要回填 JSON。这个不阻塞 Stage 2 启动但不能拖。

【你之前 silently 做的 4 件事，必须正面回应】
分别在 STATUS.md 各写一条记录，见 05 §5：
1. cond2 sign test 从 AND 改成 OR（aggregator L279）
2. cond1 用 |ρ| 而非 signed ρ（aggregator L175-190）——要复算一次 signed 版本对照
3. symmetric AUC 取代 signed AUC（见上）——靠 §1 sign-check 一起解决
4. harmful_rate 65-86% 你解释为过拟合验证——可能是 noise floor 主导，要算 median|delta_test|/recent_test_loss_std 对照

【上报节奏】
- §1.2 raw AUC + 你的判断结论 → STATUS（≤10min）
- Phase A 4 jobs 启动 → STATUS 一行（PIDs + GPU map）
- Phase B 3 jobs 启动 → STATUS 一行
- 每 merge event（每 5000 步）后 → STATUS 4-7 行（val_loss / effective_rank / drop_rate / 哪个 gate_sign 在赢）
- 任何 job 崩 / sign 方向 task-specific → 停下等我

【红线】
- EPI (2604.14010) 不作 baseline
- 不动 espo 环境
- STATUS.md append-only，不覆盖
- 不在 diagnostic/test_holdout 训
- 不碰 GPU 2

【入口】
$PY = /mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python
export HF_HOME=/mnt/cpfs/junlongke/hf_cache
cd /mnt/cpfs/junlongke/onlinelora/lora_obd

进度报告全部 append 到 STATUS.md 顶部。让我看到「sign-check + Phase A 启动 + Phase B 启动」三条 STATUS 条目后我会回来 review。

冲。
