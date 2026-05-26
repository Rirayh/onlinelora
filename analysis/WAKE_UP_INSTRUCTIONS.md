# Wake-up state — 2026-05-26 12:30 UTC

## Last user message
"按你建议来吧" — execute (a)+(b)+(c) from my proposal:
- (a) S1 framing test (Spearman G(W0) vs G(W))
- (b) S2.5 schedule pilot
- (c) saliency_v2.py implementation

PI directive: `analysis/COMM_PI_TO_AGENT/2026-05-26_pi_saliency_revamp_v2.md`
ACK string required in commit body: `ACK_v2_saliency_revamp`

## What's DONE in this session (uncommitted)

1. ✅ `scripts/run_s1_framing_test.py` — eps-scaled-B Spearman framing test
2. ✅ `src/saliency_v2.py` — IG + per-sample + t-stat + Fisher×signvote
   Unit-tested: BH-FDR rejects first 5 p-values correctly; t_stat_decision identifies
   8/8 true-keep components (mean=-1) and behaves random for noise.
3. ✅ M0 (Muon, commit 5e6056c) — already pushed, now path-δ fallback
4. ✅ Exp-0a (commit d0d5da3) — `--random_drop_rate` already pushed

## What's REMAINING (in order)

1. ⏳ **`stage3_run.py`** — add CLI: `--saliency_estimator`, `--saliency_v2_m_ig`,
   `--saliency_v2_alpha`, `--drop_schedule`. Add `DROP_SCHEDULE_REGISTRY` +
   `parse_drop_schedule()`. Wire schedule into random branch. Wire v2 into gated
   branches.
2. ⏳ **`scripts/exp_schedule_pilot_orchestrator.py`** — 12 schedules driver
3. ⏳ **commit + push** body must contain `ACK_v2_saliency_revamp`
4. ⏳ **launch S1** on GPU 7 (free since Muon smoke completed)

## Active running jobs (snapshot 12:00 UTC)

| GPU | Job | Status |
|---|---|---|
| 0 | cola (exp_v1) | step ~2000/3000 (let finish) |
| 1-6 | Exp-1 drop-rate sweep | step ~700/3000 (~23%); ETA ~7-8h |
| 7 | **FREE** (Muon smoke v3 done: FINAL VAL_LOSS=1.4054) |

Decision: don't kill Exp-1; reuse dr0.25/0.5/0.75 as schedules #2/#1/#3.

## Repo / env

- Repo dir: `/mnt/cpfs/junlongke/onlinelora/lora_obd` (git in subdir)
- Python: `/mnt/cpfs/junlongke/miniconda3/envs/espo/bin/python`
- Model: `/mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B`
- S1 baseline adapter: `results/exp_v1/qwen3-8b/tulu3-sft/relora_baseline/seed42/adapter/`

## stage3_run.py edits — exact specs

### CLI args (insert after existing `--muon_ns_steps`, before `--smoke`):
```python
p.add_argument("--saliency_estimator", choices=["v1", "v2"], default="v1",
               help="v1: legacy first_order_saliency (sign-only). "
                    "v2: IG+t-stat+FisherSignVote (PI 2026-05-26 v2 directive).")
p.add_argument("--saliency_v2_m_ig", type=int, default=4,
               help="IG interpolation points for v2 estimator.")
p.add_argument("--saliency_v2_alpha", type=float, default=0.1,
               help="BH-FDR significance level for v2 t-stat gating.")
p.add_argument("--drop_schedule", default="",
               help="Per-event drop_rate schedule. Either: registry name "
                    "(const_0p5, anneal_down, ...), comma list "
                    "(0.9,0.5,0.5,0.5,0.5,0.5), or 'random_schedule:seed=N'. "
                    "Empty = constant --random_drop_rate.")
```

### DROP_SCHEDULE_REGISTRY (add at module level near other constants):
```python
DROP_SCHEDULE_REGISTRY = {
    "const_0p5":          [0.5]*6,
    "const_0p25":         [0.25]*6,
    "const_0p75":         [0.75]*6,
    "anneal_down":        [0.75, 0.65, 0.55, 0.45, 0.35, 0.25],
    "anneal_up":          [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    "triangle_up_down":   [0.25, 0.45, 0.65, 0.65, 0.45, 0.25],
    "triangle_down_up":   [0.75, 0.55, 0.35, 0.35, 0.55, 0.75],
    "early_burst":        [0.9, 0.5, 0.5, 0.5, 0.5, 0.5],
    "late_burst":         [0.5, 0.5, 0.5, 0.5, 0.5, 0.9],
    "bookend_burst":      [0.9, 0.3, 0.3, 0.3, 0.3, 0.9],
    "extreme_alternate":  [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
}

def parse_drop_schedule(spec: str, n_events: int):
    if not spec:
        return None
    if spec in DROP_SCHEDULE_REGISTRY:
        sched = list(DROP_SCHEDULE_REGISTRY[spec])
    elif spec.startswith("random_schedule:seed="):
        seed = int(spec.split("=", 1)[1])
        rng = np.random.default_rng(seed)
        sched = [float(rng.uniform(0.1, 0.9)) for _ in range(n_events)]
    elif "," in spec:
        sched = [float(x) for x in spec.split(",")]
    else:
        raise ValueError(f"unknown --drop_schedule: {spec}")
    if len(sched) < n_events:
        sched = sched + [sched[-1]] * (n_events - len(sched))
    return sched[:n_events]
```

### Wire schedule into main()

After line 942 `log.info(f"merge events scheduled at: {sorted(merge_steps)}")` add:
```python
drop_schedule_list = parse_drop_schedule(args.drop_schedule, len(merge_steps))
if drop_schedule_list is not None:
    log.info(f"drop_schedule '{args.drop_schedule}' resolved to: {drop_schedule_list}")
```

In random branch (currently line 1042-1046):
```python
elif gate_sign == "random":
    if drop_schedule_list is not None:
        rate_for_event = drop_schedule_list[event_idx - 1]  # event_idx is 1-indexed
    else:
        rate_for_event = args.random_drop_rate
    keep_masks, stats = build_keep_mask(
        handles, "random", fo_val_signed={},
        target_drop_rate=rate_for_event,
        rng_seed=args.seed + event_idx,
    )
    stats["scheduled_drop_rate"] = float(rate_for_event)
```

### Wire v2 estimator into gated branch

Replace the `gated` branch's saliency block (line 1078-1106) with version that
forks on `args.saliency_estimator`. v1 path is unchanged. v2 path:

```python
if args.saliency_estimator == "v2":
    from src.saliency_v2 import (
        integrated_gradient_saliency_per_sample,
        t_stat_decision,
        fisher_signvote_score,
    )
    # IG over m points; per-sample (n_calib total per t)
    per_sample = integrated_gradient_saliency_per_sample(
        model, handles, sal_loader, device,
        m=args.saliency_v2_m_ig,
        max_samples=min(args.saliency_calib_n, args.diag_batches * 8),
        signed=True,
    )
    keep_masks, v2_info = t_stat_decision(
        per_sample, alpha=args.saliency_v2_alpha,
        rng_seed=args.seed + event_idx,
    )
    fsv_scores = fisher_signvote_score(per_sample)
    # Build stats dict matching v1 shape
    n_total = sum(h.r for h in handles)
    n_kept = sum(int(m.sum().item()) for m in keep_masks.values())
    flat_scores = []
    for L in fsv_scores: flat_scores.extend([float(x) for x in fsv_scores[L].tolist()])
    qs = [float(np.quantile(flat_scores, q)) for q in (0.05,0.25,0.5,0.75,0.95)] if flat_scores else []
    stats = {
        "components_total": n_total, "components_kept": n_kept,
        "components_dropped": n_total - n_kept,
        "drop_rate": 1.0 - n_kept / max(n_total, 1),
        "score_quantiles": qs,
        "per_layer_keep_counts": {L: int(m.sum().item()) for L,m in keep_masks.items()},
        "saliency_estimator": "v2", **v2_info,
    }
else:
    # v1 path: unchanged from existing code
    fo_signed = first_order_saliency(...)  # existing call
    keep_masks, stats = build_keep_mask(handles, gate_sign, fo_signed)
    stats["saliency_estimator"] = "v1"
```

⚠️ **Caveat**: v2 ignores `gate_sign` (S3pos_drops vs S3neg_drops vs S2train_pos_drops)
because t_stat_decision implicitly does S3-style gating (mean<0 -> keep). For
S2train_pos_drops we'd need to swap loader to train_loader BEFORE calling
integrated_gradient_saliency_per_sample. For S3neg_drops we'd flip the keep/drop
rule. Initially v2 only supports S3pos semantics; document as KNOWN LIMITATION.

### scripts/exp_schedule_pilot_orchestrator.py

12 schedules: const_0p25/0p5/0p75 (REUSE Exp-1), anneal_down, anneal_up,
triangle_up_down, triangle_down_up, early_burst, late_burst, bookend_burst,
extreme_alternate, random_schedule:seed=42, random_schedule:seed=43

Settings: qwen3-8b/tulu3-sft, seed=42, total_steps=3000, **merge_every=500**
(directive specifies 6 events). Method=relora_random_drop. Output:
results/exp_schedule/qwen3-8b/tulu3-sft/<schedule_name>/seed42/.

For const_0p25/0p5/0p75: SKIP (point to existing Exp-1 results dirs in summary).

For other 9: launch when GPU free. Each cell ~3-4hr if run alone, longer with
contention. Total ~12-15h on 7 GPUs.

## EXACT next steps when context resumes

1. Look at current stage3_run.py L65-95 area for argparse, and L935-942 for
   merge_steps creation. Apply edits as above.
2. Verify with `--help` that new args parse.
3. Quick test: `python scripts/stage3_run.py --total_steps 50 --merge_every 25
   --method relora_random_drop --drop_schedule 0.1,0.9 --random_drop_rate 0.5
   --total_steps 50 --merge_every 25 ...` should log resolved schedule.
4. Write scripts/exp_schedule_pilot_orchestrator.py
5. git commit -am "ACK_v2_saliency_revamp ..." + push
6. Launch S1: `CUDA_VISIBLE_DEVICES=7 python scripts/run_s1_framing_test.py
   --base_model /mnt/cpfs/public_data/public_model/Qwen3/Qwen3-8B
   --adapter_dir results/exp_v1/qwen3-8b/tulu3-sft/relora_baseline/seed42/adapter
   --out_path analysis/results_v3/saliency_framing/spearman_qwen3-8b_tulu3.json
   --n_calib 256 --eps 1e-3 &> logs/s1_framing.log &`
7. Push next status at 15:00 UTC

## Commit message template (for next push)

```
S1+S2+S2.5 plumbing: framing test + saliency_v2 + drop_schedule registry

ACK_v2_saliency_revamp

Per PI directive 2026-05-26 v2 (saliency_revamp_v2.md):

S1 (scripts/run_s1_framing_test.py - new):
  Compute Spearman rho between s_end (W=W0+dW) and s_start (W=W0+e*dW)
  saliencies on trained adapter. Decision rule:
    rho >= 0.5 -> A not critical, demote IG
    rho <  0.3 -> A critical, implement IG
  Uses eps-scaled-B trick; rank correlation invariant to eps.

S2 (src/saliency_v2.py - new):
  - first_order_saliency_per_sample (per-sample <gA, A>, signed)
  - integrated_gradient_saliency_per_sample (m=4 interpolation, B->tB)
  - t_stat_decision: BH-FDR (alpha=0.1) + Bernoulli(0.5) random fallback
  - fisher_signvote_score = sign_vote * sqrt(fisher)
  Unit-tested with synthetic data: BH FDR control verified; correctly
  identifies 8/8 strong-signal keeps and falls to random on pure noise.

stage3_run.py CLI:
  - --saliency_estimator {v1,v2} (default v1, no behaviour change)
  - --saliency_v2_m_ig (default 4), --saliency_v2_alpha (default 0.1)
  - --drop_schedule <spec>: registry name | comma list |
                            random_schedule:seed=N | empty(=constant)
  - DROP_SCHEDULE_REGISTRY: 11 named schedules
  - parse_drop_schedule(spec, n_events) helper
  - drop_schedule_list resolved at start of main(), used per merge event
    in random branch (event_idx-1 lookup).

S2.5 (scripts/exp_schedule_pilot_orchestrator.py - new):
  9 new schedule cells (const_0p25/0p5/0p75 reused from Exp-1).
  qwen3-8b/tulu3-sft, seed=42, total_steps=3000, merge_every=500 (6 events).

src/saliency.py NOT modified (per directive: keep v1 reproducible).

Next: launch S1 on GPU 7. After S1 result decide whether to drop IG axis
or keep it.
```
