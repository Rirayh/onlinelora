#!/usr/bin/env python
"""Stage 1 figures (matplotlib only — seaborn not in env).

  fig1_correlation_grid.png : 3 tasks x N_ckpts grid of scatter (x=delta_test, y=S5_fisher_val)
  fig2_rho_over_time.png    : 5 saliency lines vs step, 1 panel per task
  fig3_train_vs_val_paired.png : HEADLINE — paired rho train (S4) vs val (S5), 15 dots + means
  fig4_harmful_auc.png      : AUC of -S3_fo_val_signed for harmful detection, per checkpoint per task
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASKS = ["sst2", "mrpc", "rte"]
COLORS = {"sst2": "#1f77b4", "mrpc": "#ff7f0e", "rte": "#2ca02c"}


def _load_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_summary(stage1_root: Path, task: str) -> dict | None:
    p = stage1_root / task / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fig1_correlation_grid(stage1_root: Path, out_path: Path) -> None:
    summaries = {t: _load_summary(stage1_root, t) for t in TASKS}
    summaries = {k: v for k, v in summaries.items() if v is not None}
    if not summaries:
        return
    max_ckpts = max(len(s["checkpoints"]) for s in summaries.values())
    n_tasks = len(summaries)
    fig, axes = plt.subplots(n_tasks, max_ckpts, figsize=(3.0 * max_ckpts, 2.6 * n_tasks),
                             squeeze=False, sharex=False, sharey=False)
    for i, (task, s) in enumerate(summaries.items()):
        ckpts = sorted(s["checkpoints"], key=lambda c: int(c["step"]))
        for j in range(max_ckpts):
            ax = axes[i][j]
            if j >= len(ckpts):
                ax.axis("off"); continue
            step = int(ckpts[j]["step"])
            comp_path = stage1_root / task / str(step) / "components.jsonl"
            if not comp_path.exists():
                ax.set_title(f"missing"); continue
            recs = _load_jsonl(comp_path)
            x = np.array([r["delta_test"] for r in recs])
            y = np.array([r["S5_fisher_val"] for r in recs])
            ax.scatter(x, y, s=8, alpha=0.5, color=COLORS.get(task, "k"))
            rho = ckpts[j].get("S5_fisher_val_rho_vs_delta")
            ax.set_title(f"{task} step={step}\nrho_val={rho:.2f}" if rho is not None else f"{task} step={step}", fontsize=9)
            if j == 0:
                ax.set_ylabel("S5_fisher_val", fontsize=8)
            if i == n_tasks - 1:
                ax.set_xlabel("delta_test (oracle)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    fig.suptitle("Fig 1: oracle delta_test vs val-Fisher saliency (per task × checkpoint)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig2_rho_over_time(stage1_root: Path, out_path: Path) -> None:
    summaries = {t: _load_summary(stage1_root, t) for t in TASKS}
    summaries = {k: v for k, v in summaries.items() if v is not None}
    if not summaries:
        return
    sal_names = ["S1_mag", "S2_fo_tr", "S3_fo_val", "S4_fisher_tr", "S5_fisher_val"]
    n_tasks = len(summaries)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4.5 * n_tasks, 3.4), squeeze=False)
    for ax, (task, s) in zip(axes[0], summaries.items()):
        ckpts = sorted(s["checkpoints"], key=lambda c: int(c["step"]))
        steps = [int(c["step"]) for c in ckpts]
        for name in sal_names:
            rhos = [c.get(name + "_rho_vs_delta") for c in ckpts]
            rhos = [np.nan if r is None else r for r in rhos]
            ax.plot(steps, rhos, marker="o", label=name)
        ax.set_title(f"{task}")
        ax.set_xlabel("step")
        if ax is axes[0][0]: ax.set_ylabel("Spearman ρ(saliency, Δ_test)")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Fig 2: Predictive ρ vs training step (per saliency variant)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig3_train_vs_val_paired(stage1_root: Path, out_path: Path) -> None:
    summaries = {t: _load_summary(stage1_root, t) for t in TASKS}
    summaries = {k: v for k, v in summaries.items() if v is not None}
    if not summaries:
        return
    pts_fi = []   # (rho_train_fisher, rho_val_fisher, task, step)
    pts_fo = []
    for task, s in summaries.items():
        for ckpt in s["checkpoints"]:
            step = int(ckpt["step"])
            tr = ckpt.get("S4_fisher_tr_rho_vs_delta"); va = ckpt.get("S5_fisher_val_rho_vs_delta")
            if tr is not None and va is not None:
                pts_fi.append((tr, va, task, step))
            tr2 = ckpt.get("S2_fo_tr_rho_vs_delta"); va2 = ckpt.get("S3_fo_val_rho_vs_delta")
            if tr2 is not None and va2 is not None:
                pts_fo.append((tr2, va2, task, step))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=True)
    for ax, pts, title in [
        (axes[0], pts_fi, "Fisher (S4 train vs S5 val)"),
        (axes[1], pts_fo, "First-order (S2 train vs S3 val)")]:
        if not pts:
            ax.set_title(f"{title}: no data"); continue
        for tr, va, task, step in pts:
            ax.scatter(tr, va, s=60, color=COLORS.get(task, "k"), edgecolor="k", linewidth=0.5)
            ax.annotate(f"{task[:2]}.{step}", (tr, va), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        # y=x reference
        all_vals = [v for p in pts for v in (p[0], p[1])]
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.1 if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.7)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("ρ (train saliency)")
        if ax is axes[0]: ax.set_ylabel("ρ (val saliency)")
        # paired mean
        d = np.array([p[1] - p[0] for p in pts])
        ax.set_title(f"{title}\nmean(val - train)={d.mean():+.3f}, +ve {(d>0).sum()}/{d.size}")
    for h_task in TASKS:
        axes[0].scatter([], [], color=COLORS[h_task], label=h_task, s=60, edgecolor="k", linewidth=0.5)
    axes[0].legend(title="task", fontsize=8, loc="lower right")
    fig.suptitle("Fig 3 (headline): val saliency vs train saliency, paired per (task, checkpoint)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig4_harmful_auc(stage1_root: Path, out_path: Path) -> None:
    summaries = {t: _load_summary(stage1_root, t) for t in TASKS}
    summaries = {k: v for k, v in summaries.items() if v is not None}
    if not summaries:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for task, s in summaries.items():
        ckpts = sorted(s["checkpoints"], key=lambda c: int(c["step"]))
        steps = [int(c["step"]) for c in ckpts]
        aucs = [c.get("S3_fo_val_signed_neg_auc_harmful") for c in ckpts]
        harm = [c.get("harmful_rate") for c in ckpts]
        aucs_plot = [np.nan if v is None else v for v in aucs]
        ax.plot(steps, aucs_plot, marker="o", color=COLORS.get(task, "k"), label=f"{task} (AUC)")
        ax2 = ax.twinx() if task == TASKS[0] else None
        if ax2 is not None and task == TASKS[0]:
            ax2.set_ylabel("harmful_rate (dashed)")
            twin = ax2
        # plot harmful rate on the same axis (right) for each task as dashed
    # also plot harmful rate per task using twin axis
    ax2 = ax.twinx()
    for task, s in summaries.items():
        ckpts = sorted(s["checkpoints"], key=lambda c: int(c["step"]))
        steps = [int(c["step"]) for c in ckpts]
        harm = [c.get("harmful_rate") for c in ckpts]
        harm_plot = [np.nan if v is None else v for v in harm]
        ax2.plot(steps, harm_plot, marker="x", linestyle="--",
                 color=COLORS.get(task, "k"), label=f"{task} (harmful%)")
    ax.set_xlabel("step")
    ax.set_ylabel("AUC of -S3_fo_val_signed for harmful detection")
    ax.axhline(0.65, color="red", lw=0.6, ls=":")
    ax.text(ax.get_xlim()[1], 0.65, " 0.65 threshold", fontsize=7, color="red", va="center")
    ax.legend(loc="upper left", fontsize=8)
    ax2.set_ylabel("harmful_rate")
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Fig 4: harmful-detection AUC vs step + harmful_rate (per task)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_root", default=str(ROOT / "results/stage1"))
    parser.add_argument("--out_dir", default=str(ROOT / "plots/stage1"))
    args = parser.parse_args()
    stage1_root = Path(args.stage1_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    fig1_correlation_grid(stage1_root, out_dir / "fig1_correlation_grid.png")
    fig2_rho_over_time(stage1_root, out_dir / "fig2_rho_over_time.png")
    fig3_train_vs_val_paired(stage1_root, out_dir / "fig3_train_vs_val_paired.png")
    fig4_harmful_auc(stage1_root, out_dir / "fig4_harmful_auc.png")
    print(f"wrote 4 figures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
