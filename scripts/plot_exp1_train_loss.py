"""Exp-1 train-loss analysis plot (PI feedback §5).

Reads train_loss.jsonl + val_loss.jsonl from each Exp-1 cell
(scripts/exp_drop_rate_orchestrator.py), and produces:

  analysis/results_v3/exp1_train_loss_analysis.png

Panels:
  (1) train_loss curves for all 6 drop_rates, overlaid, with merge-event
      vertical lines (at 750, 1500, 2250 by orchestrator config)
  (2) post-merge spike height (loss[t_merge+25] - loss[t_merge-25])
      and recovery half-life (steps until loss returns within 5% of
      pre-merge level), per cell
  (3) final converged train_loss (mean of last 5 entries) vs drop_rate

Robust to partial Exp-1 runs (skips missing cells, marks incomplete).

Usage:
    python scripts/plot_exp1_train_loss.py
        [--out analysis/results_v3/exp1_train_loss_analysis.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "results" / "exp_drop_rate" / "qwen3-8b" / "tulu3-sft"
DROP_RATES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
DR_LABELS = ["dr0", "dr0.1", "dr0.25", "dr0.5", "dr0.75", "dr0.9"]
MERGE_STEPS = [750, 1500, 2250]
COLORS = plt.cm.viridis(np.linspace(0.05, 0.95, len(DROP_RATES)))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def cell_data(label: str) -> dict:
    base = EXP_ROOT / label / "seed42"
    train = load_jsonl(base / "train_loss.jsonl")
    val = load_jsonl(base / "val_loss.jsonl")
    return {"label": label, "train": train, "val": val}


def spike_and_recovery(train: list[dict], merge_step: int,
                       window: int = 25) -> tuple[float, int | None]:
    """Spike = loss[merge+window] - loss[merge-window]; half_life =
    smallest k s.t. loss[merge+k] <= 1.05 * loss[merge-window]."""
    if not train:
        return float("nan"), None
    steps = np.array([e["step"] for e in train])
    losses = np.array([e["train_loss"] for e in train])
    pre_idx = np.where(steps <= merge_step - window)[0]
    post_idx = np.where(steps >= merge_step + window)[0]
    if len(pre_idx) == 0 or len(post_idx) == 0:
        return float("nan"), None
    pre = float(losses[pre_idx[-1]])
    post0 = float(losses[post_idx[0]])
    spike = post0 - pre
    target = 1.05 * pre
    recovery = None
    for j in post_idx:
        if losses[j] <= target:
            recovery = int(steps[j]) - merge_step
            break
    return spike, recovery


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/results_v3/exp1_train_loss_analysis.png")
    args = ap.parse_args()

    cells = [cell_data(lbl) for lbl in DR_LABELS]
    has_data = [bool(c["train"]) for c in cells]
    n_have = sum(has_data)
    if n_have == 0:
        print("[ERR] No Exp-1 cells have train_loss.jsonl yet.")
        return 1

    max_step = max(
        max((e["step"] for e in c["train"]), default=0) for c in cells
    )
    incomplete = max_step < 3000

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

    # Panel 1: train_loss curves
    ax1 = fig.add_subplot(gs[0, :])
    for c, dr, color in zip(cells, DROP_RATES, COLORS):
        if not c["train"]:
            continue
        steps = [e["step"] for e in c["train"]]
        losses = [e["train_loss"] for e in c["train"]]
        ax1.plot(steps, losses, label=f"dr={dr}", color=color, lw=1.4, alpha=0.9)
    for ms in MERGE_STEPS:
        ax1.axvline(ms, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax1.set_xlabel("step")
    ax1.set_ylabel("train_loss")
    title = ("Exp-1 random-drop sweep: train_loss curves"
             + (f" (PARTIAL, max_step={max_step}/3000)" if incomplete else ""))
    ax1.set_title(title)
    ax1.legend(loc="upper right", ncol=2, fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: post-merge spike + recovery (only event 1 = step 750)
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    spikes_all = []
    recoveries_all = []
    for ms in MERGE_STEPS:
        if max_step < ms + 100:
            break
        spikes = []
        recoveries = []
        for c in cells:
            sp, rec = spike_and_recovery(c["train"], ms)
            spikes.append(sp)
            recoveries.append(rec if rec is not None else np.nan)
        spikes_all.append((ms, spikes))
        recoveries_all.append((ms, recoveries))

    for (ms, sp), color_idx in zip(spikes_all,
                                   range(len(spikes_all))):
        ax2.plot(DROP_RATES, sp, "o-", label=f"merge@{ms}",
                 color=plt.cm.plasma(color_idx / max(1, len(spikes_all) - 1)
                                     if len(spikes_all) > 1 else 0.5))
    ax2.set_xlabel("drop_rate")
    ax2.set_ylabel("post-merge spike (loss diff)")
    ax2.set_title("Post-merge spike height vs drop_rate")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    for (ms, rec), color_idx in zip(recoveries_all,
                                    range(len(recoveries_all))):
        ax3.plot(DROP_RATES, rec, "s-", label=f"merge@{ms}",
                 color=plt.cm.plasma(color_idx / max(1, len(recoveries_all) - 1)
                                     if len(recoveries_all) > 1 else 0.5))
    ax3.set_xlabel("drop_rate")
    ax3.set_ylabel("recovery half-life (steps to <=105% pre-merge)")
    ax3.set_title("Recovery speed vs drop_rate")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Panel 3: final converged train_loss vs drop_rate
    ax4 = fig.add_subplot(gs[2, :])
    finals = []
    for c in cells:
        if c["train"]:
            tail = [e["train_loss"] for e in c["train"][-5:]]
            finals.append(float(np.mean(tail)))
        else:
            finals.append(np.nan)
    ax4.plot(DROP_RATES, finals, "D-", color="firebrick", lw=2)
    for x, y in zip(DROP_RATES, finals):
        if not np.isnan(y):
            ax4.text(x, y, f"  {y:.3f}", fontsize=8, va="center")
    ax4.set_xlabel("drop_rate")
    ax4.set_ylabel("final train_loss (mean of last 5 logs)")
    ax4.set_title(f"Converged train_loss vs drop_rate (max_step={max_step})")
    ax4.grid(alpha=0.3)

    fig.tight_layout()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"WROTE {out_path}")

    # Also dump numeric summary as JSON
    summary = {
        "max_step": int(max_step),
        "incomplete": bool(incomplete),
        "drop_rates": DROP_RATES,
        "final_train_loss": [None if np.isnan(f) else f for f in finals],
        "spikes_per_event": {str(ms): [None if np.isnan(s) else s for s in sp]
                             for ms, sp in spikes_all},
        "recovery_per_event": {str(ms): [None if np.isnan(r) else int(r) for r in rec]
                               for ms, rec in recoveries_all},
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"WROTE {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
