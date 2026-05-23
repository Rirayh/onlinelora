#!/usr/bin/env python3
"""Aggregate OPLoRA analysis JSONs into Fig_A (rho_k), Fig_B (subspace overlap), Fig_C (drift heatmap)."""
from __future__ import annotations
import json
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON_DIR = Path("/mnt/cpfs/junlongke/onlinelora/lora_obd/analysis/oplora/jsons")
FIG_DIR = Path("/mnt/cpfs/junlongke/onlinelora/lora_obd/analysis/oplora/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [8, 16, 32, 64, 128]
METHOD_COLOR = {
    "lora_vanilla": "#1f77b4",
    "dora": "#2ca02c",
    "relora_diag_gated_S3pos": "#d62728",
}
METHOD_LABEL = {
    "lora_vanilla": "LoRA",
    "dora": "DoRA",
    "relora_diag_gated_S3pos": "ReLoRA-S3pos",
}


def load_all() -> Dict[tuple, dict]:
    out = {}
    for fp in sorted(JSON_DIR.glob("*.json")):
        d = json.load(open(fp))
        key = (d["model"], d["dataset"], d["method"])
        out[key] = d
    return out


def fig_a_rho_k(all_data: Dict[tuple, dict]):
    """rho_k vs k, line per method, averaged over layers, faceted by model."""
    models = sorted({k[0] for k in all_data.keys()})
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.0), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        for method in METHOD_LABEL.keys():
            key = (model, "tulu3-sft", method)
            if key not in all_data:
                continue
            d = all_data[key]
            curve = []
            for k in K_LIST:
                vals = [L["rho_k"][str(k)] for L in d["layers"]]
                curve.append(statistics.mean(vals))
            ax.plot(K_LIST, curve, "-o", color=METHOD_COLOR[method],
                    label=METHOD_LABEL[method], linewidth=1.8, markersize=5)
        ax.set_xscale("log")
        ax.set_xticks(K_LIST)
        ax.set_xticklabels([str(k) for k in K_LIST])
        ax.set_xlabel("k (top-k SVD subspace of W0)")
        ax.set_title(model)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.94, 1.005)
    axes[0].set_ylabel(r"$\rho_k = \|P_{\bot} \Delta W\|_F / \|\Delta W\|_F$")
    axes[-1].legend(loc="lower left", fontsize=9)
    fig.suptitle("Fig_A — Energy of ΔW outside top-k subspace of W0 (avg over layers)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_A_rho_k.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"saved {out} (+ .png)")


def fig_b_overlap(all_data: Dict[tuple, dict]):
    """Subspace overlap (avg layers) vs k, two panels (left, right)."""
    models = sorted({k[0] for k in all_data.keys()})
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 6.6), sharey="row", sharex=True)
    if len(models) == 1:
        axes = axes.reshape(2, 1)
    for col, model in enumerate(models):
        for row, side in enumerate(["left", "right"]):
            ax = axes[row, col]
            for method in METHOD_LABEL.keys():
                key = (model, "tulu3-sft", method)
                if key not in all_data:
                    continue
                d = all_data[key]
                curve = []
                for k in K_LIST:
                    field = f"subspace_overlap_{side}"
                    vals = [L[field][str(k)] for L in d["layers"]]
                    curve.append(statistics.mean(vals))
                ax.plot(K_LIST, curve, "-o", color=METHOD_COLOR[method],
                        label=METHOD_LABEL[method], linewidth=1.8, markersize=5)
            ax.set_xscale("log")
            ax.set_xticks(K_LIST); ax.set_xticklabels([str(k) for k in K_LIST])
            ax.grid(True, alpha=0.3)
            if row == 1: ax.set_xlabel("k")
            if col == 0: ax.set_ylabel(f"overlap_{side} (avg)")
            if row == 0: ax.set_title(model)
    axes[0, -1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Fig_B — Subspace overlap of top-k(W0) and top-k(ΔW)  (||U1^T U2||_F^2 / k)",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_B_overlap.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"saved {out} (+ .png)")


def fig_c_drift(all_data: Dict[tuple, dict]):
    """Per-window drift heatmap: rows=layers, cols=window transitions, faceted (model, method).

    Layers grouped by name (e.g. self_attn.q_proj across all model.layers.X).
    """
    models = sorted({k[0] for k in all_data.keys()})
    methods = list(METHOD_LABEL.keys())
    fig, axes = plt.subplots(len(methods), len(models), figsize=(4.2 * len(models), 3.6 * len(methods)),
                             sharex=True, sharey=False)
    if len(models) == 1 and len(methods) == 1:
        axes = np.array([[axes]])
    elif len(models) == 1:
        axes = axes.reshape(len(methods), 1)
    elif len(methods) == 1:
        axes = axes.reshape(1, len(models))

    for r, method in enumerate(methods):
        for c, model in enumerate(models):
            ax = axes[r, c]
            key = (model, "tulu3-sft", method)
            if key not in all_data:
                ax.axis("off"); continue
            d = all_data[key]
            ls = d["layers"]
            # Determine window count (first layer with drift)
            n_w = max((len(L.get("per_window_drift", [])) for L in ls), default=0)
            if n_w == 0:
                ax.axis("off"); continue
            # Build matrix [layers, n_w]; pad shorter with NaN
            M = np.full((len(ls), n_w), np.nan)
            for i, L in enumerate(ls):
                drifts = L.get("per_window_drift", [])
                for j, v in enumerate(drifts[:n_w]):
                    if v is not None:
                        M[i, j] = v
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{model} / {METHOD_LABEL[method]}", fontsize=10)
            if c == 0: ax.set_ylabel("layer index")
            if r == len(methods) - 1: ax.set_xlabel("window transition")
            ax.set_xticks(range(n_w))
            ax.set_xticklabels([f"w{i+1}->w{i+2}" for i in range(n_w)], fontsize=8, rotation=30)
    fig.suptitle("Fig_C — Per-window subspace drift  (1 - overlap_left at k=32)", fontsize=12)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="drift")
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    out = FIG_DIR / "fig_C_drift.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"saved {out} (+ .png)")


def emit_summary_table(all_data: Dict[tuple, dict]):
    out = FIG_DIR.parent / "summary.md"
    lines = ["# OPLoRA analysis — summary", "",
             "| model | method | layers | rho_k(8) | rho_k(32) | rho_k(128) | overlap_L(8) | overlap_R(8) | avg_drift |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(all_data.keys()):
        model, dataset, method = key
        d = all_data[key]
        ls = d["layers"]
        if not ls:
            continue
        def avg(field, k):
            return statistics.mean([L[field][str(k)] for L in ls])
        drifts = []
        for L in ls:
            for v in L.get("per_window_drift", []):
                if v is not None: drifts.append(v)
        avg_drift = statistics.mean(drifts) if drifts else float("nan")
        lines.append(
            f"| {model} | {METHOD_LABEL.get(method, method)} | {len(ls)} | "
            f"{avg('rho_k', 8):.4f} | {avg('rho_k', 32):.4f} | {avg('rho_k', 128):.4f} | "
            f"{avg('subspace_overlap_left', 8):.4f} | {avg('subspace_overlap_right', 8):.4f} | "
            f"{avg_drift:.4f} |"
        )
    out.write_text("\n".join(lines))
    print(f"saved {out}")


def main():
    all_data = load_all()
    print(f"loaded {len(all_data)} cells")
    if not all_data:
        print("no jsons found; nothing to do")
        return
    fig_a_rho_k(all_data)
    fig_b_overlap(all_data)
    fig_c_drift(all_data)
    emit_summary_table(all_data)


if __name__ == "__main__":
    main()
