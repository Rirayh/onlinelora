"""
Render figures from JSON metadata files.

PI hard constraint (05_pi_response_AB_parallel.md §4.4):
  Every plot script must FIRST dump a JSON file describing the figure
  (title, axes, series, x/y data, annotations), then this renderer
  reads the JSON and produces the PNG/SVG. Rationale: figures must be
  reproducible/auditable without re-running expensive experiments.

JSON schema (one figure per file):
{
  "figure_id": "fig1_<descriptor>",
  "title": "...",
  "xlabel": "...",
  "ylabel": "...",
  "xscale": "linear" | "log",
  "yscale": "linear" | "log",
  "xlim": [low, high] | null,
  "ylim": [low, high] | null,
  "kind": "line" | "scatter" | "bar" | "hbar" | "hist" | "errbar",
  "series": [
    {
      "label": "series A",
      "x": [...] | null,                # required except for hist
      "y": [...],                       # required
      "yerr": [...] | null,             # only for kind=errbar
      "color": "C0" | "#abcdef" | null,
      "marker": "o" | null,
      "linestyle": "-" | "--" | null,
      "alpha": 1.0 | null
    },
    ...
  ],
  "hlines": [{"y": val, "color": ".", "linestyle": "--", "label": "..."}] | [],
  "vlines": [{"x": val, "color": ".", "linestyle": "--", "label": "..."}] | [],
  "text_annotations": [{"x": ..., "y": ..., "text": "..."}] | [],
  "legend": true | false,
  "grid": true | false,
  "tight_layout": true,
  "figsize": [w, h],
  "dpi": 150
}

Usage:
  python scripts/plot_from_json.py --json path/to/fig.json --out path/to/fig.png
  python scripts/plot_from_json.py --json-dir path/to/json_dir --out-dir path/to/png_dir
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _g(d: dict, k: str, default=None):
    v = d.get(k, default)
    return default if v is None else v


def render(spec: dict[str, Any], out_path: str) -> None:
    figsize = _g(spec, "figsize", [6.4, 4.8])
    dpi = _g(spec, "dpi", 150)
    fig, ax = plt.subplots(figsize=tuple(figsize), dpi=dpi)

    kind = _g(spec, "kind", "line")
    series = spec.get("series", [])

    for s in series:
        x = s.get("x")
        y = s.get("y", [])
        label = s.get("label")
        color = s.get("color")
        marker = s.get("marker")
        linestyle = s.get("linestyle")
        alpha = s.get("alpha", 1.0)
        if kind == "line":
            ax.plot(x, y, label=label, color=color, marker=marker,
                    linestyle=linestyle or "-", alpha=alpha)
        elif kind == "scatter":
            ax.scatter(x, y, label=label, c=color, marker=marker or "o",
                       alpha=alpha)
        elif kind == "bar":
            ax.bar(x, y, label=label, color=color, alpha=alpha)
        elif kind == "hbar":
            ax.barh(x, y, label=label, color=color, alpha=alpha)
        elif kind == "hist":
            bins = s.get("bins", 30)
            ax.hist(y, bins=bins, label=label, color=color, alpha=alpha)
        elif kind == "errbar":
            yerr = s.get("yerr")
            ax.errorbar(x, y, yerr=yerr, label=label, color=color,
                        marker=marker or "o", linestyle=linestyle or "-",
                        alpha=alpha, capsize=3)
        else:
            raise ValueError(f"unsupported kind: {kind}")

    for hl in spec.get("hlines", []) or []:
        ax.axhline(y=hl.get("y", 0.0),
                   color=hl.get("color", "k"),
                   linestyle=hl.get("linestyle", "--"),
                   label=hl.get("label"),
                   alpha=hl.get("alpha", 0.7))
    for vl in spec.get("vlines", []) or []:
        ax.axvline(x=vl.get("x", 0.0),
                   color=vl.get("color", "k"),
                   linestyle=vl.get("linestyle", "--"),
                   label=vl.get("label"),
                   alpha=vl.get("alpha", 0.7))
    for tx in spec.get("text_annotations", []) or []:
        ax.text(tx["x"], tx["y"], tx.get("text", ""),
                fontsize=tx.get("fontsize", 9),
                color=tx.get("color", "black"))

    if spec.get("xlabel"):
        ax.set_xlabel(spec["xlabel"])
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"])
    if spec.get("title"):
        ax.set_title(spec["title"])
    if spec.get("xscale"):
        ax.set_xscale(spec["xscale"])
    if spec.get("yscale"):
        ax.set_yscale(spec["yscale"])
    if spec.get("xlim"):
        ax.set_xlim(spec["xlim"])
    if spec.get("ylim"):
        ax.set_ylim(spec["ylim"])
    if spec.get("grid", True):
        ax.grid(True, alpha=0.3)
    if spec.get("legend", True):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8)

    if spec.get("tight_layout", True):
        fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def render_file(json_path: str, out_path: str | None = None) -> str:
    with open(json_path) as f:
        spec = json.load(f)
    if out_path is None:
        base = os.path.splitext(json_path)[0]
        out_path = base + ".png"
    render(spec, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None,
                    help="single JSON spec to render")
    ap.add_argument("--out", type=str, default=None,
                    help="output image path (PNG/SVG/PDF)")
    ap.add_argument("--json-dir", type=str, default=None,
                    help="directory of *.json specs (renders each)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="output directory (if rendering a dir)")
    args = ap.parse_args()

    if args.json:
        out = render_file(args.json, args.out)
        print(f"wrote {out}")
        return
    if args.json_dir:
        in_dir = Path(args.json_dir)
        out_dir = Path(args.out_dir) if args.out_dir else in_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("*.json"))
        if not files:
            raise SystemExit(f"no *.json under {in_dir}")
        for jf in files:
            out = out_dir / (jf.stem + ".png")
            render_file(str(jf), str(out))
            print(f"wrote {out}")
        return
    raise SystemExit("must supply --json or --json-dir")


if __name__ == "__main__":
    main()
