"""
F5: build_main_table.py

Walk results/stage3_v2/{model}/{dataset}/{method}/seed{seed}/lm_eval/**/results_*.json
and produce a single CSV + Markdown table covering every (model, dataset, method, seed)
cell with key downstream metrics.

Output:
  results/stage3_v2/summary/main_table.csv
  results/stage3_v2/summary/main_table.md

Metric columns (one row per cell):
  gsm8k_strict, gsm8k_flex, hellaswag_acc_norm, arc_challenge_acc_norm,
  mmlu_acc, bbh_acc, humaneval_pass1, ifeval_strict, truthfulqa_mc1
plus:
  best_step, best_val_loss, final_val_loss, aborted, completed_steps

Missing cells are left blank.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "stage3_v2"
SUMMARY = RESULTS / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

EXCLUDE_DIRS = {"summary", "figures"}


def latest_results_json(lm_eval_dir: Path) -> Optional[Path]:
    """Pick the most recently modified results_*.json under any subdir."""
    cands = list(lm_eval_dir.rglob("results_*.json"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def best_lm_eval_dir(seed_dir: Path) -> Optional[Path]:
    """Return the most appropriate lm_eval dir for the canonical best-ckpt result.

    Priority:
      1. lm_eval_v3/ (post-P0-fix rerun: adapter/ now contains valid lora_B!=0)
      2. lm_eval_v2/ (earlier contamination-cleared rerun)
      3. lm_eval/    (original run)
    Each must contain at least one results_*.json to qualify.
    lm_eval_step*/ dirs are multi-ckpt analysis, NOT used as canonical.
    """
    candidates = []
    for d in seed_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("lm_eval") and not re.match(r"lm_eval_step\d+", d.name):
            jsons = list(d.rglob("results_*.json"))
            if jsons:
                latest = max(jsons, key=lambda p: p.stat().st_mtime)
                candidates.append((latest.stat().st_mtime, d))
    if not candidates:
        return None
    # priority: v3 > v2 > newest by mtime (v1 / lm_eval_rebuilt etc.)
    for prefer in ("lm_eval_v3", "lm_eval_v2"):
        hits = [d for _, d in candidates if d.name == prefer]
        if hits:
            return hits[0]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_metrics(res_json: Path) -> dict:
    try:
        d = json.loads(res_json.read_text())
    except Exception:
        return {}
    out = {}
    R = d.get("results", {})
    # gsm8k
    if "gsm8k" in R:
        g = R["gsm8k"]
        out["gsm8k_strict"] = g.get("exact_match,strict-match")
        out["gsm8k_flex"] = g.get("exact_match,flexible-extract")
    # hellaswag
    if "hellaswag" in R:
        h = R["hellaswag"]
        out["hellaswag_acc_norm"] = h.get("acc_norm,none")
        out["hellaswag_acc"] = h.get("acc,none")
    # arc_challenge
    if "arc_challenge" in R:
        a = R["arc_challenge"]
        out["arc_challenge_acc_norm"] = a.get("acc_norm,none")
        out["arc_challenge_acc"] = a.get("acc,none")
    # mmlu (group)
    for k in R:
        if k == "mmlu":
            out["mmlu_acc"] = R[k].get("acc,none")
        if k == "mmlu_pro":
            out["mmlu_pro_acc"] = R[k].get("acc,none") or R[k].get("exact_match,custom-extract")
        if k == "bbh":
            out["bbh_acc"] = R[k].get("acc,none") or R[k].get("exact_match,get-answer")
        if k == "hendrycks_math":
            out["math_acc"] = R[k].get("exact_match,none") or R[k].get("exact_match,strict-match")
        if k == "humaneval":
            out["humaneval_pass1"] = R[k].get("pass@1,create_test")
        if k == "ifeval":
            out["ifeval_strict"] = R[k].get("inst_level_strict_acc,none") or R[k].get("prompt_level_strict_acc,none")
        if k == "truthfulqa_mc1":
            out["truthfulqa_mc1"] = R[k].get("acc,none")
    return out


def extract_train_meta(cell_dir: Path) -> dict:
    """Extract best_step / best_val_loss / final_val_loss / aborted from logs."""
    meta = {}
    val_jsonl = cell_dir / "val_loss.jsonl"
    if val_jsonl.exists():
        try:
            rows = [json.loads(l) for l in val_jsonl.read_text().splitlines() if l.strip()]
            pre = [r for r in rows if not r.get("post_merge") and not r.get("final")
                   and "val_loss" in r and "step" in r]
            if pre:
                best = min(pre, key=lambda r: r["val_loss"])
                meta["best_step"] = int(best["step"])
                meta["best_val_loss"] = float(best["val_loss"])
            if rows:
                last = rows[-1]
                if "val_loss" in last:
                    meta["final_val_loss"] = float(last["val_loss"])
                if "step" in last:
                    meta["completed_steps"] = int(last["step"])
        except Exception:
            pass
    # aborted: scan run.log for red-line / ABORT
    log_path = cell_dir / "run.log"
    if log_path.exists():
        try:
            tail = log_path.read_text(errors="ignore")[-50000:]
            meta["aborted"] = any(
                k in tail for k in ("RED-LINE ABORT", "red-line abort", "ABORTED", "ABORT triggered")
            )
        except Exception:
            meta["aborted"] = False
    return meta


def discover_cells() -> list[dict]:
    """Yield {model, dataset, method, seed, cell_dir, lm_eval_dir} for every cell."""
    cells = []
    for model_dir in sorted(RESULTS.iterdir()):
        if not model_dir.is_dir() or model_dir.name in EXCLUDE_DIRS:
            continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for method_dir in sorted(dataset_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                for seed_dir in sorted(method_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                        continue
                    seed = seed_dir.name.replace("seed", "")
                    cells.append({
                        "model": model_dir.name,
                        "dataset": dataset_dir.name,
                        "method": method_dir.name,
                        "seed": seed,
                        "cell_dir": seed_dir,
                        "lm_eval_dir": best_lm_eval_dir(seed_dir),
                    })
    return cells


def build_table(cells: list[dict]) -> list[dict]:
    rows = []
    for c in cells:
        row = {
            "model": c["model"],
            "dataset": c["dataset"],
            "method": c["method"],
            "seed": c["seed"],
        }
        row.update(extract_train_meta(c["cell_dir"]))
        if c["lm_eval_dir"] and c["lm_eval_dir"].exists():
            rj = latest_results_json(c["lm_eval_dir"])
            if rj:
                row.update(extract_metrics(rj))
                row["lm_eval_json"] = str(rj.relative_to(ROOT))
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out: Path) -> list[str]:
    cols = [
        "model", "dataset", "method", "seed",
        "gsm8k_strict", "gsm8k_flex",
        "hellaswag_acc_norm", "arc_challenge_acc_norm",
        "mmlu_acc", "mmlu_pro_acc", "bbh_acc", "math_acc",
        "humaneval_pass1", "ifeval_strict", "truthfulqa_mc1",
        "best_step", "best_val_loss", "final_val_loss",
        "completed_steps", "aborted", "lm_eval_json",
    ]
    import csv
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # round floats
            r2 = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(r2)
    return cols


def write_markdown(rows: list[dict], out: Path) -> None:
    """Compact markdown: model x dataset, methods as rows, GSM8K/HellaSwag/ARC as cols."""
    by_md = {}
    for r in rows:
        key = (r["model"], r["dataset"])
        by_md.setdefault(key, []).append(r)

    lines = []
    lines.append("# Main Table (lm-eval downstream)\n")
    lines.append(f"_Auto-generated by `scripts/build_main_table.py`_\n")

    for (model, dataset), group in sorted(by_md.items()):
        lines.append(f"\n## {model} / {dataset}\n")
        lines.append("| method | seed | gsm8k_strict | gsm8k_flex | hellaswag | arc_chal | mmlu | bbh | best_step | best_val | aborted |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        # sort methods
        order = ["lora_vanilla", "relora_baseline",
                 "relora_diag_gated_S3pos", "relora_diag_gated_S3neg",
                 "relora_random_drop", "relora_train_gated",
                 "adalora", "dora"]
        group_sorted = sorted(group, key=lambda r: (
            order.index(r["method"]) if r["method"] in order else 999, r["seed"]))
        for r in group_sorted:
            def fmt(k, scale=100, dp=2):
                v = r.get(k)
                if v is None:
                    return ""
                if isinstance(v, float):
                    return f"{v*scale:.{dp}f}"
                return str(v)
            lines.append("| {m} | {s} | {gs} | {gf} | {hs} | {ac} | {mm} | {bb} | {bst} | {bv} | {ab} |".format(
                m=r["method"], s=r["seed"],
                gs=fmt("gsm8k_strict"), gf=fmt("gsm8k_flex"),
                hs=fmt("hellaswag_acc_norm"), ac=fmt("arc_challenge_acc_norm"),
                mm=fmt("mmlu_acc"), bb=fmt("bbh_acc"),
                bst=r.get("best_step", ""),
                bv=(f"{r.get('best_val_loss'):.4f}" if isinstance(r.get("best_val_loss"), float) else ""),
                ab="Y" if r.get("aborted") else "",
            ))

    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=Path, default=SUMMARY / "main_table.csv")
    ap.add_argument("--out_md", type=Path, default=SUMMARY / "main_table.md")
    args = ap.parse_args()

    cells = discover_cells()
    print(f"discovered {len(cells)} cells")
    rows = build_table(cells)
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)
    n_with_eval = sum(1 for r in rows if r.get("gsm8k_strict") is not None)
    print(f"wrote {args.out_csv} ({len(rows)} rows, {n_with_eval} with gsm8k)")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
