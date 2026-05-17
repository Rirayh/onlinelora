"""
F5: analyze_mmlu_per_domain.py

Compute MMLU per-subject accuracy by parsing `samples_mmlu_*.jsonl` files (per-subtask
files emitted by lm-eval-harness when --log_samples is used).  Aggregates to the four
MMLU groups (STEM / Humanities / Social Sciences / Other) using the standard mapping.

Output:
  results/stage3_v2/summary/mmlu_per_domain.csv
  results/stage3_v2/summary/figures/{model}_{dataset}_mmlu_domain.png

Skips silently when no MMLU samples are present.
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "stage3_v2"
SUMMARY = RESULTS / "summary"
FIGS = SUMMARY / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

EXCLUDE_DIRS = {"summary", "figures"}

# Standard MMLU subject -> domain mapping (from lm-eval-harness mmlu_categories.py)
SUBCATEGORIES = {
    "abstract_algebra": "stem", "anatomy": "stem", "astronomy": "stem",
    "college_biology": "stem", "college_chemistry": "stem", "college_computer_science": "stem",
    "college_mathematics": "stem", "college_physics": "stem", "computer_security": "stem",
    "conceptual_physics": "stem", "electrical_engineering": "stem", "elementary_mathematics": "stem",
    "high_school_biology": "stem", "high_school_chemistry": "stem",
    "high_school_computer_science": "stem", "high_school_mathematics": "stem",
    "high_school_physics": "stem", "high_school_statistics": "stem",
    "machine_learning": "stem",
    "formal_logic": "humanities", "high_school_european_history": "humanities",
    "high_school_us_history": "humanities", "high_school_world_history": "humanities",
    "international_law": "humanities", "jurisprudence": "humanities",
    "logical_fallacies": "humanities", "moral_disputes": "humanities",
    "moral_scenarios": "humanities", "philosophy": "humanities",
    "prehistory": "humanities", "professional_law": "humanities", "world_religions": "humanities",
    "econometrics": "social_sciences", "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences", "human_sexuality": "social_sciences",
    "professional_psychology": "social_sciences", "public_relations": "social_sciences",
    "security_studies": "social_sciences", "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "business_ethics": "other", "clinical_knowledge": "other", "college_medicine": "other",
    "global_facts": "other", "human_aging": "other", "management": "other",
    "marketing": "other", "medical_genetics": "other", "miscellaneous": "other",
    "nutrition": "other", "professional_accounting": "other", "professional_medicine": "other",
    "virology": "other",
}


def open_samples(p: Path):
    if p.suffix == ".gz":
        return gzip.open(p, "rt")
    return p.open("r")


def correct_from_record(rec: dict) -> int | None:
    """Best-effort extraction of correctness from an MMLU sample row."""
    # In lm-eval harness, MMLU samples typically include "acc" (1/0) directly
    if "acc" in rec:
        try:
            return int(bool(rec["acc"]))
        except Exception:
            pass
    # Fallback: compare argmax(logprobs) to "target"
    try:
        target = rec.get("target")
        resps = rec.get("filtered_resps") or rec.get("resps")
        if isinstance(target, int) and resps:
            # resps is list of [logprob, is_greedy] per choice
            scores = [r[0] if isinstance(r, list) else r for r in resps]
            pick = int(np.argmax(scores))
            return int(pick == target)
    except Exception:
        pass
    return None


SUBJECT_RE = re.compile(r"samples_mmlu_([a-z_]+)_\d{4}")


def discover_cells():
    out = []
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
                    out.append({
                        "model": model_dir.name,
                        "dataset": dataset_dir.name,
                        "method": method_dir.name,
                        "seed": seed_dir.name.replace("seed", ""),
                        "lm_eval_dir": seed_dir / "lm_eval",
                    })
    return out


def gather_subject_acc(cell: dict) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = defaultdict(list)
    if not cell["lm_eval_dir"].exists():
        return {}
    for sf in cell["lm_eval_dir"].rglob("samples_mmlu_*.jsonl*"):
        m = SUBJECT_RE.match(sf.name)
        if not m:
            continue
        subj = m.group(1)
        try:
            with open_samples(sf) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    c = correct_from_record(rec)
                    if c is not None:
                        out[subj].append(c)
        except Exception:
            pass
    return {s: (sum(v), len(v)) for s, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=Path, default=SUMMARY / "mmlu_per_domain.csv")
    args = ap.parse_args()

    cells = discover_cells()
    rows = []
    by_md: dict[tuple[str, str], dict[str, dict[str, tuple[int, int]]]] = defaultdict(dict)
    for c in cells:
        subj_acc = gather_subject_acc(c)
        if not subj_acc:
            continue
        # aggregate to domains
        domain_correct: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for subj, (corr, n) in subj_acc.items():
            dom = SUBCATEGORIES.get(subj, "unknown")
            domain_correct[dom][0] += corr
            domain_correct[dom][1] += n
        for dom, (corr, n) in domain_correct.items():
            acc = corr / n if n else None
            rows.append({
                "model": c["model"], "dataset": c["dataset"],
                "method": c["method"], "seed": c["seed"],
                "domain": dom, "n": n, "correct": corr,
                "acc": float(acc) if acc is not None else None,
            })
        by_md[(c["model"], c["dataset"])][c["method"]] = domain_correct

    if not rows:
        print("no MMLU samples_mmlu_*.jsonl found; rerun lm-eval with --log_samples to enable")
        return

    import csv
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "method", "seed",
                                          "domain", "n", "correct", "acc"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

    # bar chart per (model, dataset)
    domains = ["stem", "humanities", "social_sciences", "other"]
    for (model, dataset), per_method in sorted(by_md.items()):
        names = sorted(per_method.keys())
        if not names:
            continue
        x = np.arange(len(domains))
        width = 0.8 / max(1, len(names))
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(names) + 4), 4))
        for i, m in enumerate(names):
            vals = []
            for dom in domains:
                if dom in per_method[m]:
                    corr, n = per_method[m][dom]
                    vals.append(corr / n if n else 0)
                else:
                    vals.append(0)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        ax.set_ylabel("accuracy")
        ax.set_title(f"MMLU per domain — {model} / {dataset}")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        out = FIGS / f"{model}_{dataset}_mmlu_domain.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
