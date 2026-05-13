"""Data loading + canonical 3-way split for Stage 1.

Splits per task:
  - original GLUE train -> train_main (80%) + diagnostic (20%, never seen by optimizer)
  - original GLUE validation -> test_holdout (sealed; only for oracle ablation + final eval)

See handover §3.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

TASK_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "qnli": ("question", "sentence"),
}


@dataclass
class Splits:
    train_main: Dataset
    diagnostic: Dataset
    test_holdout: Dataset


def load_glue_three_split(
    task: str,
    diagnostic_ratio: float = 0.2,
    seed: int = 42,
) -> Splits:
    if task not in TASK_KEYS:
        raise ValueError(f"unsupported task {task}")
    ds = load_dataset("glue", task)
    train_pool = ds["train"]
    test_holdout = ds["validation"]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_pool))
    n_diag = int(len(train_pool) * diagnostic_ratio)
    diag_idx = idx[:n_diag].tolist()
    main_idx = idx[n_diag:].tolist()

    diagnostic = train_pool.select(diag_idx)
    train_main = train_pool.select(main_idx)
    return Splits(train_main=train_main, diagnostic=diagnostic, test_holdout=test_holdout)


def tokenize_splits(
    splits: Splits,
    task: str,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int = 128,
) -> Splits:
    k1, k2 = TASK_KEYS[task]

    def tok(batch: dict[str, Any]) -> dict[str, Any]:
        if k2 is None:
            enc = tokenizer(batch[k1], truncation=True, max_length=max_len, padding="max_length")
        else:
            enc = tokenizer(
                batch[k1], batch[k2],
                truncation=True, max_length=max_len, padding="max_length",
            )
        enc["labels"] = batch["label"]
        return enc

    def map_ds(d: Dataset) -> Dataset:
        keep = ["input_ids", "attention_mask", "labels"]
        out = d.map(tok, batched=True, remove_columns=[c for c in d.column_names if c not in keep])
        out.set_format("torch", columns=keep)
        return out

    return Splits(
        train_main=map_ds(splits.train_main),
        diagnostic=map_ds(splits.diagnostic),
        test_holdout=map_ds(splits.test_holdout),
    )


def make_loaders(
    tok_splits: Splits,
    batch_size: int,
    eval_batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        tok_splits.train_main, batch_size=batch_size, shuffle=True, drop_last=True
    )
    diag_loader = DataLoader(
        tok_splits.diagnostic, batch_size=eval_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        tok_splits.test_holdout, batch_size=eval_batch_size, shuffle=False
    )
    return train_loader, diag_loader, test_loader
