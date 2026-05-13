"""Oracle ablation: zero out one rank-1 LoRA component, measure delta test loss."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import LoraHandle


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_examples: int | None = None,
) -> tuple[float, float]:
    """Return (mean_loss, accuracy) over the loader.

    accuracy is computed via argmax for classification. If the head produces logits with
    shape (N, num_labels), this is multi-class accuracy.
    """
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    for batch in loader:
        if max_examples is not None and n >= max_examples:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        bsz = batch["input_ids"].size(0)
        total_loss += float(out.loss.item()) * bsz
        if hasattr(out, "logits") and out.logits is not None and "labels" in batch:
            pred = out.logits.argmax(dim=-1)
            correct += int((pred == batch["labels"]).sum().item())
        n += bsz
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, correct / n


@torch.no_grad()
def oracle_ablation(
    model: nn.Module,
    handles: list[LoraHandle],
    test_loader: DataLoader,
    device: torch.device,
    baseline_loss: float,
    max_test_examples: int | None = None,
) -> list[dict[str, Any]]:
    """For every (layer, comp), zero it, measure test loss, restore.

    Returns list of records: layer, comp, delta_test, loss_after.
    """
    model.eval()
    rows: list[dict[str, Any]] = []
    for h in handles:
        for i in range(h.r):
            B_col = h.B[:, i].detach().clone()
            A_row = h.A[i, :].detach().clone()
            h.B[:, i].zero_()
            h.A[i, :].zero_()
            loss_after, _ = evaluate(model, test_loader, device, max_examples=max_test_examples)
            h.B[:, i].copy_(B_col)
            h.A[i, :].copy_(A_row)
            rows.append({
                "layer": h.name,
                "comp": i,
                "delta_test": float(loss_after - baseline_loss),
                "loss_after": float(loss_after),
            })
    return rows
