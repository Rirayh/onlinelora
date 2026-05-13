"""Saliency formulas for LoRA rank-1 components.

For each LoRA layer with B in R^{d_out x r}, A in R^{r x d_in},
    Delta W = scaling * B A = scaling * sum_i b_i a_i^T

Identity (handover §3.4): with G = grad(L)/grad(Delta W),
    s_i^FO = -<G, scaling * b_i a_i^T>
           = -<grad_A L [i, :], A[i, :]>     (note: grad_A absorbs scaling*B^T G)
           = -<grad_B L [:, i], B[:, i]>     (similarly absorbs scaling*G A^T)

So the two equivalent forms differ from each other only by floating-point noise.

Saliency variants (all return dict[layer_name] -> tensor of shape (r,)):
  - S1 magnitude:        ||b_i|| * ||a_i||                                  (no signal)
  - S2 first_order_train: |<grad_A^{train} L [i,:], A[i,:]>|                (sign-stripped)
  - S3 first_order_val:   |<grad_A^{val}   L [i,:], A[i,:]>|                (sign-stripped)
  - S3 first_order_val_signed: <grad_A^{val} L [i,:], A[i,:]>               (signed)
  - S4 fisher_train:      E_{x ~ train}[ <grad_A L_x [i,:], A[i,:]>^2 ]
  - S5 fisher_val:        E_{x ~ val}  [ <grad_A L_x [i,:], A[i,:]>^2 ]
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .model import LoraHandle


@torch.no_grad()
def magnitude_saliency(handles: list[LoraHandle]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for h in handles:
        a_norm = h.A.detach().norm(dim=1)          # ||a_i||  shape (r,)
        b_norm = h.B.detach().norm(dim=0)          # ||b_i||  shape (r,)
        out[h.name] = (a_norm * b_norm).cpu()
    return out


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def first_order_saliency(
    model: nn.Module,
    handles: list[LoraHandle],
    loader: Any,
    device: torch.device,
    max_batches: int | None = None,
    signed: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute aggregated first-order saliency over a loader.

    Implementation: accumulate `loss.backward()` over `max_batches` batches in
    a single graph by `loss = sum(per_batch_loss) / total_examples`. This
    matches an average-gradient definition exactly. (Equivalent to calling
    backward in a loop and summing grads — same numerical result up to
    floating-point.)
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    grads_A: dict[str, torch.Tensor] = {h.name: torch.zeros_like(h.A.detach()) for h in handles}
    n_examples = 0
    n_batches = 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = _move_batch(batch, device)
        # We are in eval mode but still need grads on LoRA params.
        for h in handles:
            h.A.requires_grad_(True)
            h.B.requires_grad_(True)
        out = model(**batch)
        bsz = batch["input_ids"].size(0)
        # Use sum (not mean) so that accumulation is over examples regardless of batch size.
        loss = out.loss * bsz
        loss.backward()
        for h in handles:
            if h.A.grad is not None:
                grads_A[h.name] += h.A.grad.detach()
        model.zero_grad(set_to_none=True)
        n_examples += bsz
        n_batches += 1

    saliency: dict[str, torch.Tensor] = {}
    for h in handles:
        if n_examples == 0:
            saliency[h.name] = torch.zeros(h.r)
            continue
        avg_grad = grads_A[h.name] / n_examples           # (r, in)
        A = h.A.detach()                                  # (r, in)
        per_comp = (avg_grad * A).sum(dim=1)              # (r,)
        if not signed:
            per_comp = per_comp.abs()
        saliency[h.name] = per_comp.detach().cpu()
    return saliency


def fisher_saliency(
    model: nn.Module,
    handles: list[LoraHandle],
    loader: Any,
    device: torch.device,
    max_samples: int = 256,
) -> dict[str, torch.Tensor]:
    """Per-sample squared inner-product (diagonal empirical Fisher analogue).

    For each example x: g_A = grad(L_x) wrt A; per-component score
        f_i = ( <g_A[i,:], A[i,:]> )^2
    Average over examples.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    fisher: dict[str, torch.Tensor] = {h.name: torch.zeros(h.r, device=device) for h in handles}
    n = 0
    for batch in loader:
        if n >= max_samples:
            break
        batch = _move_batch(batch, device)
        bsz = batch["input_ids"].size(0)
        for j in range(bsz):
            if n >= max_samples:
                break
            for h in handles:
                h.A.requires_grad_(True)
                h.B.requires_grad_(True)
            single = {k: v[j:j+1] for k, v in batch.items()}
            out = model(**single)
            out.loss.backward()
            for h in handles:
                if h.A.grad is None:
                    continue
                A_grad = h.A.grad.detach()        # (r, in)
                A = h.A.detach()                  # (r, in)
                per_comp = (A_grad * A).sum(dim=1)  # (r,)
                fisher[h.name] += per_comp.pow(2)
            model.zero_grad(set_to_none=True)
            n += 1

    if n == 0:
        return {h.name: torch.zeros(h.r) for h in handles}
    return {k: (v / n).cpu() for k, v in fisher.items()}


def saliency_dict_to_records(
    sal: dict[str, torch.Tensor],
    name: str,
) -> list[dict[str, Any]]:
    """Flatten {layer_name -> (r,)} to list of (layer, comp, name=value) records."""
    rows: list[dict[str, Any]] = []
    for layer, vec in sal.items():
        v = vec.detach().cpu().numpy()
        for i, val in enumerate(v):
            rows.append({"layer": layer, "comp": int(i), name: float(val)})
    return rows


def merge_records_by_key(
    *record_lists: list[dict[str, Any]],
    key_fields: tuple[str, ...] = ("layer", "comp"),
) -> list[dict[str, Any]]:
    merged: dict[tuple, dict[str, Any]] = {}
    for rl in record_lists:
        for r in rl:
            k = tuple(r[f] for f in key_fields)
            if k not in merged:
                merged[k] = {f: r[f] for f in key_fields}
            for kk, vv in r.items():
                if kk not in key_fields:
                    merged[k][kk] = vv
    return list(merged.values())
