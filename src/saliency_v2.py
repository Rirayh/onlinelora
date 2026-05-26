"""Saliency v2: per-sample, IG, t-stat, Fisher x signvote (PI v2 directive 2026-05-26).

Imports `LoraHandle` from `.model`; mirrors `src/saliency.py` API style but
returns per-sample tensors (n_samples, r) instead of aggregated (r,).

DOES NOT modify v1 (`src/saliency.py`) — v1 is kept reproducible per directive.

Estimators:
  first_order_saliency_per_sample(model, handles, loader, device, max_samples)
      -> dict[layer_name -> Tensor(n_samples, r)] of <gA_x, A> per sample, signed.

  integrated_gradient_saliency_per_sample(model, handles, loader, device, m=4, ...)
      -> dict[layer_name -> Tensor(m * max_samples, r)]. Replaces B with t*B at
         m equispaced points t in {1/m, 2/m, ..., 1.0} (skips t=0 to avoid the
         degenerate B=0 endpoint where grad_A is identically zero), runs
         per-sample saliency, concatenates along sample axis.

Decision functions:
  t_stat_decision(per_sample, alpha=0.1, rng_seed=0)
      Per-component Welch-ish t-statistic (one-sample test of mean = 0) with
      Benjamini-Hochberg FDR control across (layer, comp) pairs. Significant
      negatives -> KEEP, significant positives -> DROP, non-significant ->
      Bernoulli(0.5) random.
      Returns dict[layer -> bool tensor(r,)] keep mask.

  fisher_signvote_score(per_sample)
      Per-component score = sign_vote * sqrt(fisher).
      Returns dict[layer -> tensor(r,)] (signed).

References:
- Sundararajan et al, "Axiomatic Attribution for Deep Networks", 2017 (IG).
- Benjamini & Hochberg, "Controlling the False Discovery Rate", 1995.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .model import LoraHandle


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def first_order_saliency_per_sample(
    model: nn.Module,
    handles: list[LoraHandle],
    loader: Any,
    device: torch.device,
    max_samples: int = 256,
    signed: bool = True,
) -> dict[str, torch.Tensor]:
    """Per-sample first-order saliency. Returns dict[layer -> Tensor(n_samples, r)] signed."""
    model.eval()
    model.zero_grad(set_to_none=True)
    out: dict[str, list[torch.Tensor]] = {h.name: [] for h in handles}
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
            single = {k: v[j:j + 1] for k, v in batch.items()}
            o = model(**single)
            o.loss.backward()
            for h in handles:
                if h.A.grad is None:
                    out[h.name].append(torch.zeros(h.r))
                    continue
                gA = h.A.grad.detach()
                A = h.A.detach()
                v = (gA * A).sum(dim=1)  # (r,)
                if not signed:
                    v = v.abs()
                out[h.name].append(v.cpu())
            model.zero_grad(set_to_none=True)
            n += 1
    return {k: torch.stack(v, dim=0) if v else torch.zeros(0, h.r)
            for h, (k, v) in zip(handles, out.items())}


def integrated_gradient_saliency_per_sample(
    model: nn.Module,
    handles: list[LoraHandle],
    loader: Any,
    device: torch.device,
    m: int = 4,
    max_samples: int = 256,
    signed: bool = True,
) -> dict[str, torch.Tensor]:
    """IG over t in {1/m, ..., 1}. B is replaced by t*B_orig at each step.

    Returns dict[layer_name -> Tensor(m * max_samples, r)] (concatenated along sample axis).
    """
    B_orig = {h.name: h.B.detach().clone() for h in handles}
    all_records: dict[str, list[torch.Tensor]] = {h.name: [] for h in handles}
    ts = torch.linspace(0.0, 1.0, m + 1)[1:]  # skip t=0 (B=0 -> grad_A = 0)
    for t in ts:
        with torch.no_grad():
            for h in handles:
                h.B.data.copy_(B_orig[h.name] * float(t))
        rec = first_order_saliency_per_sample(
            model, handles, loader, device,
            max_samples=max_samples, signed=signed,
        )
        for k, v in rec.items():
            all_records[k].append(v)
    with torch.no_grad():
        for h in handles:
            h.B.data.copy_(B_orig[h.name])
    return {
        k: torch.cat(v, dim=0) if v else torch.zeros(0)
        for k, v in all_records.items()
    }


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.1) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns boolean array of rejected hypotheses."""
    n = len(pvals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    ranks = np.arange(1, n + 1)
    thresh = q * ranks / n
    below = sorted_p <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_k = int(np.max(np.where(below)[0]))
    cutoff = sorted_p[max_k]
    return pvals <= cutoff


def t_stat_decision(
    per_sample: dict[str, torch.Tensor],
    alpha: float = 0.1,
    rng_seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """One-sample t-test (mean=0) per (layer, comp) with global BH-FDR.

    Significant negative mean -> KEEP (component is helping, gradient pushes
                                       loss further DOWN when in direction).
    Significant positive mean -> DROP.
    Non-significant            -> Bernoulli(0.5) random.

    Returns (keep_masks, info) where keep_masks is dict[layer -> bool(r,)] and
    info is a dict with global stats for logging.
    """
    from scipy.stats import t as student_t
    rng = np.random.default_rng(rng_seed)
    layers = list(per_sample.keys())
    means_chunks, stds_chunks, n_chunks, sizes = [], [], [], []
    for L in layers:
        S = per_sample[L].numpy().astype(np.float64)
        if S.ndim != 2:
            # malformed; treat as single component vector
            S = S.reshape(-1, 1)
        n_s, r = S.shape
        sizes.append((L, r))
        if n_s < 2:
            mean = S.mean(axis=0) if n_s > 0 else np.zeros(r)
            std = np.full(r, 1e-8)
        else:
            mean = S.mean(axis=0)
            std = S.std(axis=0, ddof=1)
        std = np.where(std < 1e-8, 1e-8, std)
        means_chunks.append(mean)
        stds_chunks.append(std)
        n_chunks.append(np.full(r, max(n_s, 2)))
    if not means_chunks:
        return {}, {"n_total": 0, "n_keep_sig": 0, "n_drop_sig": 0, "n_random": 0}
    mean_flat = np.concatenate(means_chunks)
    std_flat = np.concatenate(stds_chunks)
    n_flat = np.concatenate(n_chunks)
    t_stat = mean_flat / (std_flat / np.sqrt(n_flat))
    pvals = 2.0 * (1.0 - student_t.cdf(np.abs(t_stat), df=n_flat - 1))
    pvals = np.clip(pvals, 0.0, 1.0)
    reject = benjamini_hochberg(pvals, q=alpha)
    keep_global = reject & (mean_flat < 0)
    drop_global = reject & (mean_flat > 0)
    mid = ~reject
    rand_keep = rng.random(len(mid)) < 0.5
    final_keep = keep_global | (mid & rand_keep)

    keep_masks: dict[str, torch.Tensor] = {}
    idx = 0
    for L, r in sizes:
        keep_masks[L] = torch.from_numpy(final_keep[idx:idx + r].copy()).bool()
        idx += r

    info = {
        "n_total": int(len(final_keep)),
        "n_keep_sig": int(keep_global.sum()),
        "n_drop_sig": int(drop_global.sum()),
        "n_random": int(mid.sum()),
        "n_random_keep": int((mid & rand_keep).sum()),
        "n_kept_total": int(final_keep.sum()),
        "alpha": float(alpha),
    }
    return keep_masks, info


def fisher_signvote_score(
    per_sample: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per-component score: sign_vote * sqrt(fisher). Returns dict[layer -> Tensor(r,)] signed."""
    out = {}
    for L, S in per_sample.items():
        if S.numel() == 0 or S.ndim != 2 or S.size(0) == 0:
            out[L] = torch.zeros(S.size(-1) if S.ndim >= 1 else 0)
            continue
        fisher = (S ** 2).mean(dim=0)
        sign_vote = S.sign().mean(dim=0)
        out[L] = sign_vote * torch.sqrt(fisher.clamp_min(1e-12))
    return out
