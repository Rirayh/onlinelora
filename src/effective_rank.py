"""Effective rank and condition number (Roy & Vetterli 2007)."""
from __future__ import annotations

import math

import torch


def effective_rank(M: torch.Tensor, eps: float = 1e-10) -> float:
    """exp( entropy of normalized singular value distribution )."""
    s = torch.linalg.svdvals(M.float())
    total = float(s.sum().item()) + eps
    p = s / total
    H = -(p * (p + eps).log()).sum().item()
    return math.exp(H)


def condition_number(M: torch.Tensor, eps: float = 1e-10) -> float:
    s = torch.linalg.svdvals(M.float())
    s_max = float(s.max().item())
    nonzero = s[s > eps]
    if nonzero.numel() == 0:
        return float("inf")
    s_min = float(nonzero.min().item())
    return s_max / max(s_min, eps)
