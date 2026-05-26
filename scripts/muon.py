"""Muon optimizer (vendored from KellerJordan/Muon).

Source: https://github.com/KellerJordan/Muon (MIT License)
Vendored 2026-05-26 for PI Muon-decoupling experiment (Exp-2).

Muon = MomentUm Orthogonalized by Newton-Schulz.
Use Muon for 2D weight matrices; use AdamW for 1D params, embeddings, head.

Critical for ReLoRA / LoRA: lora_B is initialised to zero. A zero-norm matrix
fed to Newton-Schulz produces NaN. We detect that case in `step()` and fall back
to a plain SGD-momentum update (no orthogonalization) for that single step.

Reference: Keller Jordan, "Modded NanoGPT speedrun" (2024).
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to orthogonalize G.

    Coefficients (3.4445, -4.7750, 2.0315) tuned by Keller Jordan to maximize
    convergence at small input norms. Operates in fp32 for stability.

    NOTE: We deliberately do NOT @torch.compile this function. ReLoRA training
    has 252 LoRA layers × 2 matrices each, with several distinct shapes, plus
    optimizer rebuilds at every merge event. Per-shape torch.compile recompile
    cost dominates training time. Eager mode is fast enough on A100.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon optimizer (orthogonalized momentum).

    Args:
      params: iterable of 2D parameters (e.g. LoRA A/B matrices).
      lr:     learning rate.
      momentum: SGD momentum (default 0.95 per Keller).
      nesterov: use Nesterov momentum (default True).
      ns_steps: Newton-Schulz iterations (default 5).
      weight_decay: decoupled weight decay (default 0.0).
      zero_grad_eps: threshold below which a parameter is treated as zero
        (skip NS, use plain momentum step). Used for first step on zero-init B.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0, zero_grad_eps: float = 1e-12):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        zero_grad_eps=zero_grad_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            eps_zero = group["zero_grad_eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    # Should be filtered out by caller but guard anyway.
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                gp = g.add(buf, alpha=momentum) if nesterov else buf
                # Zero-norm guard: if gradient is essentially zero (e.g. first
                # step with B = 0 producing 0 grad somewhere downstream), skip
                # NS to avoid 0/0 → NaN.
                gnorm = gp.norm().item()
                if gnorm < eps_zero:
                    # plain momentum step (no orthogonalization)
                    update = gp
                else:
                    update = zeropower_via_newtonschulz5(gp, steps=ns_steps)
                    # Scale update to keep effective step size comparable to AdamW.
                    # Per Keller: scale by max(1, sqrt(rows/cols)).
                    scale = max(1.0, (gp.size(0) / gp.size(1)) ** 0.5)
                    update = update * scale
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


def split_params_for_muon(named_params):
    """Partition named parameters into (muon_params, adamw_params).

    Muon: 2D LoRA weight matrices (`lora_A`, `lora_B`).
    AdamW: everything else (1D, embeddings, head, scaling, biases).
    """
    muon_params, adamw_params = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        # Heuristic: route 2D LoRA matrices to Muon.
        is_lora_2d = (p.ndim == 2) and (("lora_A" in name) or ("lora_B" in name))
        if is_lora_2d:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
