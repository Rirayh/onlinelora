"""Unit tests for saliency formulas (handover §3.12).

These tests catch silent bugs in gradient sign conventions and the equivalence
of <grad_A, A> and <grad_B, B>. Both must agree numerically — see §3.4.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Helpers ---------------------------------------------------------------

class TinyLoRALinear(nn.Module):
    """A linear layer with a LoRA delta exposed as explicit A (r, in) and B (out, r).

    Forward:  y = x @ W_0^T + scaling * x @ A^T @ B^T = x @ W_0^T + scaling * (B A) x^T (transposed view)
    Concretely: y_i = x W_0^T + scaling * x A^T B^T  (i.e. delta is x @ (BA)^T).
    """
    def __init__(self, in_features: int, out_features: int, r: int = 4, scaling: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W0 = nn.Parameter(torch.randn(out_features, in_features, generator=g, dtype=torch.float64) * 0.05, requires_grad=False)
        self.A = nn.Parameter(torch.randn(r, in_features, generator=g, dtype=torch.float64) * 0.05)
        self.B = nn.Parameter(torch.randn(out_features, r, generator=g, dtype=torch.float64) * 0.05)
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in)
        base = x @ self.W0.t()                       # (N, out)
        delta = (x @ self.A.t()) @ self.B.t() * self.scaling  # (N, r) @ (r, out)
        return base + delta


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


# ---- Tests -----------------------------------------------------------------

def test_first_order_identity() -> None:
    """The two equivalent forms must agree numerically.

    For loss L and LoRA delta = scaling * B A, with G = dL/d(delta):
        <grad_A L, A>_{row i} =  scaling * <B^T G, A>_{row i}
        <grad_B L, B>_{col i} =  scaling * <G A^T, B>_{col i}
    Both formulas reduce to scaling * <G, b_i a_i^T> = saliency_i (signed).
    """
    _seed(7)
    in_f, out_f, r = 16, 8, 4
    model = TinyLoRALinear(in_f, out_f, r=r, scaling=1.0, seed=7)
    x = torch.randn(32, in_f, dtype=torch.float64)
    y_true = torch.randn(32, out_f, dtype=torch.float64)
    out = model(x)
    loss = F.mse_loss(out, y_true)
    grads = torch.autograd.grad(loss, [model.A, model.B], create_graph=False)
    gA, gB = grads
    # Form 1: <gA[i,:], A[i,:]>
    s_via_A = (gA * model.A.detach()).sum(dim=1)        # (r,)
    # Form 2: <gB[:,i], B[:,i]>
    s_via_B = (gB * model.B.detach()).sum(dim=0)        # (r,)
    diff = (s_via_A - s_via_B).abs().max().item()
    assert diff < 1e-5, f"FO identity failed: max abs diff = {diff:.3e}"
    print(f"  OK test_first_order_identity: max|diff| = {diff:.3e}")


def test_zeroing_component_matches_saliency_sign() -> None:
    """Sign of signed first-order saliency must agree with the direction of loss change
    when the component is shrunk, for SMALL perturbations (first-order regime).

    Derivation: with delta = B A (scaling=1) and G = dL/d(delta),
        s_i  := -<G, b_i a_i^T>              (handover convention)
              = -<grad_A L [i,:], A[i,:]>
    If we replace b_i by (1-eps)*b_i, then d(delta)_i = -eps * b_i a_i^T and
        dL = <G, -eps b_i a_i^T> = -eps * <G, b_i a_i^T> = +eps * s_i
    So dL_pred = +eps * s_signed[i] with our s_signed = -<grad_A, A> = s_i.
    """
    _seed(11)
    in_f, out_f, r = 12, 6, 3
    model = TinyLoRALinear(in_f, out_f, r=r, scaling=1.0, seed=11)
    x = torch.randn(64, in_f, dtype=torch.float64)
    y_true = torch.randn(64, out_f, dtype=torch.float64)

    # Baseline loss + gradients
    out = model(x)
    L0 = F.mse_loss(out, y_true)
    grads = torch.autograd.grad(L0, [model.A, model.B], create_graph=False)
    gA, gB = grads
    # s_signed (handover convention): s_i = - <grad_{delta}, b_i a_i^T> = - <gA[i,:], A[i,:]>
    s_signed = -(gA * model.A.detach()).sum(dim=1)  # (r,)
    L0_val = float(L0.item())

    eps = 1e-5  # small enough to make 2nd-order term negligible compared to 1st-order
    for i in range(r):
        with torch.no_grad():
            B_col = model.B[:, i].clone()
            A_row = model.A[i, :].clone()
            # Shrink comp i by (1 - eps): equivalent to delta_i_new = (1-eps) * b_i a_i^T
            model.B[:, i].mul_(1.0 - eps)
            # A unchanged is enough because b_i scales jointly. (Equivalent: scale A row.)
        out_new = model(x)
        L_new = float(F.mse_loss(out_new, y_true).item())
        with torch.no_grad():
            model.B[:, i].copy_(B_col)
            model.A[i, :].copy_(A_row)

        # Predicted: dL ≈ +eps * s_signed[i]
        dL_pred = +eps * float(s_signed[i].item())
        dL_actual = L_new - L0_val
        # Should match to second-order; allow modest tolerance relative to magnitude.
        # Sign agreement is the strict check.
        if abs(dL_actual) > 1e-12:
            assert (dL_pred * dL_actual) >= -1e-12, \
                f"sign disagreement at i={i}: pred={dL_pred:.3e} actual={dL_actual:.3e}"
        # Relative error check; tolerance is generous because actual changes are O(eps*s)
        # and we have finite-precision arithmetic when L is in the ~1e-1 regime.
        denom = max(abs(dL_actual), 1e-14)
        rel = abs(dL_pred - dL_actual) / denom
        assert rel < 0.1, f"i={i}: |pred-actual|/|actual| = {rel:.3f} too large (pred={dL_pred:.3e}, actual={dL_actual:.3e})"
        print(f"  OK comp {i}: pred={dL_pred:+.3e}  actual={dL_actual:+.3e}  rel={rel:.3f}")
    print("  OK test_zeroing_component_matches_saliency_sign")


def test_magnitude_saliency_shape() -> None:
    _seed(3)
    in_f, out_f, r = 10, 5, 4
    model = TinyLoRALinear(in_f, out_f, r=r, seed=3)
    # ||b_i||*||a_i|| per component i
    a_norm = model.A.detach().norm(dim=1)   # (r,)
    b_norm = model.B.detach().norm(dim=0)   # (r,)
    mag = a_norm * b_norm
    assert mag.shape == (r,)
    assert (mag > 0).all()
    print(f"  OK test_magnitude_saliency_shape: mag = {mag.tolist()}")


if __name__ == "__main__":
    test_first_order_identity()
    test_zeroing_component_matches_saliency_sign()
    test_magnitude_saliency_shape()
    print("\nAll saliency tests passed.")
