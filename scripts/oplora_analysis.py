#!/usr/bin/env python3
"""OPLoRA-style offline subspace analysis (X-1 + X-2).

X-1 (per (model, dataset, method, layer)):
  rho_k(W0, dW; k) = || (I - U_k U_k^T) dW ||_F / || dW ||_F     # 'novelty' (high = away from W0 top-k)
  subspace_overlap_left(W0, dW; k)  = || U_k_W0^T U_k_dW ||_F^2 / k    in [0,1]
  subspace_overlap_right(W0, dW; k) = || V_k_W0^T V_k_dW ||_F^2 / k    in [0,1]

X-2 (per layer, sliding-window via checkpoints/step_*):
  per_window_drift[w] = 1 - subspace_overlap_left(dW_w, dW_{w+1}; k_default)

Usage:
  python scripts/oplora_analysis.py \
      --model qwen3-1p7b \
      --dataset tulu3-sft \
      --method dora \
      --base /mnt/cpfs/junlongke/onlinelora/models/qwen3-1p7b \
      --seed_dir results/stage3_v2/qwen3-1p7b/tulu3-sft/dora/seed42 \
      --out_dir analysis/oplora/jsons

  # Batch run all (model, method) combos using --auto_run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

K_LIST = [8, 16, 32, 64, 128]
K_DEFAULT_FOR_DRIFT = 32


def load_base_weight(base_dir: Path, layer_name: str) -> Optional[torch.Tensor]:
    """Load a single weight from sharded base model. Search all *.safetensors shards.

    Handles vanilla Qwen3 ("model.layers.X...") and Qwen3.5 VLM packaging
    where the LM weights are nested under "model.language_model.layers.X...".
    """
    candidates = [layer_name + ".weight"]
    if layer_name.startswith("model.layers."):
        # Qwen3.5 VLM-style packaging
        candidates.append(layer_name.replace("model.layers.", "model.language_model.layers.", 1) + ".weight")
    for shard in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            shard_keys = set(f.keys())
            for ck in candidates:
                if ck in shard_keys:
                    return f.get_tensor(ck).to(torch.float32)
    return None


def load_lora_dw(adapter_dir: Path, layer_name: str, scaling: float) -> Optional[torch.Tensor]:
    """Read lora_A, lora_B (and optionally magnitude vector for DoRA), return ΔW = scaling * B @ A.

    For DoRA: peft saves an extra `magnitude_vector` per adapter. Effective merged weight is
    m * (W0 + ΔW) / ||W0 + ΔW||_2,col, but for subspace analysis we just consider the additive
    ΔW = scaling * B @ A which spans the same column subspace as the DoRA correction modulo
    a diagonal column-wise rescale (which preserves left-singular subspace exactly and almost
    preserves right-singular subspace at top-k for k << min(d_in,d_out)).
    """
    fp = adapter_dir / "adapter_model.safetensors"
    if not fp.exists():
        return None
    A = None; B = None
    candidates_A = [
        f"base_model.model.{layer_name}.lora_A.weight",
        f"base_model.model.{layer_name}.lora_A.default.weight",
    ]
    candidates_B = [
        f"base_model.model.{layer_name}.lora_B.weight",
        f"base_model.model.{layer_name}.lora_B.default.weight",
    ]
    with safe_open(str(fp), framework="pt") as f:
        keys = set(f.keys())
        for ck in candidates_A:
            if ck in keys:
                A = f.get_tensor(ck).to(torch.float32); break
        for ck in candidates_B:
            if ck in keys:
                B = f.get_tensor(ck).to(torch.float32); break
    if A is None or B is None:
        return None
    return scaling * (B @ A)


def list_lora_layers(adapter_dir: Path) -> List[str]:
    """Return canonical layer names like 'model.layers.5.self_attn.q_proj'."""
    fp = adapter_dir / "adapter_model.safetensors"
    out = []
    with safe_open(str(fp), framework="pt") as f:
        for k in f.keys():
            if ".lora_A" in k:
                # base_model.model.<canonical>.lora_A[.default].weight
                canon = k.replace("base_model.model.", "").split(".lora_A")[0]
                out.append(canon)
    return sorted(set(out))


def topk_basis(W: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k left (U) and right (V) singular vectors of W. W: [d_out, d_in]."""
    # Use truncated SVD via torch.linalg.svd with full_matrices=False then slice.
    # For very wide matrices this is fine for our shapes (d ~ 2k-8k).
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U[:, :k], Vh[:k, :].T  # V is [d_in, k]


def subspace_overlap(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """||U1^T U2||_F^2 / k where k = U1.shape[1] = U2.shape[1]."""
    k = U1.shape[1]
    M = U1.T @ U2
    return (M.float().pow(2).sum() / k).item()


def rho_k(W0: torch.Tensor, dW: torch.Tensor, k: int) -> float:
    """Fraction of ΔW outside top-k left subspace of W0."""
    Uk, _ = topk_basis(W0, k)
    proj_into = Uk @ (Uk.T @ dW)
    perp = dW - proj_into
    denom = dW.norm().item()
    if denom == 0:
        return 0.0
    return (perp.norm() / dW.norm()).item()


def analyze_layer(W0: torch.Tensor, dW: torch.Tensor, ks: List[int]) -> dict:
    res = {"rho_k": {}, "subspace_overlap_left": {}, "subspace_overlap_right": {}}
    # Precompute SVD of W0 and dW once at max k for efficiency
    k_max = max(ks)
    k_max_W0 = min(k_max, min(W0.shape))
    k_max_dW = min(k_max, min(dW.shape))
    Uw0, Sw0, Vhw0 = torch.linalg.svd(W0, full_matrices=False)
    Udw, Sdw, Vhdw = torch.linalg.svd(dW, full_matrices=False)
    Vw0 = Vhw0.T; Vdw = Vhdw.T
    for k in ks:
        kk = min(k, min(W0.shape), min(dW.shape))
        Uk_w0 = Uw0[:, :kk]; Vk_w0 = Vw0[:, :kk]
        Uk_dw = Udw[:, :kk]; Vk_dw = Vdw[:, :kk]
        # rho_k via the precomputed Uk_w0
        proj = Uk_w0 @ (Uk_w0.T @ dW)
        perp = dW - proj
        denom = dW.norm().item()
        res["rho_k"][k] = (perp.norm() / dW.norm()).item() if denom > 0 else 0.0
        res["subspace_overlap_left"][k] = subspace_overlap(Uk_w0, Uk_dw)
        res["subspace_overlap_right"][k] = subspace_overlap(Vk_w0, Vk_dw)
    return res


def get_scaling(adapter_cfg: dict) -> float:
    r = adapter_cfg.get("r", 16)
    alpha = adapter_cfg.get("lora_alpha", 32)
    use_rslora = adapter_cfg.get("use_rslora", False)
    if use_rslora:
        return alpha / (r ** 0.5)
    return alpha / r


def list_window_ckpts(seed_dir: Path) -> List[Path]:
    """Return sorted list of step_NNN directories that contain adapter weights."""
    ck_dir = seed_dir / "checkpoints"
    if not ck_dir.exists():
        return []
    out = []
    for p in sorted(ck_dir.glob("step_*")):
        if (p / "adapter_model.safetensors").exists():
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--base", required=True, help="Base model dir (full pretrained weights)")
    ap.add_argument("--seed_dir", required=True, help=".../seedNN dir containing adapter/ + checkpoints/")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_layers", type=int, default=0,
                    help="If >0, limit analysis to first N layers (debug).")
    ap.add_argument("--n_windows", type=int, default=4,
                    help="Number of equally-spaced checkpoint windows for X-2 drift.")
    ap.add_argument("--skip_x2", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base)
    seed_dir = Path(args.seed_dir)
    adapter_dir = seed_dir / "adapter"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = get_scaling(cfg)
    use_dora = cfg.get("use_dora", False)
    print(f"[oplora] {args.model}/{args.dataset}/{args.method}: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"scaling={scaling:.4f} use_dora={use_dora}", flush=True)

    layers = list_lora_layers(adapter_dir)
    if args.limit_layers > 0:
        layers = layers[: args.limit_layers]
    print(f"[oplora] {len(layers)} target layers", flush=True)

    # X-1: final adapter
    out_records = []
    t0 = time.time()
    for i, ln in enumerate(layers):
        W0 = load_base_weight(base_dir, ln)
        if W0 is None:
            print(f"  WARN no base weight for {ln}", flush=True); continue
        dW = load_lora_dw(adapter_dir, ln, scaling)
        if dW is None:
            print(f"  WARN no lora deltaW for {ln}", flush=True); continue
        # ΔW shape from peft: B=[d_out, r], A=[r, d_in] → ΔW=[d_out, d_in]; W0=[d_out, d_in]
        # Sanity: shapes must match
        if dW.shape != W0.shape:
            print(f"  WARN shape mismatch {ln}: W0={W0.shape} dW={dW.shape}", flush=True); continue
        rec = {
            "layer": ln,
            **analyze_layer(W0, dW, K_LIST),
        }
        out_records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(layers)}] elapsed={time.time()-t0:.1f}s", flush=True)

    # X-2: per-window drift using checkpoint sequence, equispaced
    if not args.skip_x2:
        ckpts = list_window_ckpts(seed_dir)
        # Subsample to n_windows points (roughly equispaced, including the last)
        if len(ckpts) >= 2:
            n = min(args.n_windows + 1, len(ckpts))
            idxs = [int(round(i * (len(ckpts) - 1) / (n - 1))) for i in range(n)]
            idxs = sorted(set(idxs))
            window_ckpts = [ckpts[i] for i in idxs]
            print(f"[oplora] X-2 drift across {len(window_ckpts)} window points: "
                  f"{[p.name for p in window_ckpts]}", flush=True)
            for rec in out_records:
                ln = rec["layer"]
                drifts = []
                prev_dW = None
                for wp in window_ckpts:
                    dW_w = load_lora_dw(wp, ln, scaling)
                    if dW_w is None:
                        drifts.append(None); prev_dW = None; continue
                    if prev_dW is not None and dW_w.shape == prev_dW.shape:
                        kk = min(K_DEFAULT_FOR_DRIFT, min(prev_dW.shape), min(dW_w.shape))
                        U_prev, _, _ = torch.linalg.svd(prev_dW, full_matrices=False)
                        U_curr, _, _ = torch.linalg.svd(dW_w, full_matrices=False)
                        ov = subspace_overlap(U_prev[:, :kk], U_curr[:, :kk])
                        drifts.append(1.0 - ov)
                    prev_dW = dW_w
                rec["per_window_drift"] = [d for d in drifts if d is not None]
                rec["window_ckpts"] = [p.name for p in window_ckpts]

    # Emit JSON (one file per (model, dataset, method))
    out_path = out_dir / f"{args.model}__{args.dataset}__{args.method}.json"
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "method": args.method,
        "use_dora": use_dora,
        "scaling": scaling,
        "layers": out_records,
        "k_list": K_LIST,
        "k_drift": K_DEFAULT_FOR_DRIFT,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[oplora] wrote {out_path}  ({len(out_records)} layers)  elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
