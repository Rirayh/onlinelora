"""Model + LoRA wrapping. Exposes per-layer B,A weight handles for saliency/ablation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class LoraHandle:
    """Per-layer LoRA handle.

    Convention (peft default for Linear):
        lora_A.default.weight : shape (r, in_features)
        lora_B.default.weight : shape (out_features, r)
        merged update         : B @ A          shape (out, in)
        scaling factor        : lora_alpha / r (applied at forward)

    Notation in the handover: B in R^{d_out x r}, A in R^{r x d_in},
        Delta W = B A = sum_i b_i a_i^T  where b_i = B[:, i], a_i = A[i, :].
    """
    name: str
    A: nn.Parameter   # (r, in)
    B: nn.Parameter   # (out, r)
    scaling: float    # alpha / r
    r: int


def build_lora_model(
    model_name: str,
    num_labels: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> tuple[nn.Module, AutoTokenizer]:
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base, cfg)
    return model, tok


def get_lora_BA_handles(peft_model: nn.Module) -> list[LoraHandle]:
    """Walk modules, return all LoRA (A, B) pairs.

    peft's `LoraLayer` typically exposes a `lora_A` and `lora_B` `ModuleDict`
    keyed by adapter name (default: 'default'). Each value is an `nn.Linear`,
    so its `.weight` parameter is what we use.
    """
    handles: list[LoraHandle] = []
    seen: set[tuple[int, int]] = set()
    for name, mod in peft_model.named_modules():
        has_A = hasattr(mod, "lora_A") and isinstance(getattr(mod, "lora_A"), nn.ModuleDict)
        has_B = hasattr(mod, "lora_B") and isinstance(getattr(mod, "lora_B"), nn.ModuleDict)
        if not (has_A and has_B):
            continue
        for adapter_key in mod.lora_A.keys():
            A_mod = mod.lora_A[adapter_key]
            B_mod = mod.lora_B[adapter_key]
            if not (isinstance(A_mod, nn.Linear) and isinstance(B_mod, nn.Linear)):
                continue
            A_p = A_mod.weight  # (r, in)
            B_p = B_mod.weight  # (out, r)
            key = (id(A_p), id(B_p))  # tuple dedup is collision-proof
            if key in seen:
                continue
            seen.add(key)
            r = A_p.shape[0]
            alpha_attr = getattr(mod, "lora_alpha", None)
            if isinstance(alpha_attr, dict):
                alpha = alpha_attr.get(adapter_key, r)
            elif alpha_attr is None:
                alpha = r
            else:
                alpha = alpha_attr
            try:
                alpha = float(alpha)
            except (TypeError, ValueError):
                alpha = float(r)
            scaling = float(alpha) / float(r)
            handles.append(LoraHandle(
                name=f"{name}.{adapter_key}",
                A=A_p, B=B_p, scaling=scaling, r=r,
            ))
    return handles


def count_lora_components(handles: list[LoraHandle]) -> int:
    return sum(h.r for h in handles)
