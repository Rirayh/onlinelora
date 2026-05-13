"""Small LLaMA-style decoder-only LM for Stage 2 (ReLoRA failure reproduction).

Hand-built (no torchtitan dependency). Configurations:
  11M : hidden=192,  n_layers=8,  n_heads=6,  ffn_mult=4   (~11M with 50k vocab)
  33M : hidden=384,  n_layers=8,  n_heads=8,  ffn_mult=4   (~33M)
  66M : hidden=512,  n_layers=12, n_heads=8,  ffn_mult=4   (~66M)

Uses pre-norm RMSNorm + RoPE + SwiGLU MLP, weight-tied embedding. Standard recipe.

For Stage 2 we wrap query/value projections of each attention block with LoRA via peft.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMConfig:
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    ffn_mult: int = 4
    max_seq_len: int = 256
    rope_base: float = 10000.0
    tie_embeddings: bool = True


SIZE_CONFIGS = {
    "11M": dict(hidden_size=192, n_layers=8,  n_heads=6, ffn_mult=4, max_seq_len=256),
    "33M": dict(hidden_size=384, n_layers=8,  n_heads=8, ffn_mult=4, max_seq_len=256),
    "66M": dict(hidden_size=512, n_layers=12, n_heads=8, ffn_mult=4, max_seq_len=256),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm) * self.weight


def precompute_rope(dim: int, max_len: int, base: float = 10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.stack((rx1, rx2), dim=-1).flatten(-2)
    return out


class Attention(nn.Module):
    """Standard MHA with q/v as separate Linear modules so peft can wrap them."""
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden_size // cfg.n_heads
        assert self.head_dim * cfg.n_heads == cfg.hidden_size
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.key   = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.query(x).view(B, T, H, D).transpose(1, 2)
        k = self.key(x).view(B, T, H, D).transpose(1, 2)
        v = self.value(x).view(B, T, H, D).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        inner = cfg.hidden_size * cfg.ffn_mult
        # round to multiple of 64
        inner = (inner // 64) * 64
        self.w1 = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.w2 = nn.Linear(inner, cfg.hidden_size, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, inner, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """Decoder-only LM with LLaMA-style blocks, weight-tied lm_head."""
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, T: int, device):
        head_dim = self.cfg.hidden_size // self.cfg.n_heads
        if self._rope_cos.numel() == 0 or self._rope_cos.shape[0] < T or self._rope_cos.device != device:
            cos, sin = precompute_rope(head_dim, max(T, self.cfg.max_seq_len),
                                       self.cfg.rope_base, device=device)
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos[:T], self._rope_sin[:T]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # accepted, ignored (causal)
    ):
        B, T = input_ids.shape
        cos, sin = self._rope(T, input_ids.device)
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # shift-by-one causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # mimic HF output container
        class _Out:
            pass
        o = _Out()
        o.loss = loss
        o.logits = logits
        return o


def build_tiny_lm(size: str, vocab_size: int) -> TinyLM:
    if size not in SIZE_CONFIGS:
        raise ValueError(f"unknown size {size}; valid: {list(SIZE_CONFIGS)}")
    cfg = LMConfig(vocab_size=vocab_size, **SIZE_CONFIGS[size])
    return TinyLM(cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_params_no_embed(model: TinyLM) -> int:
    total = count_params(model)
    embed = model.embed.weight.numel()
    if model.cfg.tie_embeddings:
        return total - embed   # tied: lm_head not separate
    return total - 2 * embed
