"""
Base module for algorithmic reasoning architectures.
Provides shared Rotary Position Embedding (RoPE) caches, RMSNorm,
and abstract base class BaseReasoningModel.
"""

import math
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int = 2048, theta_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes rotary frequency tables (cos, sin) for RoPE embeddings."""
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    freqs = 1.0 / (theta_base ** (dim_indices / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embeddings to query or key tensor of shape [B, n_head, T, head_dim]."""
    T = x.size(2)
    cos_t = cos[:T, :].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin_t = sin[:T, :].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rot_even = x_even * cos_t - x_odd * sin_t
    rot_odd = x_even * sin_t + x_odd * cos_t

    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for attractor dynamics and FP16 numerical stability."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class BaseReasoningModel(nn.Module):
    """Abstract base class for Phase 1 and Phase 2 algorithmic reasoning models."""

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns total non-embedding parameter count to verify <20M ceiling."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive greedy or sampling generation loop."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx, **kwargs)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            next_token = idx_next.item()
            if eos_id is not None and next_token == eos_id:
                break
            if stop_tokens is not None and next_token in stop_tokens:
                break

        return idx
