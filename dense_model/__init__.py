"""
Dense Autoregressive Transformer Model package.
Exports MathGPT and related architectural components.
"""

from dense_model.model import (
    GPTConfig,
    MathGPT,
    Block,
    MLP,
    CausalSelfAttention,
    precompute_rope_freqs,
    apply_rope,
    precompute_2d_rope_freqs,
    apply_2d_rope,
)

__all__ = [
    "GPTConfig",
    "MathGPT",
    "Block",
    "MLP",
    "CausalSelfAttention",
    "precompute_rope_freqs",
    "apply_rope",
    "precompute_2d_rope_freqs",
    "apply_2d_rope",
]
