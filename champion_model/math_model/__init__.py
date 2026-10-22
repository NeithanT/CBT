"""
Dense Autoregressive Transformer Model package (MathGPT).
Exports MathGPT and related architectural components.
"""

from champion_model.math_model.model import (
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
