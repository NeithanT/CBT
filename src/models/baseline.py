"""
Baseline MathGPT architecture wrapper for Phase 1 models.
"""

from src.model import (
    GPTConfig,
    MathGPT,
    Block,
    MLP,
    CausalSelfAttention,
)

__all__ = ["GPTConfig", "MathGPT", "Block", "MLP", "CausalSelfAttention"]
