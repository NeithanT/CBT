"""
Champion Model Package: Character-Based Transformer (CBT) and Recurrent Architectures.
"""

# Scaled T4 Champion Architecture (CBT-18.95M)
try:
    from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig
    from champion_model.scaled_t4.tokenizer import ScaledTokenizer
except ImportError:
    pass

# Legacy 100k Architectures (optional)
try:
    from champion_model.math_model.model import MathGPT, GPTConfig
except ImportError:
    pass

try:
    from champion_model.tool_model.model import ToolGPT, ToolGPTConfig
    from champion_model.tool_model.tokenizer import ToolTokenizer
except ImportError:
    pass

__all__ = [
    "ScaledGPT",
    "ScaledGPTConfig",
    "ScaledTokenizer",
    "MathGPT",
    "GPTConfig",
    "ToolGPT",
    "ToolGPTConfig",
    "ToolTokenizer",
]
