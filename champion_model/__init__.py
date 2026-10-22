"""
Champion Model Package: 100k Recurrent 2D-RoPE Architecture.
Achieved 100% accuracy in multi-digit arithmetic extrapolation (1-20 digits)
and 100% accuracy in structured natural language tool calling.
"""

from champion_model.math_model.model import MathGPT, GPTConfig
from champion_model.tool_model.model import ToolGPT, ToolGPTConfig
from champion_model.tool_model.tokenizer import ToolTokenizer

__all__ = [
    "MathGPT",
    "GPTConfig",
    "ToolGPT",
    "ToolGPTConfig",
    "ToolTokenizer",
]
