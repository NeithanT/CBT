"""
Package src.models: Architectures for algorithmic length generalization.
"""

from src.models.base import BaseReasoningModel, RMSNorm, apply_rope, precompute_rope_freqs
from src.models.baseline import MathGPT, GPTConfig, Block, MLP, CausalSelfAttention
from src.models.factory import create_model
from src.models.looped import (
    LoopedGPTConfig,
    LoopedCausalAttention,
    LoopedMLP,
    CoreRecurrentBlock,
    LoopedMathGPT,
)
from src.models.hierarchical import (
    HierarchicalGPTConfig,
    HierarchicalCausalAttention,
    HierarchicalMLP,
    FiLMModulation,
    FastReasoningBlock,
    SlowPlanningBlock,
    HierarchicalReasoningGPT,
)
from src.models.glru import GatedLinearRecurrentUnit, parallel_associative_scan
from src.models.hybrid import HybridConfig, HybridMathGPT
from src.models.alsm import ALSMConfig, AbacusLoopedMathGPT
from src.models.hpf import HarmonicOscillatorBank, HPFCausalAttention
from src.models.attractor_alu import BistableAttractorRegister, NeuralALUCell
from src.models.hnrm import HNRMConfig, HarmonicNRM
from src.models.micromath import MicroMathConfig, MicroMath2M
from src.models.hrm_rope import HRMRoPEConfig, HRMRoPE
from src.models.hrm_text import HRMTextConfig, HRMTextModel

__all__ = [
    "BaseReasoningModel",
    "RMSNorm",
    "apply_rope",
    "precompute_rope_freqs",
    "MathGPT",
    "GPTConfig",
    "Block",
    "MLP",
    "CausalSelfAttention",
    "create_model",
    "LoopedGPTConfig",
    "LoopedCausalAttention",
    "LoopedMLP",
    "CoreRecurrentBlock",
    "LoopedMathGPT",
    "HierarchicalGPTConfig",
    "HierarchicalCausalAttention",
    "HierarchicalMLP",
    "FiLMModulation",
    "FastReasoningBlock",
    "SlowPlanningBlock",
    "HierarchicalReasoningGPT",
    "GatedLinearRecurrentUnit",
    "parallel_associative_scan",
    "HybridConfig",
    "HybridMathGPT",
    "ALSMConfig",
    "AbacusLoopedMathGPT",
    "HarmonicOscillatorBank",
    "HPFCausalAttention",
    "BistableAttractorRegister",
    "NeuralALUCell",
    "HNRMConfig",
    "HarmonicNRM",
    "MicroMathConfig",
    "MicroMath2M",
    "HRMRoPEConfig",
    "HRMRoPE",
    "HRMTextConfig",
    "HRMTextModel",
]

