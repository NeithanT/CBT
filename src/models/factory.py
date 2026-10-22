"""
Model factory for instantiating algorithmic reasoning architectures across Phase 1 and Phase 2.
"""

from typing import Union, Any
import torch.nn as nn

from src.models.baseline import MathGPT, GPTConfig


def create_model(config: Any) -> nn.Module:
    """
    Instantiates an algorithmic reasoning model based on the configuration object.
    Supports MathGPT (standard), LoopedMathGPT (looped), and HierarchicalReasoningGPT (hierarchical).
    """
    model_type = getattr(config, "model_type", "standard")
    config_cls_name = type(config).__name__

    if model_type == "looped" or config_cls_name == "LoopedGPTConfig":
        from src.models.looped import LoopedMathGPT
        return LoopedMathGPT(config)
    elif model_type == "hierarchical" or config_cls_name == "HierarchicalGPTConfig":
        from src.models.hierarchical import HierarchicalReasoningGPT
        return HierarchicalReasoningGPT(config)
    elif model_type == "hybrid" or config_cls_name == "HybridConfig":
        from src.models.hybrid import HybridMathGPT
        return HybridMathGPT(config)
    elif model_type in ["alsm", "abacus_looped"] or config_cls_name == "ALSMConfig":
        from src.models.alsm import AbacusLoopedMathGPT
        return AbacusLoopedMathGPT(config)
    elif model_type == "hnrm" or config_cls_name == "HNRMConfig":
        from src.models.hnrm import HarmonicNRM
        return HarmonicNRM(config)
    elif model_type in ["micromath", "micromath2m"] or config_cls_name == "MicroMathConfig":
        from src.models.micromath import MicroMath2M
        return MicroMath2M(config)
    elif model_type == "hrm_rope" or config_cls_name == "HRMRoPEConfig":
        from src.models.hrm_rope import HRMRoPE
        return HRMRoPE(config)
    elif model_type in ["hrm_text", "hrmtext"] or config_cls_name == "HRMTextConfig":
        from src.models.hrm_text import HRMTextModel
        return HRMTextModel(config)
    else:
        return MathGPT(config)

