"""
ScaledGPT Architecture for Dual NVIDIA T4 (~18.95M Parameters).
"""

from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig
from champion_model.scaled_t4.tokenizer import ScaledTokenizer
from champion_model.scaled_t4.dataset import generate_joint_dataset, encode_joint_batch

__all__ = [
    "ScaledGPT",
    "ScaledGPTConfig",
    "ScaledTokenizer",
    "generate_joint_dataset",
    "encode_joint_batch",
]
