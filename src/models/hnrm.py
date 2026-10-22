"""
Harmonic Neural Register Machine (H-NRM) Architecture.
Combines Self-Synchronizing Harmonic Positional Fields (HPF) with
Differentiable Neural ALU and Bistable Attractor Dynamics (NRM-BA).
Provides autonomous coordinate invariance and noise-immune carry state propagation.
Strictly under the 20M non-embedding parameter ceiling (~18.93M parameters).
Turing sm_75 FP16 AMP compliant.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import GPTConfig
from src.models.base import BaseReasoningModel, RMSNorm
from src.models.hpf import HarmonicOscillatorBank, HPFCausalAttention
from src.models.attractor_alu import NeuralALUCell


@dataclass
class HNRMConfig(GPTConfig):
    """Configuration hyperparameters for HarmonicNRM."""
    vocab_size: int = 32
    block_size: int = 512
    n_layer: int = 6
    n_embd: int = 512
    n_head: int = 8
    d_ffn: int = 2048
    dropout: float = 0.05
    bias: bool = False
    base_freq: float = 1.0
    model_type: str = "hnrm"

    def __post_init__(self):
        self.pos_emb_type = "hnrm"
        self.use_pos_emb = False


class HNRMMLP(nn.Module):
    """Position-wise feed-forward network with 4x expansion and GELU activation."""

    def __init__(self, config: HNRMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.d_ffn, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ffn, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class HNRMBlock(nn.Module):
    """
    Dual-Manifold H-NRM Transformer Block.
    Integrates Harmonic Causal Attention, Differentiable Neural ALU with
    bistable attractor dynamics, and dense feed-forward MLP.
    """

    def __init__(self, config: HNRMConfig):
        super().__init__()
        self.config = config
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = HPFCausalAttention(
            embed_dim=config.n_embd,
            n_head=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.ln_alu = RMSNorm(config.n_embd)
        self.alu = NeuralALUCell(d_model=config.n_embd)

        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = HNRMMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_field: torch.Tensor,
        sin_field: torch.Tensor,
        c_prev: Optional[torch.Tensor] = None,
        save_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Harmonic Self-Attention
        attn_out = self.attn(
            self.ln_1(x), cos_field=cos_field, sin_field=sin_field, save_attention=save_attention
        )
        x = x + attn_out

        # Step 2: Neural ALU with Bistable Attractor Dynamics
        alu_delta, c_last = self.alu(self.ln_alu(x), c_prev=c_prev)
        x = x + alu_delta

        # Step 3: Feed-Forward MLP
        x = x + self.mlp(self.ln_2(x))
        return x, c_last


class HarmonicNRM(BaseReasoningModel):
    """
    Harmonic Neural Register Machine (H-NRM).
    An entirely novel architecture combining:
    1. Self-Synchronizing Harmonic Positional Field (HPF) for autonomous coordinate invariance.
    2. Differentiable Neural ALU with Bistable Attractor Dynamics (NRM-BA) for noise-immune carry tracking.
    3. Causal Multi-Head Attention with SDPA Cutlass kernels.
    Strictly under 20M non-embedding parameters.
    """

    def __init__(self, config: HNRMConfig):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                hpf_bank=HarmonicOscillatorBank(
                    embed_dim=config.n_embd, head_dim=head_dim, base_freq=config.base_freq
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([HNRMBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding and LM head share parameters
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def get_attention_maps(self) -> List[torch.Tensor]:
        return [
            block.attn.last_attn_weights
            for block in self.transformer.h
            if block.attn.last_attn_weights is not None
        ]

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        save_attention: bool = False,
        is_causal_generation: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()

        # Step 1: Token embedding
        tok_emb = self.transformer.wte(idx)

        # Step 2: Compute Autonomous Harmonic Positional Field (HPF)
        cos_field, sin_field, hpf_emb = self.transformer.hpf_bank(
            tok_emb, is_causal_generation=is_causal_generation
        )

        # Fuse token embedding with continuous additive harmonic field
        x = self.transformer.drop(tok_emb + hpf_emb)

        # Step 3: Forward through Dual-Manifold Transformer Blocks
        c_state: Optional[torch.Tensor] = None
        for block in self.transformer.h:
            x, c_state = block(
                x,
                cos_field=cos_field,
                sin_field=sin_field,
                c_prev=c_state,
                save_attention=save_attention,
            )

        # Step 4: Final norm & logits
        normed_x = self.transformer.ln_f(x)
        logits = self.lm_head(normed_x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss
