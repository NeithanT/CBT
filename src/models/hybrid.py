"""
Hybrid SSM-Transformer Architecture for Algorithmic Length Generalization.
Interleaves Gated Linear Recurrent Units (GLRU) with Causal Self-Attention (RoPE + SDPA).
Strictly under the 20M non-embedding parameter ceiling (~18.89M parameters).
Turing sm_75 FP16 AMP compliant with zero external Triton/CUDA dependencies.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import GPTConfig
from src.models.base import BaseReasoningModel, RMSNorm, apply_rope, precompute_rope_freqs
from src.models.glru import GatedLinearRecurrentUnit


@dataclass
class HybridConfig(GPTConfig):
    """Configuration hyperparameters for HybridMathGPT."""
    vocab_size: int = 32
    block_size: int = 512
    n_layer: int = 6
    n_embd: int = 512
    n_head: int = 8
    d_ffn: int = 2048
    dropout: float = 0.05
    bias: bool = False
    rope_theta: float = 10000.0
    pattern: str = "interleaved"  # "interleaved" (GLRU/Attn), "pure_ssm", "pure_attn"
    model_type: str = "hybrid"

    def __post_init__(self):
        self.pos_emb_type = "rope"
        self.use_pos_emb = False


class HybridCausalAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE position embeddings and SDPA.
    Turing sm_75 FP16 compliant with Cutlass kernel activation.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        max_seq = max(config.block_size * 2, 1024)
        cos, sin = precompute_rope_freqs(
            self.head_dim, max_seq_len=max_seq, theta_base=config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, save_attention: bool = False) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if T > self.rope_cos.size(0):
            cos, sin = precompute_rope_freqs(
                self.head_dim, max_seq_len=T * 2, theta_base=self.rope_theta
            )
            self.rope_cos = cos.to(x.device)
            self.rope_sin = sin.to(x.device)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if save_attention:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            self.last_attn_weights = att.detach().cpu()
            att = self.attn_dropout(att)
            y = att @ v
        else:
            dropout_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class HybridMLP(nn.Module):
    """Feed-forward network with 4x expansion and GELU activation."""

    def __init__(self, config: HybridConfig):
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


class HybridBlock(nn.Module):
    """
    Modular block supporting either GLRU temporal recurrence or SDPA attention.
    """

    def __init__(self, config: HybridConfig, mixer_type: str = "glru"):
        super().__init__()
        self.mixer_type = mixer_type
        self.ln_1 = RMSNorm(config.n_embd)

        if mixer_type == "glru":
            self.mixer = GatedLinearRecurrentUnit(
                d_model=config.n_embd, conv_kernel=4, bias=config.bias
            )
        else:
            self.mixer = HybridCausalAttention(config)

        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = HybridMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        save_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mixer_type == "glru":
            mixer_out, next_state = self.mixer(self.ln_1(x), state=state)
        else:
            mixer_out = self.mixer(self.ln_1(x), save_attention=save_attention)
            next_state = None

        x = x + mixer_out
        x = x + self.mlp(self.ln_2(x))
        return x, next_state


class HybridMathGPT(BaseReasoningModel):
    """
    Hybrid SSM-Transformer for Algorithmic Length Generalization.
    Interleaves Gated Linear Recurrent Units (GLRU) with SDPA Attention blocks.
    Strictly constrained below 20M non-embedding parameters.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        # Determine mixer type for each layer
        blocks = []
        for i in range(config.n_layer):
            if config.pattern == "pure_ssm":
                m_type = "glru"
            elif config.pattern == "pure_attn":
                m_type = "attention"
            else:  # interleaved: even=GLRU, odd=Attention
                m_type = "glru" if (i % 2 == 0) else "attention"
            blocks.append(HybridBlock(config, mixer_type=m_type))

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(blocks),
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
        """Retrieves saved attention maps from attention blocks."""
        maps = []
        for block in self.transformer.h:
            if block.mixer_type == "attention" and block.mixer.last_attn_weights is not None:
                maps.append(block.mixer.last_attn_weights)
        return maps

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        states: Optional[List[Optional[torch.Tensor]]] = None,
        save_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        x = self.transformer.drop(self.transformer.wte(idx))

        next_states: List[Optional[torch.Tensor]] = []
        for i, block in enumerate(self.transformer.h):
            prev_s = states[i] if states is not None else None
            x, next_s = block(x, state=prev_s, save_attention=save_attention)
            next_states.append(next_s)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss
