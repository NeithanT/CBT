"""
Abacus-Anchored Looped State Machine (ALSM-GPT) Architecture.
Combines recurrent depth with Abacus Positional Anchoring and Gated Input Injection.
Provides invariant column-level key-query alignment and unbounded sequential reasoning.
Strictly under the 20M non-embedding parameter ceiling (~12.68M parameters).
Turing sm_75 FP16 AMP compliant.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import GPTConfig
from src.models.base import BaseReasoningModel, RMSNorm, apply_rope, precompute_rope_freqs


@dataclass
class ALSMConfig(GPTConfig):
    """Configuration hyperparameters for AbacusLoopedMathGPT (ALSM)."""
    vocab_size: int = 32
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 8
    n_core_layers: int = 4
    train_loops: int = 6
    eval_loops: int = 12
    max_places: int = 128
    k_rand_shift: int = 32
    dropout: float = 0.05
    bias: bool = False
    rope_theta: float = 10000.0
    deep_supervision: bool = True
    model_type: str = "alsm"

    def __post_init__(self):
        self.pos_emb_type = "rope"
        self.use_pos_emb = False
        self.n_layer = self.n_core_layers


class ALSMCausalAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE position embeddings and SDPA.
    Turing sm_75 FP16 compliant with Cutlass kernel activation.
    """

    def __init__(self, config: ALSMConfig):
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


class ALSMMLP(nn.Module):
    """Feed-forward network with 4x expansion and GELU activation."""

    def __init__(self, config: ALSMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class ALSMBlock(nn.Module):
    """Core recurrent block with RMSNorm, attention, and MLP."""

    def __init__(self, config: ALSMConfig):
        super().__init__()
        self.config = config
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = ALSMCausalAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = ALSMMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        scale_factor: float,
        save_attention: bool = False,
    ) -> torch.Tensor:
        x = x + scale_factor * self.attn(self.ln_1(x), save_attention=save_attention)
        x = x + scale_factor * self.mlp(self.ln_2(x))
        return x


class AbacusLoopedMathGPT(BaseReasoningModel):
    """
    Abacus-Anchored Looped State Machine GPT (ALSM-GPT).
    Features:
    1. Abacus Positional Embeddings aligning digit place values relative to LSD.
    2. Offset Randomization during training for relative-place invariance.
    3. Gated Input Injection Highway preventing representation decay.
    4. Weight-tied recurrent depth with attractor scaling (1/sqrt(K)).
    5. Linear-ramp deep supervision across recurrent loops.
    """

    def __init__(self, config: ALSMConfig):
        super().__init__()
        self.config = config

        total_places = config.max_places + config.k_rand_shift + 1
        self.wpe_abacus = nn.Embedding(total_places, config.n_embd)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                injection_norm=RMSNorm(config.n_embd),
                h=nn.ModuleList([ALSMBlock(config) for _ in range(config.n_core_layers)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-loop input injection gate parameters (up to 32 loops max)
        self.injection_gates = nn.Parameter(torch.full((32, config.n_embd), 0.1))

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
        place_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        num_loops: Optional[int] = None,
        save_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        device = idx.device

        if place_ids is None:
            place_ids = torch.zeros((B, T), dtype=torch.long, device=device)

        # Offset jitter during training for translation-invariant place representations
        if self.training and self.config.k_rand_shift > 0:
            shifts = torch.randint(0, self.config.k_rand_shift + 1, (B, 1), device=device)
            shift_mask = (place_ids > 0).long()
            place_ids_effective = place_ids + shifts * shift_mask
        else:
            place_ids_effective = place_ids

        # Clamp to valid vocabulary of abacus embedding table
        place_ids_effective = torch.clamp(
            place_ids_effective, 0, self.wpe_abacus.num_embeddings - 1
        )

        tok_emb = self.transformer.wte(idx)
        plc_emb = self.wpe_abacus(place_ids_effective)
        e_in = self.transformer.drop(tok_emb + plc_emb)

        if num_loops is None:
            num_loops = self.config.train_loops if self.training else self.config.eval_loops

        scale_factor = 1.0 / math.sqrt(max(1, num_loops))
        use_deep_supervision = (
            self.training and self.config.deep_supervision and targets is not None
        )

        intermediate_losses: List[torch.Tensor] = []
        last_logits: Optional[torch.Tensor] = None
        x = e_in

        normed_input = self.transformer.injection_norm(e_in)

        for loop_idx in range(num_loops):
            if loop_idx > 0:
                gate_idx = min(loop_idx, self.injection_gates.size(0) - 1)
                gate = self.injection_gates[gate_idx].unsqueeze(0).unsqueeze(0)
                x = x + gate * normed_input

            for block in self.transformer.h:
                if self.training and torch.is_grad_enabled():
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, scale_factor, save_attention, use_reentrant=False
                    )
                else:
                    x = block(x, scale_factor=scale_factor, save_attention=save_attention)

            if use_deep_supervision:
                normed_x = self.transformer.ln_f(x)
                loop_logits = self.lm_head(normed_x)
                loop_loss = F.cross_entropy(
                    loop_logits.view(-1, loop_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
                intermediate_losses.append(loop_loss)
                if loop_idx == num_loops - 1:
                    last_logits = loop_logits

        if not use_deep_supervision:
            normed_x = self.transformer.ln_f(x)
            last_logits = self.lm_head(normed_x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    last_logits.view(-1, last_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
        else:
            total_weight = sum(range(1, num_loops + 1))
            weights = [(k + 1) / total_weight for k in range(num_loops)]
            loss = sum(w * l for w, l in zip(weights, intermediate_losses))

        return last_logits, loss
