"""
Hierarchical Reasoning Model (HRM) with Dual-Loop Mechanism.
Implements multi-timescale latent reasoning separating high-level strategic planning
from low-level algorithmic computation, compliant with Turing sm_75 FP16 kernels.
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
class HierarchicalGPTConfig(GPTConfig):
    """Hyperparameters for HierarchicalReasoningGPT."""
    vocab_size: int = 32
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 8
    n_fast_steps: int = 4  # inner T
    n_slow_cycles: int = 3  # outer K
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    halt_epsilon: float = 1e-3
    ponder_penalty: float = 0.01
    one_step_gradient: bool = False
    model_type: str = "hierarchical"

    def __post_init__(self):
        self.pos_emb_type = "rope"
        self.use_pos_emb = False
        self.n_layer = self.n_slow_cycles * self.n_fast_steps


class HierarchicalCausalAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE position embeddings and SDPA.
    Turing sm_75 FP16 compliant with Cutlass kernel activation.
    """

    def __init__(self, config: HierarchicalGPTConfig):
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

        # Dynamic RoPE cache extension if sequence length exceeds precomputed buffer
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
        y = self.resid_dropout(self.c_proj(y))
        return y


class HierarchicalMLP(nn.Module):
    """Position-wise feed-forward network with 4x expansion and GELU activation."""

    def __init__(self, config: HierarchicalGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Applies affine modulation: gamma(z_H) * Norm(z_L) + beta(z_H)
    where gamma and beta are projected from strategic state z_H.
    """

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, 2 * dim, bias=bias)

    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor) -> torch.Tensor:
        gamma_raw, beta = self.proj(z_H).chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma_raw)
        return gamma * self.norm(z_L) + beta


class FastReasoningBlock(nn.Module):
    """
    Low-Level (L) module executing fast inner steps.
    Features FiLM conditioning on strategic state z_H, causal SDPA attention
    with Rotary Position Embeddings (RoPE), feed-forward MLP, and dynamic halting logit.
    """

    def __init__(self, config: HierarchicalGPTConfig):
        super().__init__()
        self.config = config
        self.film = FiLMModulation(config.n_embd, bias=config.bias)
        self.attn = HierarchicalCausalAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = HierarchicalMLP(config)
        self.halt_head = nn.Linear(config.n_embd, 1, bias=config.bias)

    def forward(
        self,
        z_L: torch.Tensor,
        z_H: torch.Tensor,
        scale_factor: float = 1.0,
        save_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_L_mod = self.film(z_L, z_H)
        z_L = z_L + scale_factor * self.attn(z_L_mod, save_attention=save_attention)
        z_L = z_L + scale_factor * self.mlp(self.ln_2(z_L))
        halt_logits = self.halt_head(z_L)
        return z_L, halt_logits


class SlowPlanningBlock(nn.Module):
    """
    High-Level (H) module updating strategic state z_H at step boundary T.
    Features gated upward feedback integration: gate = sigmoid(W_g [z_H; candidate]),
    causal SDPA attention with RoPE, and feed-forward MLP.
    """

    def __init__(self, config: HierarchicalGPTConfig):
        super().__init__()
        self.config = config
        self.w_g = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = HierarchicalCausalAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = HierarchicalMLP(config)
        self.last_gate: Optional[torch.Tensor] = None

    def forward(
        self,
        z_H: torch.Tensor,
        candidate: Optional[torch.Tensor] = None,
        scale_factor: float = 1.0,
        save_attention: bool = False,
    ) -> torch.Tensor:
        if candidate is not None:
            gate = torch.sigmoid(self.w_g(torch.cat([z_H, candidate], dim=-1)))
            self.last_gate = gate.detach()
            z_H = gate * z_H + (1.0 - gate) * candidate

        z_H = z_H + scale_factor * self.attn(self.ln_1(z_H), save_attention=save_attention)
        z_H = z_H + scale_factor * self.mlp(self.ln_2(z_H))
        return z_H


class HierarchicalReasoningGPT(BaseReasoningModel):
    """
    Hierarchical Reasoning Model (HRM) with Dual-Loop Mechanism.
    Maintains slow-loop strategic state z_H and fast-loop operational state z_L.
    Features:
    - Outer cycle loop (K slow cycles) with inner loop (T fast steps)
    - FiLM downward modulation and gated upward feedback
    - Dynamic pondering / delta-norm halting support
    - Truncated/1-step gradient support via optional context detachment for earlier cycles if one_step_gradient=True
    - Weight tying: wte.weight = lm_head.weight
    - Strictly < 20M non-embedding parameters
    """

    def __init__(self, config: HierarchicalGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                fast_block=FastReasoningBlock(config),
                slow_block=SlowPlanningBlock(config),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding and LM head share parameters
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # Zero-initialize FiLM projection weight and bias so it begins as identity modulation
        if hasattr(self.fast_block, "film") and hasattr(self.fast_block.film, "proj"):
            nn.init.zeros_(self.fast_block.film.proj.weight)
            if self.fast_block.film.proj.bias is not None:
                nn.init.zeros_(self.fast_block.film.proj.bias)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    @property
    def fast_block(self) -> FastReasoningBlock:
        return self.transformer.fast_block

    @property
    def slow_block(self) -> SlowPlanningBlock:
        return self.transformer.slow_block

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns non-embedding parameter count deducting wte."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def get_attention_maps(self) -> List[torch.Tensor]:
        """Retrieves saved attention weight tensors from fast and slow blocks."""
        maps = []
        if self.fast_block.attn.last_attn_weights is not None:
            maps.append(self.fast_block.attn.last_attn_weights)
        if self.slow_block.attn.last_attn_weights is not None:
            maps.append(self.slow_block.attn.last_attn_weights)
        return maps

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        save_attention: bool = False,
        n_slow_cycles: Optional[int] = None,
        n_fast_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T_seq = idx.size()
        tok_emb = self.transformer.wte(idx)
        z_L = self.transformer.drop(tok_emb)
        z_H = z_L.clone()

        K = n_slow_cycles if n_slow_cycles is not None else self.config.n_slow_cycles
        T = n_fast_steps if n_fast_steps is not None else self.config.n_fast_steps
        scale_factor = 1.0 / math.sqrt(max(1, K * T))

        halt_logits_list: List[torch.Tensor] = []

        for cycle in range(K):
            # Truncated / 1-step gradient support: detach context for earlier cycles
            if self.config.one_step_gradient and self.training and cycle > 0:
                z_H = z_H.detach()
                z_L = z_L.detach()

            # Fast reasoning inner loop (T fast steps)
            for step in range(T):
                z_L_prev = z_L
                # Fast reasoning step with FiLM conditioning on strategic state z_H
                if self.training and torch.is_grad_enabled():
                    z_L, halt_logits = torch.utils.checkpoint.checkpoint(
                        self.fast_block, z_L, z_H, scale_factor, save_attention, use_reentrant=False
                    )
                else:
                    z_L, halt_logits = self.fast_block(
                        z_L, z_H, scale_factor=scale_factor, save_attention=save_attention
                    )
                halt_logits_list.append(halt_logits)

                # Dynamic pondering / delta-norm halting support during evaluation
                if not self.training:
                    delta_norm = torch.norm(z_L - z_L_prev, p=2, dim=-1).mean()
                    halt_prob = torch.sigmoid(halt_logits).mean()
                    if delta_norm < self.config.halt_epsilon or halt_prob > (1.0 - self.config.halt_epsilon):
                        break

            # Slow planning update at step boundary T with gated upward feedback
            if self.training and torch.is_grad_enabled():
                z_H = torch.utils.checkpoint.checkpoint(
                    self.slow_block, z_H, z_L, scale_factor, save_attention, use_reentrant=False
                )
            else:
                z_H = self.slow_block(
                    z_H, candidate=z_L, scale_factor=scale_factor, save_attention=save_attention
                )

        normed_z_L = self.transformer.ln_f(z_L)
        logits = self.lm_head(normed_z_L)

        loss = None
        if targets is not None:
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            if self.config.ponder_penalty > 0.0 and len(halt_logits_list) > 0:
                all_halts = torch.cat(halt_logits_list, dim=-1)
                ponder_cost = torch.sigmoid(all_halts).mean()
                loss = task_loss + self.config.ponder_penalty * ponder_cost
            else:
                loss = task_loss

        return logits, loss
