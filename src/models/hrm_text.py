"""
HRM-Text: Compressed Adaptation of Sapient Inc's Hierarchical Reasoning Model.
Paper: "HRM-Text: Efficient Pretraining Beyond Scaling" (arXiv:2605.20613).

Features:
1. Capacity: Strictly < 2.0M parameters (~1.44M non-embedding parameters).
2. Dual Hierarchical Recurrence:
   - High-level module (H_level): Strategic macro transformer block.
   - Low-level module (L_level): Operational micro transformer block.
   - Additive input injection: z_L = L(z_L + z_H), z_H = H(z_H + z_L).
3. Gate-Augmented Multi-Head Attention:
   - Joint projection of [Gate, Query, Key, Value].
   - Gating activation: output = W_O(sigmoid(Gate) * Attention(Q, K, V)).
4. SwiGLU Feed-Forward Network:
   - Gate-up projection with SiLU gating: W_down(SiLU(Gate) * Up).
5. Turing sm_75 FP16 AMP compliant via PyTorch Cutlass SDPA.
6. Weight tying between token embedding and LM head.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseReasoningModel, RMSNorm, precompute_rope_freqs, apply_rope


@dataclass
class HRMTextConfig:
    """Hyperparameters for compressed HRM-Text architecture."""
    vocab_size: int = 32
    block_size: int = 512
    hidden_size: int = 256
    num_heads: int = 4
    expansion: float = 3.0
    h_cycles: int = 2
    l_cycles: int = 3
    n_layers: int = 2          # Total layers; half_layers=True splits 1 to H and 1 to L
    half_layers: bool = True
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    rope_mode: str = "1d"      # "1d" (official HRM-Text) or "2d" (decoupled place-step)
    model_type: str = "hrm_text"

    @property
    def n_embd(self) -> int:
        return self.hidden_size

    @property
    def n_head(self) -> int:
        return self.num_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        raw = round(self.expansion * self.hidden_size * 2 / 3)
        # Find multiple of 256 matching official HRM-Text
        return (- (raw // -256)) * 256


class GateAttention(nn.Module):
    """
    Gate-Augmented Multi-Head Attention from official HRM-Text.
    Projects hidden state into [Gate, Query, Key, Value] and applies
    output = W_O(sigmoid(Gate) * SDPA(Q, K, V)).
    """

    def __init__(self, config: HRMTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        # gqkv projection: 2 * num_heads (gate + query) + 2 * num_heads (key + value)
        # Total output channels: 4 * hidden_size
        total_out = 4 * self.hidden_size
        self.gqkv_proj = nn.Linear(self.hidden_size, total_out, bias=config.bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        save_attention: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.size()
        gqkv = self.gqkv_proj(x)  # [B, T, 4 * C]

        # Reshape to [B, T, 4 * num_heads, head_dim]
        gqkv = gqkv.view(B, T, 4 * self.num_heads, self.head_dim)
        gate, query, key, value = gqkv.chunk(4, dim=-2)

        # Transpose to [B, num_heads, T, head_dim]
        gate = gate.transpose(1, 2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply RoPE
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)

        if save_attention:
            att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            att_weights = F.softmax(att, dim=-1)
            self.last_attn_weights = att_weights.detach()
            attn_out = att_weights @ value
        else:
            self.last_attn_weights = None
            attn_out = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )

        # Gate-augmented activation: sigmoid(gate) * attn_out
        gated_out = torch.sigmoid(gate) * attn_out

        # Merge heads back to [B, T, hidden_size]
        gated_out = gated_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(gated_out))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network from official HRM-Text."""

    def __init__(self, config: HRMTextConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=config.bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.dropout(self.down_proj(F.silu(gate) * up))


class HRMTransformerBlock(nn.Module):
    """Single Pre-Norm Transformer block with GateAttention and SwiGLU."""

    def __init__(self, config: HRMTextConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GateAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        save_attention: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, save_attention=save_attention)
        x = x + self.mlp(self.norm2(x))
        return x


class HRMTextRecurrentBlock(nn.Module):
    """
    Recurrent block wrapping Transformer sub-network with additive input injection.
    From official HRM-Text: forward(hidden_states, input_injection) -> core(hidden_states + input_injection).
    """

    def __init__(self, config: HRMTextConfig, n_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([HRMTransformerBlock(config) for _ in range(n_layers)])
        self.norm_f = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        save_attention: bool = False,
    ) -> torch.Tensor:
        # Additive input injection from counterpart hierarchical level
        x = hidden_states + input_injection
        for layer in self.layers:
            x = layer(x, cos, sin, save_attention=save_attention)
        return self.norm_f(x)


class HRMTextModel(BaseReasoningModel):
    """
    Compressed HRM-Text Foundation Architecture.
    Faithful adaptation of sapientinc/HRM-Text under < 2.0M parameters.
    """

    def __init__(self, config: HRMTextConfig):
        super().__init__()
        self.config = config

        layers_per_level = (config.n_layers // 2) if config.half_layers else config.n_layers
        assert layers_per_level >= 1, "Must have at least 1 layer per hierarchical level"

        # RoPE frequency buffers
        cos, sin = precompute_rope_freqs(
            config.head_dim, max_seq_len=config.block_size * 2, theta_base=config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

        # Hierarchical levels
        self.H_level = HRMTextRecurrentBlock(config, n_layers=layers_per_level)
        self.L_level = HRMTextRecurrentBlock(config, n_layers=layers_per_level)

        # Initial learnable/static operational state z_L
        self.zL_init = nn.Parameter(torch.zeros(config.hidden_size))
        torch.nn.init.normal_(self.zL_init, mean=0.0, std=0.02)

        # Final norm and output head
        self.ln_f = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying: share parameters between embedding and output projection
        self.wte.weight = self.lm_head.weight

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
            n_params -= self.wte.weight.numel()
        return n_params

    def get_attention_maps(self) -> List[torch.Tensor]:
        weights = []
        for block in list(self.H_level.layers) + list(self.L_level.layers):
            if block.attn.last_attn_weights is not None:
                weights.append(block.attn.last_attn_weights)
        return weights

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        save_attention: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        device = idx.device

        # Support keyword passing or positional permutations
        if "targets" in kwargs and targets is None:
            targets = kwargs.pop("targets")

        # Dynamically extend RoPE table if T exceeds precomputed block size
        if T > self.rope_cos.size(0):
            cos, sin = precompute_rope_freqs(
                self.config.head_dim, max_seq_len=T * 2, theta_base=self.config.rope_theta
            )
            self.rope_cos = cos.to(device)
            self.rope_sin = sin.to(device)

        # Token embedding
        x = self.drop(self.wte(idx))

        # Initial hierarchical states
        z_H = x
        z_L = self.zL_init.view(1, 1, -1).expand(B, T, -1)

        # Official HRM-Text hierarchical unrolling loop:
        # H_cycles outer loops, L_cycles inner loops
        for _ in range(self.config.h_cycles):
            for _ in range(self.config.l_cycles):
                z_L = self.L_level(
                    z_L, z_H, self.rope_cos, self.rope_sin, save_attention=save_attention
                )
            z_H = self.H_level(
                z_H, z_L, self.rope_cos, self.rope_sin, save_attention=save_attention
            )

        z_out = self.ln_f(z_H)
        logits = self.lm_head(z_out)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive generation loop for HRM-Text."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx, targets=None, **kwargs)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            next_token = idx_next.item()
            if eos_id is not None and next_token == eos_id:
                break
            if stop_tokens is not None and next_token in stop_tokens:
                break

        return idx
