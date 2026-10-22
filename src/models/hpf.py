"""
Self-Synchronizing Harmonic Positional Field (HPF) Module.
Generates autonomous, length-invariant right-to-left decimal coordinates
by phase-locking an internal bank of neural oscillators to structural delimiters.
Hardware compliant with Dual NVIDIA T4 (Turing sm_75, FP16 AMP).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_harmonic_rope(
    x: torch.Tensor,
    cos_field: torch.Tensor,
    sin_field: torch.Tensor,
) -> torch.Tensor:
    """
    Applies Harmonic Rotary Positional Embedding to query or key tensor.
    Args:
        x: (B, H, T, D_head)
        cos_field: (B, 1, T, D_head // 2)
        sin_field: (B, 1, T, D_head // 2)
    Returns:
        Rotated tensor of shape (B, H, T, D_head)
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rot_even = x_even * cos_field - x_odd * sin_field
    rot_odd = x_even * sin_field + x_odd * cos_field

    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class HarmonicOscillatorBank(nn.Module):
    """
    Bank of trainable neural oscillators synchronizing to structural delimiters.
    Computes autonomous, length-invariant right-to-left decimal coordinates.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        head_dim: int = 64,
        base_freq: float = 1.0,
    ):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_oscillators = head_dim // 2

        # Delimiter detection gate: learns delimiter tokens (+, =, <THINK>, \n)
        self.gate_proj = nn.Linear(embed_dim, 1, bias=True)

        # Trainable log-frequencies initialized geometrically across base-10 scales
        init_freqs = base_freq * (
            10.0
            ** (
                -torch.arange(0, self.num_oscillators, dtype=torch.float32)
                / max(1, self.num_oscillators - 1)
            )
        )
        self.log_omega = nn.Parameter(torch.log(torch.clamp(init_freqs, min=1e-5)))

        # Trainable initial phase offsets
        self.theta_0 = nn.Parameter(torch.zeros(self.num_oscillators, dtype=torch.float32))

        # Additive projection back to embed_dim
        self.proj_out = nn.Linear(head_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        is_causal_generation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing harmonic rotary coordinates and additive field.
        Args:
            x: Hidden representations (B, T, embed_dim).
            is_causal_generation: Whether decoding autoregressively step-by-step.
        Returns:
            cos_field: (B, 1, T, num_oscillators) in x.dtype
            sin_field: (B, 1, T, num_oscillators) in x.dtype
            hpf_emb:   (B, T, embed_dim) in x.dtype
        """
        B, T, D = x.size()
        device = x.device

        # Compute autonomous delimiter probability gate in FP32 for numerical stability
        gate_logits = self.gate_proj(x.float()).squeeze(-1)
        gate = torch.sigmoid(gate_logits)
        mask_hard = (gate > 0.5).float()
        mask_st = gate + (mask_hard - gate).detach()

        # Vectorized segment distance calculation in O(T) parallel time
        seq_idx = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, T)

        if not is_causal_generation:
            # Anticipatory right-to-left decimal coordinates (distance to NEXT delimiter)
            masked_idx = mask_st * seq_idx + (1.0 - mask_st) * (-1.0)
            flipped_masked = torch.flip(masked_idx, dims=[1])
            flipped_cummax = torch.cummax(flipped_masked, dim=1).values
            next_delim_idx = torch.flip(flipped_cummax, dims=[1])

            # For tokens with no following delimiter, default to end of sequence
            next_delim_idx = torch.where(next_delim_idx < 0, float(T), next_delim_idx)
            delta = torch.clamp(next_delim_idx - seq_idx, min=0.0)
        else:
            # Forward segment coordinates (distance from LAST delimiter)
            masked_idx = mask_st * seq_idx + (1.0 - mask_st) * (-1.0)
            last_delim_idx = torch.cummax(masked_idx, dim=1).values
            last_delim_idx = torch.where(last_delim_idx < 0, 0.0, last_delim_idx)
            delta = torch.clamp(seq_idx - last_delim_idx, min=0.0)

        # Compute phase accumulation in FP32 with 2*pi wrapping to preserve FP16 accuracy
        omega = torch.exp(self.log_omega).view(1, 1, self.num_oscillators)
        theta = self.theta_0.view(1, 1, self.num_oscillators)
        delta_expanded = delta.unsqueeze(-1)

        raw_phase = delta_expanded * omega + theta
        wrapped_phase = torch.remainder(raw_phase, 2.0 * math.pi)

        cos_field = torch.cos(wrapped_phase)
        sin_field = torch.sin(wrapped_phase)

        # Construct full additive harmonic representation: [cos; sin] -> (B, T, head_dim)
        harmonic_repr = torch.cat([cos_field, sin_field], dim=-1)
        hpf_emb = self.proj_out(harmonic_repr.to(dtype=x.dtype)) + 0.01 * gate.unsqueeze(-1).to(dtype=x.dtype)

        # Format for multi-head attention broadcasting: (B, 1, T, K)
        cos_field = cos_field.unsqueeze(1).to(dtype=x.dtype)
        sin_field = sin_field.unsqueeze(1).to(dtype=x.dtype)

        return cos_field, sin_field, hpf_emb


class HPFCausalAttention(nn.Module):
    """
    Causal Multi-Head Attention integrating Harmonic Rotary Embeddings (H-RoPE).
    Fully compliant with Turing sm_75 FP16 AMP and PyTorch SDPA Cutlass kernels.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_head: int = 8,
        dropout: float = 0.05,
        bias: bool = False,
    ):
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.dropout = dropout

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        cos_field: torch.Tensor,
        sin_field: torch.Tensor,
        save_attention: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply Harmonic Rotary Positional Embedding (H-RoPE)
        q = apply_harmonic_rope(q, cos_field, sin_field)
        k = apply_harmonic_rope(k, cos_field, sin_field)

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
