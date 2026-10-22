"""
Gated Linear Recurrent Unit (GLRU) for algorithmic sequence modeling.
Implements a hardware-efficient, real-valued diagonal recurrence with 1D depthwise
causal convolution, compatible with Dual NVIDIA T4 (Turing sm_75, FP16 AMP).
Zero external Triton/CUDA C++ build dependencies.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_associative_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes linear recurrence h_t = a_t * h_{t-1} + b_t in parallel along dim=1.
    Uses Hillis-Steele prefix scan in O(log2 T) parallel steps.
    Numerically stabilized by executing scan accumulation in float32.

    Args:
        a: Recurrence gate tensor of shape [B, T, D], values in (0, 1).
        b: Input injection tensor of shape [B, T, D].
    Returns:
        h: Accumulated state tensor of shape [B, T, D].
    """
    orig_dtype = a.dtype
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)

    T = a_f.size(1)
    if T <= 1:
        return b_f.to(orig_dtype)

    num_steps = math.ceil(math.log2(T))

    for step in range(num_steps):
        offset = 1 << step
        if offset >= T:
            break

        # Associative binary composition: (a_r * a_l, a_r * b_l + b_r)
        a_left = a_f[:, :-offset]
        b_left = b_f[:, :-offset]
        a_right = a_f[:, offset:]
        b_right = b_f[:, offset:]

        a_combined = a_right * a_left
        b_combined = a_right * b_left + b_right

        a_f = torch.cat([a_f[:, :offset], a_combined], dim=1)
        b_f = torch.cat([b_f[:, :offset], b_combined], dim=1)

    return b_f.to(orig_dtype)


class GatedLinearRecurrentUnit(nn.Module):
    """
    Real-Gated Linear Recurrent Unit (Hawk/Griffin inspired).
    Features 1D Depthwise Causal Convolution, real-valued diagonal gating,
    and dual-mode execution (Parallel Scan for training, O(1) step for generation).
    """

    def __init__(self, d_model: int = 512, conv_kernel: int = 4, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.conv_kernel = conv_kernel

        # 3x projection: [x_val, x_gate, x_out]
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # 1D Depthwise Causal Convolution for local n-gram context
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            padding=conv_kernel - 1,
            bias=bias,
        )

        # Learnable base decay parameter Lambda in R^d_model
        # Initialized to ensure a_base = sigmoid(Lambda) in [0.88, 0.99]
        self.lambda_param = nn.Parameter(torch.empty(d_model).uniform_(2.0, 4.5))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward execution supporting both sequence training and step generation.

        Args:
            x: [B, T, D] input tensor
            state: Optional [B, D] previous hidden state (used during autoregressive generation)
        Returns:
            y: [B, T, D] output tensor
            next_state: [B, D] updated hidden state
        """
        B, T, D = x.shape

        # Step 1: In-projection
        proj = self.in_proj(x)
        x_val, x_gate, x_out = proj.chunk(3, dim=-1)

        # Step 2: Local 1D Depthwise Causal Convolution
        # Transpose to [B, D, T] for Conv1d
        x_val_conv = self.conv1d(x_val.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_val_act = F.silu(x_val_conv)

        # Step 3: Recurrence and Input Gating
        a_base = torch.sigmoid(self.lambda_param).view(1, 1, D)
        r_t = torch.sigmoid(x_gate)
        a_t = a_base * r_t  # strictly bounded in (0, 1)

        # Input injection modulated by (1 - a_t) for convex combination
        b_t = (1.0 - a_t) * x_val_act

        # Step 4: Recurrent State Update
        if T > 1:
            # Parallel training mode: Associative Prefix Scan
            if self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    parallel_associative_scan, a_t, b_t, use_reentrant=False
                )
            else:
                h = parallel_associative_scan(a_t, b_t)
            if state is not None:
                # Add influence of carry-over state from prior context
                h = h + (torch.cumprod(a_t.to(torch.float32), dim=1) * state.unsqueeze(1).to(torch.float32)).to(x.dtype)
            next_state = h[:, -1, :]
        else:
            # Autoregressive generation mode: O(1) step update
            prev_h = torch.zeros(B, D, device=x.device, dtype=x.dtype) if state is None else state
            h = a_t[:, 0, :] * prev_h + b_t[:, 0, :]
            next_state = h
            h = h.unsqueeze(1)

        # Step 5: Multiplicative Gating and Out-projection
        y = self.out_proj(h * F.silu(x_out))
        return y, next_state
