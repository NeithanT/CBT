"""
HRM-RoPE: Hierarchical Reasoning Model with Decoupled 2D-RoPE.
Features:
1. Strictly < 2M parameters (~1.57M non-embedding, ~1.58M total).
2. Decoupled 2D-RoPE: Macro Place Coordinate (p) + Micro Template Step (tau).
3. Dual-timescale recurrence: FastReasoningBlock (Inner ALU) + SlowPlanningBlock (Outer Controller).
4. FiLM modulation from slow strategic state into fast operational state.
5. In-model dynamic 2D coordinate calculation for seamless generation.
6. Tied embedding and LM-head weights.
Turing sm_75 FP16 AMP compliant with Cutlass SDPA.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for FP16 numerical stability."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class BaseReasoningModel(nn.Module):
    """Abstract base class for algorithmic reasoning models."""

    def get_num_params(self, non_embedding: bool = True) -> int:
        raise NotImplementedError


@dataclass
class HRMRoPEConfig:
    """Hyperparameters for HRM-RoPE."""
    vocab_size: int = 32
    block_size: int = 512
    n_embd: int = 256
    n_head: int = 4
    d_ffn: int = 768
    n_fast_steps: int = 2
    n_slow_cycles: int = 2
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    max_coords: int = 2048
    k_rand_shift: int = 32
    model_type: str = "hrm_rope"


def precompute_2d_rope_freqs(
    head_dim: int, max_coords: int = 2048, theta_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes rotary frequency tables for dual-coordinate 2D-RoPE.
    First half of head_dim encodes macro place rank p, second half encodes micro step tau.
    """
    half_dim = head_dim // 2
    assert half_dim % 2 == 0, f"head_dim // 2 must be even, got {half_dim}"
    dim_indices = torch.arange(0, half_dim, 2, dtype=torch.float32)
    freqs = 1.0 / (theta_base ** (dim_indices / half_dim))
    coords = torch.arange(max_coords, dtype=torch.float32)
    angles = torch.outer(coords, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_2d_rope(
    x: torch.Tensor,
    place_coords: torch.Tensor,
    step_coords: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Applies decoupled 2D-RoPE to query or key tensor of shape [B, n_head, T, head_dim].
    Rotates the first half of channels by place_coords (p) and the second half by step_coords (tau).
    """
    B, H, T, D = x.shape
    half_d = D // 2

    # Clamp coordinates to table bounds
    max_c = cos.size(0) - 1
    p_clamped = torch.clamp(place_coords, 0, max_c)
    s_clamped = torch.clamp(step_coords, 0, max_c)

    cos_p = cos[p_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)
    sin_p = sin[p_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)
    cos_s = cos[s_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)
    sin_s = sin[s_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)

    x_place = x[..., :half_d]
    x_step = x[..., half_d:]

    # Rotate place subspace
    p_even = x_place[..., 0::2]
    p_odd = x_place[..., 1::2]
    rot_p_even = p_even * cos_p - p_odd * sin_p
    rot_p_odd = p_even * sin_p + p_odd * cos_p
    rot_place = torch.empty_like(x_place)
    rot_place[..., 0::2] = rot_p_even
    rot_place[..., 1::2] = rot_p_odd

    # Rotate step subspace
    s_even = x_step[..., 0::2]
    s_odd = x_step[..., 1::2]
    rot_s_even = s_even * cos_s - s_odd * sin_s
    rot_s_odd = s_even * sin_s + s_odd * cos_s
    rot_step = torch.empty_like(x_step)
    rot_step[..., 0::2] = rot_s_even
    rot_step[..., 1::2] = rot_s_odd

    return torch.cat([rot_place, rot_step], dim=-1)


class HRMRoPECausalAttention(nn.Module):
    """Causal multi-head self-attention with 2D-RoPE and SDPA."""

    def __init__(self, config: HRMRoPEConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        place_coords: torch.Tensor,
        step_coords: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        save_attention: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_2d_rope(q, place_coords, step_coords, cos, sin)
        k = apply_2d_rope(k, place_coords, step_coords, cos, sin)

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


class HRMRoPEMLP(nn.Module):
    """Feed-forward network with 3x expansion and GELU."""

    def __init__(self, config: HRMRoPEConfig):
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


class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation conditioning fast state on slow state."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, z_l: torch.Tensor, z_h: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(z_h)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        scale = 1.0 + torch.tanh(gamma)
        return scale * z_l + beta


class FastReasoningBlock(nn.Module):
    """Operational fast block executing arithmetic ALU steps under FiLM control."""

    def __init__(self, config: HRMRoPEConfig):
        super().__init__()
        self.config = config
        self.film = FiLMModulation(config.n_embd)
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = HRMRoPECausalAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = HRMRoPEMLP(config)

    def forward(
        self,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        place_coords: torch.Tensor,
        step_coords: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        scale_factor: float,
        save_attention: bool = False,
    ) -> torch.Tensor:
        z_mod = self.film(self.ln_1(z_l), z_h)
        z_l = z_l + scale_factor * self.attn(
            z_mod, place_coords, step_coords, cos, sin, save_attention=save_attention
        )
        z_l = z_l + scale_factor * self.mlp(self.ln_2(z_l))
        return z_l


class SlowPlanningBlock(nn.Module):
    """Strategic slow block tracking macro column progression and carry state."""

    def __init__(self, config: HRMRoPEConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = HRMRoPECausalAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = HRMRoPEMLP(config)

    def forward(
        self,
        z_h: torch.Tensor,
        z_l_cand: torch.Tensor,
        place_coords: torch.Tensor,
        step_coords: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        scale_factor: float,
        save_attention: bool = False,
    ) -> torch.Tensor:
        gate_input = torch.cat([z_h, z_l_cand], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        z_h = gate * z_h + (1.0 - gate) * z_l_cand
        z_h = z_h + scale_factor * self.attn(
            self.ln_1(z_h), place_coords, step_coords, cos, sin, save_attention=save_attention
        )
        z_h = z_h + scale_factor * self.mlp(self.ln_2(z_h))
        return z_h


class HRMRoPE(BaseReasoningModel):
    """
    Hierarchical Reasoning Model with Decoupled 2D-RoPE.
    Total non-embedding parameters: ~1.57M (strictly < 2M).
    """

    def __init__(self, config: HRMRoPEConfig):
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

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        cos, sin = precompute_2d_rope_freqs(
            config.n_embd // config.n_head,
            max_coords=config.max_coords,
            theta_base=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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
        weights = []
        if self.transformer.fast_block.attn.last_attn_weights is not None:
            weights.append(self.transformer.fast_block.attn.last_attn_weights)
        if self.transformer.slow_block.attn.last_attn_weights is not None:
            weights.append(self.transformer.slow_block.attn.last_attn_weights)
        return weights

    def compute_tensor_2d_coords(
        self,
        idx: torch.Tensor,
        plus_token_id: int = 16,
        eq_token_id: int = 24,
        lparen_token_id: int = 22,
        think_start_id: int = 4,
        think_end_id: int = 5,
        digit_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Macro Place (p) and Micro Step (tau) coordinates for 2D-RoPE."""
        B, T = idx.size()
        device = idx.device
        if digit_ids is None:
            digit_ids = list(range(6, 16))

        place_coords = torch.zeros((B, T), dtype=torch.long, device=device)
        step_coords = torch.zeros((B, T), dtype=torch.long, device=device)

        digit_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        for did in digit_ids:
            digit_mask |= (idx == did)

        for b in range(B):
            row = idx[b]
            d_row = digit_mask[b]

            plus_matches = (row == plus_token_id).nonzero(as_tuple=True)[0]
            plus_idx = plus_matches[0].item() if len(plus_matches) > 0 else -1

            eq_matches = (row == eq_token_id).nonzero(as_tuple=True)[0]
            eq_idx = -1
            if plus_idx != -1:
                for eq_pos in eq_matches:
                    if eq_pos.item() > plus_idx:
                        eq_idx = eq_pos.item()
                        break

            think_s_matches = (row == think_start_id).nonzero(as_tuple=True)[0]
            think_s = think_s_matches[0].item() if len(think_s_matches) > 0 else -1

            think_e_matches = (row == think_end_id).nonzero(as_tuple=True)[0]
            think_e = think_e_matches[0].item() if len(think_e_matches) > 0 else -1

            # Operand 1 (before '+')
            if plus_idx > 0:
                for i in range(plus_idx):
                    if d_row[i]:
                        place_coords[b, i] = plus_idx - 1 - i
                        step_coords[b, i] = 0

            # Operand 2 (between '+' and '=')
            if eq_idx > plus_idx > -1:
                for i in range(plus_idx + 1, eq_idx):
                    if d_row[i]:
                        place_coords[b, i] = eq_idx - 1 - i
                        step_coords[b, i] = 0

            # Inside THINK
            col_count = 0
            if think_s != -1:
                end_pos = think_e if think_e != -1 else T
                cur_col = -1
                cur_step = 0
                for i in range(think_s + 1, end_pos):
                    if row[i] == lparen_token_id:
                        cur_col += 1
                        cur_step = 0
                    if cur_col >= 0:
                        place_coords[b, i] = cur_col
                        step_coords[b, i] = min(cur_step, 9)
                        cur_step += 1
                col_count = cur_col + 1

            # After THINK (final answer)
            if think_e != -1:
                ans_s = -1
                for eq_pos in eq_matches:
                    if eq_pos.item() > think_e:
                        ans_s = eq_pos.item() + 1
                        break
                if ans_s != -1 and ans_s < T:
                    ans_d_idx = 0
                    for i in range(ans_s, T):
                        if d_row[i]:
                            place_coords[b, i] = max(0, col_count - 1 - ans_d_idx)
                            step_coords[b, i] = 0
                            ans_d_idx += 1

        return place_coords, step_coords

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        place_coords: Optional[torch.Tensor] = None,
        step_coords: Optional[torch.Tensor] = None,
        save_attention: bool = False,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        device = idx.device

        # Support keyword passing or positional permutations
        if "targets" in kwargs and targets is None:
            targets = kwargs.pop("targets")

        # Fallback if caller passed model(idx, place_coords, step_coords) positionally
        if targets is not None and place_coords is not None and step_coords is None:
            step_coords = place_coords
            place_coords = targets
            targets = None

        if place_coords is None or step_coords is None:
            if tokenizer is not None:
                plus_id = tokenizer.token_to_id.get("+", 16)
                eq_id = tokenizer.token_to_id.get("=", 24)
                lparen_id = tokenizer.token_to_id.get("(", 22)
                ts_id = tokenizer.token_to_id.get("<THINK>", 4)
                te_id = tokenizer.token_to_id.get("</THINK>", 5)
                d_ids = [tokenizer.token_to_id[str(d)] for d in range(10) if str(d) in tokenizer.token_to_id]
            else:
                plus_id, eq_id, lparen_id, ts_id, te_id = 16, 24, 22, 4, 5
                d_ids = list(range(6, 16))
            p_comp, s_comp = self.compute_tensor_2d_coords(
                idx, plus_id, eq_id, lparen_id, ts_id, te_id, d_ids
            )
            place_coords = place_coords if place_coords is not None else p_comp
            step_coords = step_coords if step_coords is not None else s_comp

        if self.training and self.config.k_rand_shift > 0:
            shifts = torch.randint(0, self.config.k_rand_shift + 1, (B, 1), device=device)
            shift_mask = (place_coords > 0).long()
            place_coords = place_coords + shifts * shift_mask

        # Ensure frequency cache capacity
        max_req = max(place_coords.max().item(), step_coords.max().item()) + 1
        if max_req > self.rope_cos.size(0):
            cos, sin = precompute_2d_rope_freqs(
                self.config.n_embd // self.config.n_head,
                max_coords=max_req * 2,
                theta_base=self.config.rope_theta,
            )
            self.rope_cos = cos.to(device)
            self.rope_sin = sin.to(device)

        # Initial embedding projection
        x = self.transformer.wte(idx)
        z_l = self.transformer.drop(x)
        z_h = z_l.clone()

        total_passes = self.config.n_slow_cycles * self.config.n_fast_steps
        scale_factor = 1.0 / math.sqrt(max(1, total_passes))

        # Hierarchical recurrent unrolling
        for _ in range(self.config.n_slow_cycles):
            for _ in range(self.config.n_fast_steps):
                z_l = self.transformer.fast_block(
                    z_l,
                    z_h,
                    place_coords,
                    step_coords,
                    self.rope_cos,
                    self.rope_sin,
                    scale_factor,
                    save_attention=save_attention,
                )
            z_h = self.transformer.slow_block(
                z_h,
                z_l,
                place_coords,
                step_coords,
                self.rope_cos,
                self.rope_sin,
                scale_factor,
                save_attention=save_attention,
            )

        logits = self.lm_head(self.transformer.ln_f(z_l))

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
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive generation loop with synchronized dynamic 2D coordinates."""
        for _ in range(max_new_tokens):
            logits, _ = self(
                idx,
                targets=None,
                place_coords=None,
                step_coords=None,
                tokenizer=tokenizer,
                **kwargs,
            )
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
