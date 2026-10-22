"""
ScaledGPT: ~18.95M Parameter Recurrent 2D-RoPE Transformer.
Optimized for 25-Hour Dual NVIDIA T4 (sm_75, FP16 AMP, DDP).

Architecture:
- d_model = 512, n_head = 8 (head_dim = 64)
- Physical layers L = 6
- Recurrent passes R = 3 (Effective depth = 18 layers at zero parameter overhead)
- Residual recurrence update: x = x + (1/sqrt(R)) * (x_res - x)
- Tied embedding and output projection weights (wte.weight == lm_head.weight)
- Decoupled 2D-RoPE (Macro place p, Micro step tau)
- Selective shift masking: (place_coords > 0).long()
- O(1) SDPA causal attention
- Non-embedding params: ~18,887,680 | Total params: ~18,953,216 (~18.95M target)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from champion_model.scaled_t4.tokenizer import ScaledTokenizer
except ImportError:
    try:
        from tokenizer import ScaledTokenizer
    except ImportError:
        from champion_model.unified_200k.tokenizer import UnifiedTokenizer as ScaledTokenizer


@dataclass
class ScaledGPTConfig:
    """Hyperparameters for ScaledGPT (~18.95M parameters)."""
    vocab_size: int = 115
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    n_recur: int = 3
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    max_coords: int = 4096
    k_rand_shift: int = 32
    pos_emb_type: str = "rope_2d"


def precompute_2d_rope_freqs(
    head_dim: int, max_coords: int = 4096, theta_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes rotary frequency tables (cos, sin) for decoupled 2D-RoPE."""
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
    """Applies decoupled 2D-RoPE to tensor of shape [B, n_head, T, head_dim]."""
    B, H, T, D = x.shape
    half_d = D // 2

    max_c = cos.size(0) - 1
    p_clamped = torch.clamp(place_coords, 0, max_c)
    s_clamped = torch.clamp(step_coords, 0, max_c)

    cos_p = cos[p_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)
    sin_p = sin[p_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)
    cos_s = cos[s_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)
    sin_s = sin[s_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)

    x_place = x[..., :half_d]
    x_step = x[..., half_d:]

    # Place subspace rotation
    x_p_even = x_place[..., 0::2]
    x_p_odd = x_place[..., 1::2]
    rot_p_even = x_p_even * cos_p - x_p_odd * sin_p
    rot_p_odd = x_p_even * sin_p + x_p_odd * cos_p
    rot_place = torch.empty_like(x_place)
    rot_place[..., 0::2] = rot_p_even
    rot_place[..., 1::2] = rot_p_odd

    # Step subspace rotation
    x_s_even = x_step[..., 0::2]
    x_s_odd = x_step[..., 1::2]
    rot_s_even = x_s_even * cos_s - x_s_odd * sin_s
    rot_s_odd = x_s_even * sin_s + x_s_odd * cos_s
    rot_step = torch.empty_like(x_step)
    rot_step[..., 0::2] = rot_s_even
    rot_step[..., 1::2] = rot_s_odd

    return torch.cat([rot_place, rot_step], dim=-1)


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with SDPA and 2D-RoPE."""

    def __init__(self, config: ScaledGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        cos, sin = precompute_2d_rope_freqs(
            self.head_dim, max_coords=config.max_coords, theta_base=config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        place_coords: torch.Tensor,
        step_coords: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Dynamic frequency expansion if coordinates exceed precomputed length
        max_coord = max(int(place_coords.max().item()), int(step_coords.max().item())) + 1
        if max_coord > self.rope_cos.size(0):
            new_max = max(max_coord * 2, 4096)
            cos, sin = precompute_2d_rope_freqs(self.head_dim, max_coords=new_max, theta_base=self.rope_theta)
            self.rope_cos = cos.to(x.device)
            self.rope_sin = sin.to(x.device)

        q = apply_2d_rope(q, place_coords, step_coords, self.rope_cos, self.rope_sin)
        k = apply_2d_rope(k, place_coords, step_coords, self.rope_cos, self.rope_sin)

        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network with 4x expansion and GELU."""

    def __init__(self, config: ScaledGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block with Pre-LN residuals."""

    def __init__(self, config: ScaledGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        place_coords: torch.Tensor,
        step_coords: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), place_coords, step_coords)
        x = x + self.mlp(self.ln_2(x))
        return x


class ScaledGPT(nn.Module):
    """Scaled ~18.95M Parameter Recurrent 2D-RoPE Transformer (sm_75 Dual T4)."""

    def __init__(self, config: ScaledGPTConfig, tokenizer: Optional[ScaledTokenizer] = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer or ScaledTokenizer()

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: wte.weight == lm_head.weight
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def compute_2d_coords(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified 2D coordinate mapper:
        1. If <USER> is present, routes as Discourse 2D coordinates (segment p in [0, 4], token offset tau).
           In multi-turn dialogues, handles turns gracefully.
        2. Else, routes as Arithmetic 2D coordinates (column rank p up to 50+, intra-column step tau in [0, 9]).
        """
        B, T = idx.size()
        device = idx.device
        tok = self.tokenizer

        p_coords = torch.zeros((B, T), dtype=torch.long, device=device)
        s_coords = torch.zeros((B, T), dtype=torch.long, device=device)

        user_s_id = tok.user_s_id
        plus_id = tok.plus_id
        eq_id = tok.eq_id
        think_s_id = tok.think_s_id
        think_e_id = tok.think_e_id
        lparen_id = tok.lparen_id
        digit_ids = set(tok.digit_ids)

        segment_tags = {
            tok.user_s_id: 0,
            tok.think_s_id: 1,
            tok.call_s_id: 2,
            tok.result_s_id: 3,
            tok.answer_s_id: 4,
        }

        for b in range(B):
            row = idx[b]
            is_dialogue = (row == user_s_id).any().item()

            if is_dialogue:
                # Discourse routing
                cur_seg = 0
                cur_step = 0
                for t in range(T):
                    token_id = row[t].item()
                    if token_id in segment_tags:
                        cur_seg = segment_tags[token_id]
                        cur_step = 0
                    p_coords[b, t] = cur_seg
                    s_coords[b, t] = cur_step
                    cur_step += 1
            else:
                # Arithmetic scratchpad routing (extrapolates cleanly from 1 to 50+ digits)
                plus_matches = (row == plus_id).nonzero(as_tuple=True)[0]
                plus_idx = plus_matches[0].item() if len(plus_matches) > 0 else -1

                eq_matches = (row == eq_id).nonzero(as_tuple=True)[0]
                eq_idx = -1
                if plus_idx != -1:
                    for pos in eq_matches:
                        if pos.item() > plus_idx:
                            eq_idx = pos.item()
                            break

                think_s_matches = (row == think_s_id).nonzero(as_tuple=True)[0]
                think_s = think_s_matches[0].item() if len(think_s_matches) > 0 else -1

                think_e_matches = (row == think_e_id).nonzero(as_tuple=True)[0]
                think_e = think_e_matches[0].item() if len(think_e_matches) > 0 else -1

                # Operand 1 (before '+')
                if plus_idx > 0:
                    for i in range(plus_idx):
                        if row[i].item() in digit_ids:
                            p_coords[b, i] = plus_idx - 1 - i
                            s_coords[b, i] = 0

                # Operand 2 (between '+' and '=')
                if eq_idx > plus_idx > -1:
                    for i in range(plus_idx + 1, eq_idx):
                        if row[i].item() in digit_ids:
                            p_coords[b, i] = eq_idx - 1 - i
                            s_coords[b, i] = 0

                # Inside THINK
                col_count = 0
                if think_s != -1:
                    end_pos = think_e if think_e != -1 else T
                    cur_col = -1
                    cur_step = 0
                    for i in range(think_s + 1, end_pos):
                        if row[i].item() == lparen_id:
                            cur_col += 1
                            cur_step = 0
                        if cur_col >= 0:
                            p_coords[b, i] = cur_col
                            s_coords[b, i] = min(cur_step, 9)
                            cur_step += 1
                    col_count = cur_col + 1

                # After THINK (final answer)
                if think_e != -1:
                    ans_s = -1
                    for pos in eq_matches:
                        if pos.item() > think_e:
                            ans_s = pos.item() + 1
                            break
                    if ans_s != -1 and ans_s < T:
                        ans_d_idx = 0
                        for i in range(ans_s, T):
                            if row[i].item() in digit_ids:
                                p_coords[b, i] = max(0, col_count - 1 - ans_d_idx)
                                s_coords[b, i] = 0
                                ans_d_idx += 1

        return p_coords, s_coords

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        place_coords: Optional[torch.Tensor] = None,
        step_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        device = idx.device

        if place_coords is None or step_coords is None:
            place_coords, step_coords = self.compute_2d_coords(idx)

        # Shift invariance data augmentation during training:
        # Strictly applies selective shift masking (place_coords > 0).long()
        if self.training and self.config.k_rand_shift > 0:
            shifts = torch.randint(0, self.config.k_rand_shift + 1, (B, 1), device=device)
            shift_mask = (place_coords > 0).long()
            place_coords = place_coords + shifts * shift_mask

        x = self.transformer.drop(self.transformer.wte(idx))

        # Per-block recurrence residual updates: x = x + (1/sqrt(R)) * (x_res - x)
        n_recur = max(1, self.config.n_recur)
        scale_factor = 1.0 / math.sqrt(n_recur) if n_recur > 1 else 1.0
        for _ in range(n_recur):
            for block in self.transformer.h:
                x_res = block(x, place_coords, step_coords)
                if n_recur > 1:
                    x = x + scale_factor * (x_res - x)
                else:
                    x = x_res

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Autoregressive token generation with greedy / multinomial sampling."""
        stop_set = set(stop_tokens) if stop_tokens is not None else set()

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() in stop_set:
                break

        return idx
