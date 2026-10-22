"""
ToolGPT: ~100k Parameter Recurrent 2D-Discourse RoPE Language Model for Tool Calling.

Features:
1. ~107k total parameters (99,840 non-embedding, vocab=111).
2. Hierarchical 2D Discourse RoPE:
   - Macro Coordinate (p): Discourse Segment (USER -> THINK -> CALL -> RESULT -> ANSWER).
   - Micro Coordinate (tau): Token position within the current discourse segment.
3. Recurrent Unrolling (R=3 passes through 2 layers with residual scaling, depth=6 equivalent).
4. Full greedy/temperature generation with stop token handling.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tool_model.tokenizer import ToolTokenizer


@dataclass
class ToolGPTConfig:
    """Hyperparameters for ToolGPT."""
    vocab_size: int = 111
    block_size: int = 512
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    max_coords: int = 2048
    n_recur: int = 3
    k_rand_shift: int = 8
    pos_emb_type: str = "rope_2d"  # "rope_2d" or "learned_2d"
    max_segments: int = 16


def precompute_2d_rope_freqs(
    head_dim: int, max_coords: int = 2048, theta_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes frequency tables for dual-coordinate 2D-RoPE."""
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

    p_even = x_place[..., 0::2]
    p_odd = x_place[..., 1::2]
    rot_p_even = p_even * cos_p - p_odd * sin_p
    rot_p_odd = p_even * sin_p + p_odd * cos_p
    rot_place = torch.empty_like(x_place)
    rot_place[..., 0::2] = rot_p_even
    rot_place[..., 1::2] = rot_p_odd

    s_even = x_step[..., 0::2]
    s_odd = x_step[..., 1::2]
    rot_s_even = s_even * cos_s - s_odd * sin_s
    rot_s_odd = s_even * sin_s + s_odd * cos_s
    rot_step = torch.empty_like(x_step)
    rot_step[..., 0::2] = rot_s_even
    rot_step[..., 1::2] = rot_s_odd

    return torch.cat([rot_place, rot_step], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with 2D Discourse RoPE."""

    def __init__(self, config: ToolGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.pos_emb_type = getattr(config, "pos_emb_type", "rope_2d")
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
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

        if getattr(self, "pos_emb_type", "rope_2d") == "rope_2d":
            # Dynamic frequency expansion if needed
            max_coord = max(place_coords.max().item(), step_coords.max().item()) + 1
            if max_coord > self.rope_cos.size(0):
                cos, sin = precompute_2d_rope_freqs(self.head_dim, max_coords=max_coord * 2)
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
    """Position-wise feed-forward network with 4x expansion and GELU."""

    def __init__(self, config: ToolGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer decoder block with LayerNorm and Pre-LN residuals."""

    def __init__(self, config: ToolGPTConfig):
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


class ToolGPT(nn.Module):
    """
    ToolGPT: ~100k Recurrent 2D-Discourse RoPE Language Model.
    """

    def __init__(self, config: ToolGPTConfig, tokenizer: Optional[ToolTokenizer] = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer if tokenizer is not None else ToolTokenizer()

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        )
        if config.pos_emb_type == "learned_2d":
            half_d = config.n_embd // 2
            transformer_modules["wpe_p"] = nn.Embedding(config.max_segments, half_d)
            transformer_modules["wpe_tau"] = nn.Embedding(config.block_size, half_d)

        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def compute_discourse_2d_coords(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps token sequences into 2D Discourse coordinates:
        p:   Discourse segment (0=USER, 1=THINK, 2=CALL, 3=RESULT, 4=ANSWER)
        tau: Token index within the current discourse segment.
        """
        B, T = idx.size()
        device = idx.device
        tok = self.tokenizer

        p_coords = torch.zeros((B, T), dtype=torch.long, device=device)
        s_coords = torch.zeros((B, T), dtype=torch.long, device=device)

        # Segment start token IDs
        segment_tags = {
            tok.user_s_id: 0,
            tok.think_s_id: 1,
            tok.call_s_id: 2,
            tok.result_s_id: 3,
            tok.answer_s_id: 4,
        }

        for b in range(B):
            row = idx[b]
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
            place_coords, step_coords = self.compute_discourse_2d_coords(idx)

        # Training place shift augmentation
        if self.training and self.config.k_rand_shift > 0:
            shifts = torch.randint(0, self.config.k_rand_shift + 1, (B, 1), device=device)
            place_coords = place_coords + shifts

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_emb_type == "learned_2d":
            max_seg = self.config.max_segments - 1
            max_step = self.config.block_size - 1
            p_clamped = torch.clamp(place_coords, 0, max_seg)
            s_clamped = torch.clamp(step_coords, 0, max_step)
            emb_p = self.transformer.wpe_p(p_clamped)
            emb_tau = self.transformer.wpe_tau(s_clamped)
            pos_emb = torch.cat([emb_p, emb_tau], dim=-1)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        # Recurrent unrolling across the blocks
        n_recur = self.config.n_recur
        scale_factor = 1.0 / math.sqrt(max(1, n_recur)) if n_recur > 1 else 1.0
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
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Autoregressive generation loop with stop token support."""
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_id]

        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if idx.size(0) == 1 and idx_next.item() in stop_tokens:
                break

        return idx
