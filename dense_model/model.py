"""
Dense Autoregressive Transformer Model (MathGPT).
Self-contained stand-alone implementation.

Features:
1. Standard causal multi-head self-attention with SDPA (Scaled Dot-Product Attention).
2. Support for Rotary Position Embeddings (RoPE) and Absolute Positional Embeddings.
3. Pre-LayerNorm residual architecture with GELU MLP.
4. Tied embedding and LM-head projection weights.
5. Memory-efficient chunked cross-entropy for large vocabularies.
6. Standalone autoregressive generation with temperature, top-k, and stop token filtering.
Turing sm_75 FP16 AMP compliant with Cutlass SDPA kernels.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters for MathGPT."""
    vocab_size: int = 25
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
    pos_emb_type: str = "absolute"
    use_pos_emb: Optional[bool] = None
    rope_theta: float = 10000.0
    max_coords: int = 2048
    model_type: str = "standard"
    n_recur: int = 1
    k_rand_shift: int = 0

    def __post_init__(self):
        if self.use_pos_emb is not None:
            if not self.use_pos_emb:
                self.pos_emb_type = "none"
            elif self.pos_emb_type == "none":
                self.pos_emb_type = "absolute"
        else:
            self.use_pos_emb = (self.pos_emb_type == "absolute")

    @classmethod
    def create_100k(
        cls,
        vocab_size: int = 27,
        block_size: int = 512,
        dropout: float = 0.0,
        pos_emb_type: str = "rope",
        n_recur: int = 1,
        k_rand_shift: int = 0,
    ) -> "GPTConfig":
        """Factory preset for ~100k parameter model (100,672 total, 98,944 non-embedding)."""
        return cls(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=dropout,
            bias=False,
            pos_emb_type=pos_emb_type,
            model_type="standard",
            n_recur=n_recur,
            k_rand_shift=k_rand_shift,
        )


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int = 1024, theta_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes rotary frequency tables (cos, sin) for RoPE embeddings."""
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    freqs = 1.0 / (theta_base ** (dim_indices / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embeddings to query or key tensor of shape [B, n_head, T, head_dim]."""
    T = x.size(2)
    cos_t = cos[:T, :].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin_t = sin[:T, :].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rot_even = x_even * cos_t - x_odd * sin_t
    rot_odd = x_even * sin_t + x_odd * cos_t

    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


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
    """Multi-head causal self-attention supporting RoPE, 2D-RoPE, and SDPA with attention saving."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.pos_emb_type = config.pos_emb_type

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        if self.pos_emb_type == "rope":
            max_seq = max(config.block_size * 4, 1024)
            cos, sin = precompute_rope_freqs(
                self.head_dim, max_seq_len=max_seq, theta_base=config.rope_theta
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        elif self.pos_emb_type == "rope_2d":
            cos_2d, sin_2d = precompute_2d_rope_freqs(
                self.head_dim, max_coords=config.max_coords, theta_base=config.rope_theta
            )
            self.register_buffer("rope_2d_cos", cos_2d, persistent=False)
            self.register_buffer("rope_2d_sin", sin_2d, persistent=False)

        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        place_coords: Optional[torch.Tensor] = None,
        step_coords: Optional[torch.Tensor] = None,
        save_attention: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.pos_emb_type == "rope":
            if T > self.rope_cos.size(0):
                cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len=T * 2)
                self.rope_cos = cos.to(x.device)
                self.rope_sin = sin.to(x.device)
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        elif self.pos_emb_type == "rope_2d" and place_coords is not None and step_coords is not None:
            q = apply_2d_rope(q, place_coords, step_coords, self.rope_2d_cos, self.rope_2d_sin)
            k = apply_2d_rope(k, place_coords, step_coords, self.rope_2d_cos, self.rope_2d_sin)

        if save_attention:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if T <= self.bias.size(-1):
                mask = self.bias[:, :, :T, :T]
            else:
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


class MLP(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig):
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


class Block(nn.Module):
    """Transformer decoder block with Pre-LayerNorm and residual streams."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        place_coords: Optional[torch.Tensor] = None,
        step_coords: Optional[torch.Tensor] = None,
        save_attention: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x),
            place_coords=place_coords,
            step_coords=step_coords,
            save_attention=save_attention,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class MathGPT(nn.Module):
    """Decoder-only dense Transformer for arithmetic and algorithmic reasoning."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        transformer_dict = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        )
        if config.pos_emb_type == "absolute":
            transformer_dict["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
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
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns total parameter count, optionally excluding token embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            if self.config.pos_emb_type == "absolute" and "wpe" in self.transformer:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()

        if self.config.pos_emb_type == "absolute":
            assert t <= self.config.block_size, (
                f"Sequence length {t} exceeds absolute block size {self.config.block_size}"
            )
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            tok_emb = self.transformer.wte(idx)
            x = self.transformer.drop(tok_emb)

        if self.config.pos_emb_type == "rope_2d" and (place_coords is None or step_coords is None):
            place_coords, step_coords = self.compute_tensor_2d_coords(idx)

        # Shift invariance data augmentation during training
        if self.training and self.config.k_rand_shift > 0 and place_coords is not None:
            shifts = torch.randint(0, self.config.k_rand_shift + 1, (b, 1), device=device)
            shift_mask = (place_coords > 0).long()
            place_coords = place_coords + shifts * shift_mask

        # Recurrent unrolling across the blocks
        n_recur = getattr(self.config, "n_recur", 1)
        scale_factor = 1.0 / math.sqrt(max(1, n_recur)) if n_recur > 1 else 1.0
        for _ in range(n_recur):
            for block in self.transformer.h:
                x_res = block(
                    x,
                    place_coords=place_coords,
                    step_coords=step_coords,
                    save_attention=save_attention,
                )
                if n_recur > 1:
                    x = x + scale_factor * (x_res - x)
                else:
                    x = x_res
        x = self.transformer.ln_f(x)

        loss = None
        if targets is not None:
            targets_flat = targets.view(-1)
            x_flat = x.view(-1, x.size(-1))
            # Memory-efficient chunked cross-entropy to prevent CUDA OOM on large vocabularies (e.g. BPE 50k)
            if self.config.vocab_size > 1024 and x_flat.size(0) > 2048:
                chunk_size = 2048
                total_loss = 0.0
                total_valid = 0
                for i in range(0, x_flat.size(0), chunk_size):
                    chunk_t = targets_flat[i : i + chunk_size]
                    mask = (chunk_t != -100)
                    if mask.any():
                        chunk_x = x_flat[i : i + chunk_size]
                        chunk_logits = self.lm_head(chunk_x)
                        chunk_loss = F.cross_entropy(
                            chunk_logits,
                            chunk_t,
                            ignore_index=-100,
                            reduction="sum",
                        )
                        total_loss = total_loss + chunk_loss
                        total_valid += mask.sum().item()
                loss = total_loss / max(1, total_valid)
                logits = None if self.training else self.lm_head(x)
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets_flat,
                    ignore_index=-100,
                )
        else:
            logits = self.lm_head(x)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        save_attention: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation loop supporting greedy decoding, temperature scaling, and top-k."""
        for _ in range(max_new_tokens):
            if self.config.pos_emb_type == "absolute":
                idx_cond = idx[:, -self.config.block_size:]
            else:
                idx_cond = idx

            logits, _ = self(idx_cond, save_attention=save_attention)
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

            if idx.size(0) == 1:
                next_token = idx_next.item()
                if eos_id is not None and next_token == eos_id:
                    break
                if stop_tokens is not None and next_token in stop_tokens:
                    break
            elif eos_id is not None and (idx_next == eos_id).all():
                break

        return idx

    def get_attention_maps(self) -> List[torch.Tensor]:
        """Returns captured attention matrices from blocks where save_attention was enabled."""
        return [
            block.attn.last_attn_weights
            for block in self.transformer.h
            if block.attn.last_attn_weights is not None
        ]
