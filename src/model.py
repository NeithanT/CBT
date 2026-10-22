import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
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

    def __post_init__(self):
        if self.use_pos_emb is not None:
            if not self.use_pos_emb:
                self.pos_emb_type = "none"
            elif self.pos_emb_type == "none":
                self.pos_emb_type = "absolute"
        else:
            self.use_pos_emb = (self.pos_emb_type == "absolute")


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int = 1024, theta_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    freqs = 1.0 / (theta_base ** (dim_indices / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention supporting RoPE, NoPE, and SDPA with attention saving."""

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

        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, save_attention: bool = False) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, save_attention: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), save_attention=save_attention)
        x = x + self.mlp(self.ln_2(x))
        return x


class MathGPT(nn.Module):
    """Decoder-only Transformer for arithmetic and algorithmic reasoning."""

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
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            if self.config.pos_emb_type == "absolute" and "wpe" in self.transformer:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
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

        for block in self.transformer.h:
            x = block(x, save_attention=save_attention)
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

            next_token = idx_next.item()
            if eos_id is not None and next_token == eos_id:
                break
            if stop_tokens is not None and next_token in stop_tokens:
                break

        return idx

    def get_attention_maps(self) -> List[torch.Tensor]:
        return [
            block.attn.last_attn_weights
            for block in self.transformer.h
            if block.attn.last_attn_weights is not None
        ]


# Expose Phase 2 architectures and factory for unified access
from src.models.looped import LoopedMathGPT, LoopedGPTConfig
from src.models.hierarchical import HierarchicalReasoningGPT, HierarchicalGPTConfig
from src.models.factory import create_model

