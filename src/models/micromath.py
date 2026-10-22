"""
MicroMath-2M: Compact Algorithmic Reasoning Transformer for Invariant Arithmetic.
Features:
1. Strictly < 2M parameters (~1.37M total, ~1.33M non-embedding).
2. NoPE (No Positional Encoding) + Abacus Decimal Place Embeddings.
3. Translation-invariant column alignment via random shift jitter during training.
4. Tied embedding and LM-head weights.
5. In-model dynamic place-id tracking for 100% training-generation synchronization.
Turing sm_75 FP16 AMP compliant with Cutlass SDPA.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseReasoningModel, RMSNorm


@dataclass
class MicroMathConfig:
    """Hyperparameters for MicroMath-2M."""
    vocab_size: int = 32
    block_size: int = 512
    n_embd: int = 192
    n_head: int = 4
    n_layer: int = 3
    d_ffn: int = 768
    dropout: float = 0.0
    bias: bool = False
    max_places: int = 128
    k_rand_shift: int = 32
    model_type: str = "micromath"


class MicroMathAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with Cutlass SDPA and NoPE.
    Relies purely on causal masking and Abacus place embeddings for routing.
    """

    def __init__(self, config: MicroMathConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, save_attention: bool = False) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if save_attention:
            att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
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


class MicroMathMLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: MicroMathConfig):
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


class MicroMathBlock(nn.Module):
    """Transformer block with RMSNorm, attention, and MLP."""

    def __init__(self, config: MicroMathConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = MicroMathAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MicroMathMLP(config)

    def forward(self, x: torch.Tensor, save_attention: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), save_attention=save_attention)
        x = x + self.mlp(self.ln_2(x))
        return x


class MicroMath2M(BaseReasoningModel):
    """
    MicroMath-2M: Pure Attention + Abacus Positional Reasoning Model.
    Total parameters: ~1.36M (strictly < 2M).
    """

    def __init__(self, config: MicroMathConfig):
        super().__init__()
        self.config = config

        total_places = config.max_places + config.k_rand_shift + 1
        self.wpe_abacus = nn.Embedding(total_places, config.n_embd)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([MicroMathBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
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

    def compute_tensor_place_ids(
        self,
        idx: torch.Tensor,
        plus_token_id: int,
        eq_token_id: int,
        lparen_token_id: int,
        think_start_id: int,
        think_end_id: int,
        digit_ids: List[int],
    ) -> torch.Tensor:
        """
        Fast tensor place ID computation executed on GPU.
        Aligns digit places relative to arithmetic boundaries.
        """
        B, T = idx.size()
        device = idx.device
        place_ids = torch.zeros((B, T), dtype=torch.long, device=device)
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

            # Operand 1
            if plus_idx > 0:
                for i in range(plus_idx):
                    if d_row[i]:
                        place_ids[b, i] = plus_idx - 1 - i

            # Operand 2
            if eq_idx > plus_idx > -1:
                for i in range(plus_idx + 1, eq_idx):
                    if d_row[i]:
                        place_ids[b, i] = eq_idx - 1 - i

            # Inside THINK
            col_count = 0
            if think_s != -1:
                end_pos = think_e if think_e != -1 else T
                cur_col = -1
                for i in range(think_s + 1, end_pos):
                    if row[i] == lparen_token_id:
                        cur_col += 1
                    if cur_col >= 0:
                        place_ids[b, i] = cur_col
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
                            place_ids[b, i] = max(0, col_count - 1 - ans_d_idx)
                            ans_d_idx += 1

        return place_ids

    def forward(
        self,
        idx: torch.Tensor,
        place_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        save_attention: bool = False,
        tokenizer: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        device = idx.device

        if place_ids is None:
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
            place_ids = self.compute_tensor_place_ids(
                idx, plus_id, eq_id, lparen_id, ts_id, te_id, d_ids
            )

        if self.training and self.config.k_rand_shift > 0:
            shifts = torch.randint(0, self.config.k_rand_shift + 1, (B, 1), device=device)
            shift_mask = (place_ids > 0).long()
            place_ids_effective = place_ids + shifts * shift_mask
        else:
            place_ids_effective = place_ids

        place_ids_effective = torch.clamp(
            place_ids_effective, 0, self.wpe_abacus.num_embeddings - 1
        )

        tok_emb = self.transformer.wte(idx)
        plc_emb = self.wpe_abacus(place_ids_effective)
        x = self.transformer.drop(tok_emb + plc_emb)

        for block in self.transformer.h:
            x = block(x, save_attention=save_attention)

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
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive generation loop with synchronized dynamic place coordinates."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx, place_ids=None, tokenizer=tokenizer, **kwargs)
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
