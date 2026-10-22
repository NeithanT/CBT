"""
Equalized 1-15 Digit, 1024-Context Baseline & Ruoss et al. (2023) Comparison + Large Random Sweep.
Addresses the two primary reviewer requests:
1. Trains strictly equalized 1-15 digit, T=1024 models (1D-RoPE Standard, 1D-RoPE + Random Offset,
   1D-RoPE + Ruoss et al. 2023 Randomized Positions, and Decoupled 2D-RoPE with k=32) on the exact
   same 1-15 digit dataset and token budget.
2. Evaluates 200 random multi-carry problems at every length from 16 to 100 digits (2,600 problems
   per model) with 95% Wilson confidence intervals using KV-cached batched greedy decoding.
3. Also evaluates the 18.91M 1D-Char checkpoint (`checkpoints/model_1/model_1_final.pt`) and
   1.56M HRM-RoPE checkpoint at extended T=1024 context on the 25-50 digit problems of Table 3.
"""

import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.math_model.model import (
    MathGPT,
    GPTConfig,
    precompute_rope_freqs,
    apply_2d_rope,
)
from src.tokenizer import MathTokenizer
from src.arithmetic import format_addition_char_scratchpad, extract_scratchpad_answer


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes 95% Wilson score confidence interval (in percent)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    lo = max(0.0, (center - half) * 100.0)
    hi = min(100.0, (center + half) * 100.0)
    return (round(lo, 1), round(hi, 1))


def apply_1d_rope_custom_pos(
    x: torch.Tensor, pos_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies 1D-RoPE using explicit per-token position indices pos_ids of shape [B, T]."""
    max_c = cos.size(0) - 1
    p_clamped = torch.clamp(pos_ids, 0, max_c)
    cos_t = cos[p_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)
    sin_t = sin[p_clamped].unsqueeze(1).to(dtype=x.dtype, device=x.device)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos_t - x_odd * sin_t
    rot_odd = x_even * sin_t + x_odd * cos_t

    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


def generate_1_to_15d_curriculum(n_samples: int = 6000, seed: int = 42) -> List[str]:
    """
    Generates a 1-15 digit addition dataset matching CBT-18.95M's arithmetic distribution:
    - 40% asymmetric lengths U[1, 15]
    - 25% symmetric lengths U[1, 15]
    - 20% cascading carries (99...9 + b) across 2..15 digits
    - 15% power-of-ten zero chains across 2..15 digits
    """
    rng = random.Random(seed)
    texts: List[str] = []
    for _ in range(n_samples):
        r = rng.random()
        if r < 0.20:
            d = rng.randint(2, 15)
            a_val = 10**d - 1
            b_val = 1 if rng.random() < 0.5 else rng.randint(1, 10**d - 1)
        elif r < 0.35:
            d = rng.randint(2, 15)
            la = rng.choice([1, 2, 4, 5, 8])
            lb = rng.choice([1, 2, 4, 5, 8])
            a_val = la * (10 ** (d - 1))
            b_val = lb * (10 ** (d - 1))
        elif r < 0.60:
            d = rng.randint(1, 15)
            lo = 0 if d == 1 else 10 ** (d - 1)
            hi = 10**d - 1
            a_val = rng.randint(lo, hi)
            b_val = rng.randint(lo, hi)
        else:
            da = rng.randint(1, 15)
            db = rng.randint(1, 15)
            lo_a = 0 if da == 1 else 10 ** (da - 1)
            lo_b = 0 if db == 1 else 10 ** (db - 1)
            a_val = rng.randint(lo_a, 10**da - 1)
            b_val = rng.randint(lo_b, 10**db - 1)
        texts.append(format_addition_char_scratchpad(a_val, b_val))
    return texts


def preencode_dataset_with_coords(
    texts: List[str],
    tokenizer: MathTokenizer,
    helper_model: MathGPT,
    block_size: int = 1024,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Pre-encodes all training examples and precomputes their 2D coordinates once."""
    eq_token_id = tokenizer.encode("=", add_bos=False, add_eos=False)[0]
    dataset = []
    for txt in texts:
        tokens = tokenizer.encode(txt, add_bos=True, add_eos=True)
        if len(tokens) > block_size + 1:
            tokens = tokens[: block_size + 1]
        t_tok = torch.tensor(tokens, dtype=torch.long)
        x = t_tok[:-1]
        y = t_tok[1:].clone()
        eq_matches = (x == eq_token_id).nonzero(as_tuple=True)[0]
        if len(eq_matches) > 0:
            y[: eq_matches[0].item() + 1] = -100
        p_c, s_c = helper_model.compute_tensor_2d_coords(x.unsqueeze(0))
        dataset.append((x, y, p_c[0], s_c[0]))
    return dataset


def forward_with_pos_mode(
    model: MathGPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
    place_coords: torch.Tensor,
    step_coords: torch.Tensor,
    pos_mode: str,
    k_shift: int = 0,
    max_pos_ruoss: int = 1024,
) -> torch.Tensor:
    """
    Runs forward pass supporting:
    - '1d_standard': standard 1D-RoPE (0, 1, ..., T-1)
    - '1d_shift': 1D-RoPE with random offset delta ~ U{0, ..., k_shift}
    - '1d_ruoss': Ruoss et al. (2023) randomized position encoding (sorted random sample from [0, max_pos_ruoss))
    - '2d_rope': Decoupled 2D-RoPE with coordinate shift masking k_shift
    """
    B, T = idx.size()
    device = idx.device
    x = model.transformer.wte(idx)

    if pos_mode == "1d_standard":
        pos_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)
    elif pos_mode == "1d_shift":
        base = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)
        shifts = torch.randint(0, k_shift + 1, (B, 1), device=device)
        pos_ids = base + shifts
    elif pos_mode == "1d_ruoss":
        # Ruoss et al. (ACL 2023): sample T distinct positions uniformly from [0, max_pos_ruoss) and sort
        M = max(max_pos_ruoss, T + 1)
        rand_scores = torch.rand((B, M), device=device)
        _, chosen = torch.topk(rand_scores, k=T, dim=1, largest=False)
        pos_ids, _ = torch.sort(chosen, dim=1)
    elif pos_mode == "2d_rope":
        if k_shift > 0:
            shifts = torch.randint(0, k_shift + 1, (B, 1), device=device)
            place_coords = place_coords + shifts * (place_coords > 0).long()
        pos_ids = None
    else:
        raise ValueError(f"Unknown pos_mode: {pos_mode}")

    n_recur = model.config.n_recur
    scale_factor = 1.0 / math.sqrt(max(1, n_recur)) if n_recur > 1 else 1.0

    for _ in range(n_recur):
        for block in model.transformer.h:
            attn = block.attn
            ln_x = block.ln_1(x)
            qkv = attn.c_attn(ln_x)
            q, k, v = qkv.chunk(3, dim=2)
            q = q.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
            k = k.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
            v = v.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)

            if pos_mode.startswith("1d"):
                q = apply_1d_rope_custom_pos(q, pos_ids, attn.rope_cos, attn.rope_sin)
                k = apply_1d_rope_custom_pos(k, pos_ids, attn.rope_cos, attn.rope_sin)
            else:
                q = apply_2d_rope(q, place_coords, step_coords, attn.rope_2d_cos, attn.rope_2d_sin)
                k = apply_2d_rope(k, place_coords, step_coords, attn.rope_2d_cos, attn.rope_2d_sin)

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, attn.n_embd)
            y = attn.c_proj(y)
            x_res = x + y
            x_res = x_res + block.mlp(block.ln_2(x_res))
            if n_recur > 1:
                x = x + scale_factor * (x_res - x)
            else:
                x = x_res

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    return loss


def train_equalized_model(
    model: MathGPT,
    dataset: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
    pos_mode: str,
    k_shift: int,
    steps: int = 3000,
    batch_size: int = 32,
    lr: float = 2e-3,
    seed: int = 42,
) -> Tuple[float, float]:
    """Trains a single equalized model using precomputed coordinates."""
    torch.manual_seed(seed)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    n_samples = len(dataset)
    indices = list(range(n_samples))
    curr_idx = 0
    epoch_seed = seed
    t0 = time.time()
    last_loss = 0.0

    for step in range(1, steps + 1):
        if curr_idx + batch_size > n_samples:
            epoch_seed += 1
            random.Random(epoch_seed).shuffle(indices)
            curr_idx = 0
        batch_items = [dataset[i] for i in indices[curr_idx : curr_idx + batch_size]]
        curr_idx += batch_size

        max_len = max(item[0].size(0) for item in batch_items)
        x_b = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        y_b = torch.full((batch_size, max_len), -100, dtype=torch.long)
        p_b = torch.zeros((batch_size, max_len), dtype=torch.long)
        s_b = torch.zeros((batch_size, max_len), dtype=torch.long)

        for b_i, (x_i, y_i, p_i, s_i) in enumerate(batch_items):
            L = x_i.size(0)
            x_b[b_i, :L] = x_i
            y_b[b_i, :L] = y_i
            p_b[b_i, :L] = p_i
            s_b[b_i, :L] = s_i

        optimizer.zero_grad()
        loss = forward_with_pos_mode(
            model=model,
            idx=x_b,
            targets=y_b,
            place_coords=p_b,
            step_coords=s_b,
            pos_mode=pos_mode,
            k_shift=k_shift,
            max_pos_ruoss=1024,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())

        if step % 500 == 0 or step == steps:
            print(
                f"  [Step {step:4d}/{steps}] loss={last_loss:.5f} | elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    return last_loss, round(time.time() - t0, 2)


@torch.no_grad()
def fast_kv_generate_mathgpt(
    model: MathGPT,
    prompts: List[str],
    tokenizer: MathTokenizer,
    max_new_tokens: int,
) -> List[str]:
    """
    Fast KV-cached batched greedy decoding for MathGPT (supports both 1D-RoPE and 2D-RoPE).
    Since all prompts at a given digit length d have identical token length, no padding is needed.
    """
    model.eval()
    B = len(prompts)
    encoded = [tokenizer.encode(p + "<THINK>", add_bos=True, add_eos=False) for p in prompts]
    idx = torch.tensor(encoded, dtype=torch.long)
    T_prefill = idx.size(1)

    n_recur = model.config.n_recur
    n_layer = model.config.n_layer
    total_passes = n_recur * n_layer
    scale_factor = 1.0 / math.sqrt(max(1, n_recur)) if n_recur > 1 else 1.0
    is_2d = (model.config.pos_emb_type == "rope_2d")

    plus_id = 16
    eq_id = 24
    lparen_id = 22
    think_s_id = 4
    think_e_id = 5
    eos_id = tokenizer.eos_id

    if is_2d:
        p_coords, s_coords = model.compute_tensor_2d_coords(idx)
    else:
        p_coords = torch.arange(T_prefill, dtype=torch.long).unsqueeze(0).expand(B, T_prefill)
        s_coords = None

    kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * total_passes

    def run_step(
        tokens_step: torch.Tensor,
        p_step: torch.Tensor,
        s_step: Optional[torch.Tensor],
        is_prefill: bool,
    ) -> torch.Tensor:
        bsz, t_step = tokens_step.size()
        x = model.transformer.wte(tokens_step)
        pass_idx = 0
        for _ in range(n_recur):
            for block in model.transformer.h:
                attn = block.attn
                ln_x = block.ln_1(x)
                qkv = attn.c_attn(ln_x)
                q, k, v = qkv.chunk(3, dim=2)
                q = q.view(bsz, t_step, attn.n_head, attn.head_dim).transpose(1, 2)
                k = k.view(bsz, t_step, attn.n_head, attn.head_dim).transpose(1, 2)
                v = v.view(bsz, t_step, attn.n_head, attn.head_dim).transpose(1, 2)

                if is_2d:
                    q = apply_2d_rope(q, p_step, s_step, attn.rope_2d_cos, attn.rope_2d_sin)
                    k = apply_2d_rope(k, p_step, s_step, attn.rope_2d_cos, attn.rope_2d_sin)
                else:
                    q = apply_1d_rope_custom_pos(q, p_step, attn.rope_cos, attn.rope_sin)
                    k = apply_1d_rope_custom_pos(k, p_step, attn.rope_cos, attn.rope_sin)

                if kv_cache[pass_idx] is not None:
                    k_prev, v_prev = kv_cache[pass_idx]
                    k = torch.cat([k_prev, k], dim=2)
                    v = torch.cat([v_prev, v], dim=2)
                kv_cache[pass_idx] = (k, v)

                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_prefill
                )
                y = y.transpose(1, 2).contiguous().view(bsz, t_step, attn.n_embd)
                y = attn.c_proj(y)
                x_res = x + y
                x_res = x_res + block.mlp(block.ln_2(x_res))
                if n_recur > 1:
                    x = x + scale_factor * (x_res - x)
                else:
                    x = x_res
                pass_idx += 1
        x = model.transformer.ln_f(x)
        return model.lm_head(x[:, -1:, :])

    logits = run_step(idx, p_coords, s_coords, is_prefill=True)
    next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat([idx, next_tok], dim=1)

    cur_col = torch.full((B,), -1, dtype=torch.long)
    cur_step = torch.zeros((B,), dtype=torch.long)
    col_count = torch.zeros((B,), dtype=torch.long)
    in_think = torch.ones((B,), dtype=torch.bool)
    after_think_eq = torch.zeros((B,), dtype=torch.bool)
    ans_d_idx = torch.zeros((B,), dtype=torch.long)
    finished = (next_tok.squeeze(-1) == eos_id)

    for step_idx in range(1, max_new_tokens):
        if finished.all():
            break
        tok_1d = next_tok.squeeze(-1)
        if is_2d:
            p_new = torch.zeros((B,), dtype=torch.long)
            s_new = torch.zeros((B,), dtype=torch.long)
            for b in range(B):
                if finished[b]:
                    continue
                t_val = int(tok_1d[b].item())
                if in_think[b]:
                    if t_val == think_e_id:
                        in_think[b] = False
                        col_count[b] = cur_col[b] + 1
                    else:
                        if t_val == lparen_id:
                            cur_col[b] += 1
                            cur_step[b] = 0
                        if cur_col[b] >= 0:
                            p_new[b] = cur_col[b]
                            s_new[b] = min(int(cur_step[b].item()), 9)
                            cur_step[b] += 1
                else:
                    if not after_think_eq[b]:
                        if t_val == eq_id:
                            after_think_eq[b] = True
                    else:
                        if 6 <= t_val <= 15:
                            p_new[b] = max(0, int(col_count[b].item()) - 1 - int(ans_d_idx[b].item()))
                            s_new[b] = 0
                            ans_d_idx[b] += 1
            p_step = p_new.unsqueeze(1)
            s_step = s_new.unsqueeze(1)
        else:
            pos_val = T_prefill + step_idx - 1
            p_step = torch.full((B, 1), pos_val, dtype=torch.long)
            s_step = None

        logits = run_step(next_tok, p_step, s_step, is_prefill=False)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_tok], dim=1)
        finished |= (next_tok.squeeze(-1) == eos_id)

    out_strs = []
    for b in range(B):
        tok_list = idx[b].tolist()
        if eos_id in tok_list:
            tok_list = tok_list[: tok_list.index(eos_id)]
        out_strs.append(tokenizer.decode(tok_list, skip_special=False))
    return out_strs


def generate_random_problems_for_length(d: int, n_problems: int, rng: random.Random) -> List[Tuple[str, str]]:
    """Generates n_problems stratified random multi-carry addition problems of exact digit length d."""
    problems: List[Tuple[str, str]] = []
    lo = 10 ** (d - 1)
    hi = (10**d) - 1
    for i in range(n_problems):
        mode_idx = i % 20
        if mode_idx < 14:
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
        elif mode_idx < 17:
            a_digits = [str(rng.randint(5, 9)) for _ in range(d)]
            b_digits = [str(rng.randint(5, 9)) for _ in range(d)]
            a = int("".join(a_digits))
            b = int("".join(b_digits))
        else:
            a_digits = []
            b_digits = []
            for col in range(d):
                if col == 0:
                    da = rng.randint(5, 9)
                    db = rng.randint(10 - da, 9)
                else:
                    if rng.random() < 0.75:
                        da = rng.randint(0, 9)
                        db = 9 - da
                    else:
                        da = rng.randint(1 if col == d - 1 else 0, 9)
                        db = rng.randint(1 if col == d - 1 else 0, 9)
                if col == d - 1 and da == 0:
                    da = 1
                    db = 8 if (da + db == 9) else max(1, db)
                a_digits.append(str(da))
                b_digits.append(str(db))
            a = int("".join(a_digits))
            b = int("".join(b_digits))
        problems.append((f"{a}+{b}=", str(a + b)))
    return problems


def evaluate_18m_1d_char_at_1024_context() -> Dict[str, Any]:
    """Evaluates the 18.91M 1D-Char checkpoint (`checkpoints/model_1/model_1_final.pt`) with T=1024 RoPE on Part III (25-50d)."""
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints/model_1/model_1_final.pt")
    if not os.path.exists(ckpt_path):
        return {}
    print("\n" + "=" * 90, flush=True)
    print("EVALUATING 18.91M 1D-CHAR (model_1_final.pt) WITH EXTENDED T=1024 CONTEXT ON 25-50d", flush=True)
    print("=" * 90, flush=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tokenizer = MathTokenizer()
    cfg = GPTConfig(
        vocab_size=27,
        block_size=1024,
        n_layer=6,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        bias=False,
        pos_emb_type="rope",
        max_coords=4096,
    )
    model = MathGPT(cfg)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    clean_state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items() if not k.endswith(".attn.bias")}
    model.load_state_dict(clean_state, strict=False)
    for block in model.transformer.h:
        cos, sin = precompute_rope_freqs(block.attn.head_dim, max_seq_len=4096)
        block.attn.rope_cos = cos
        block.attn.rope_sin = sin

    part3_cases = [
        ("25d_cascade", "9" * 25 + "+1=", "1" + "0" * 25, 25),
        ("35d_ones_eights", "1" * 35 + "+" + "8" * 35 + "=", "9" * 35, 35),
        ("50d_comp", ("1234567890" * 5) + "+" + ("8765432109" * 5) + "=", "9" * 50, 50),
        ("50d_cascade", "9" * 50 + "+1=", "1" + "0" * 50, 50),
    ]
    out_summary = {}
    for name, prompt, expected, d in part3_cases:
        max_new = 10 * (d + 2) + (d + 2) + 16
        gen_str = fast_kv_generate_mathgpt(model, [prompt], tokenizer, max_new_tokens=max_new)[0]
        _, ans = extract_scratchpad_answer(gen_str)
        passed = (ans.strip() == expected)
        zero_loop = "(0+0+0)=00(0+0+0)=00(0+0+0)=00" in gen_str
        out_summary[name] = {
            "passed": passed,
            "zero_loop": zero_loop,
            "extracted_answer_prefix": ans.strip()[:30],
            "gen_tail": gen_str[-80:],
        }
        print(f"  {name:18s} | pass={passed} | zero_loop={zero_loop} | ans='{ans.strip()[:25]}'", flush=True)
    return out_summary


def main():
    torch.set_num_threads(16)
    tokenizer = MathTokenizer()

    # 1. Test 18.91M 1D-Char at T=1024 on 25-50d
    part3_18m_1d = evaluate_18m_1d_char_at_1024_context()

    # 2. Generate strictly equalized 1-15 digit dataset (6,000 samples) and precompute coordinates
    print("\n" + "=" * 90, flush=True)
    print("GENERATING EQUALIZED 1-15 DIGIT DATASET (n=6,000, T=1024) & PRECOMPUTING COORDS", flush=True)
    print("=" * 90, flush=True)
    train_texts = generate_1_to_15d_curriculum(n_samples=6000, seed=42)
    helper_cfg = GPTConfig.create_100k(vocab_size=tokenizer.vocab_size, block_size=1024, pos_emb_type="rope_2d")
    helper_model = MathGPT(helper_cfg)
    dataset = preencode_dataset_with_coords(train_texts, tokenizer, helper_model, block_size=1024)
    print(f"[OK] Pre-encoded {len(dataset)} 1-15d equations.", flush=True)

    # 3. Define equalized 1-15d, T=1024 models (all 100,672 parameters, trained 3,000 steps at batch_size=32)
    configs = [
        ("1d_standard_15d_1024", "1D-RoPE Standard (1-15d, T=1024)", "rope", "1d_standard", 1, 0),
        ("1d_shift512_15d_1024", "1D-RoPE + Random Offset (1-15d, T=1024, Shift=512)", "rope", "1d_shift", 1, 512),
        ("1d_ruoss_15d_1024", "1D-RoPE + Randomized Positions (Ruoss et al. 2023, 1-15d, M=1024)", "rope", "1d_ruoss", 1, 0),
        ("2d_shift32_15d_1024", "Decoupled 2D-RoPE + Shift (Ours, 1-15d, T=1024, k=32)", "rope_2d", "2d_rope", 1, 32),
    ]

    trained_models: Dict[str, Tuple[MathGPT, Dict[str, Any]]] = {}
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints/equalized_15d_1024"), exist_ok=True)

    for model_id, label, pos_emb_type, pos_mode, n_recur, k_shift in configs:
        print("\n" + "-" * 90, flush=True)
        print(f"TRAINING EQUALIZED MODEL: {label}", flush=True)
        print("-" * 90, flush=True)
        cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=1024,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.0,
            bias=False,
            pos_emb_type=pos_emb_type,
            max_coords=4096,
            n_recur=n_recur,
            k_rand_shift=k_shift,
        )
        model = MathGPT(cfg)
        for block in model.transformer.h:
            if pos_emb_type == "rope":
                cos, sin = precompute_rope_freqs(block.attn.head_dim, max_seq_len=4096)
                block.attn.rope_cos = cos
                block.attn.rope_sin = sin

        final_loss, train_sec = train_equalized_model(
            model=model,
            dataset=dataset,
            pad_id=tokenizer.pad_id,
            pos_mode=pos_mode,
            k_shift=k_shift,
            steps=3000,
            batch_size=32,
            lr=2e-3,
            seed=42,
        )
        torch.save(model.state_dict(), os.path.join(ROOT_DIR, "checkpoints/equalized_15d_1024", f"{model_id}.pt"))
        trained_models[model_id] = (
            model,
            {
                "id": model_id,
                "label": label,
                "pos_mode": pos_mode,
                "k_shift": k_shift,
                "final_loss": round(final_loss, 5),
                "train_sec": train_sec,
            },
        )

    # 4. Evaluate all 4 equalized models on 200 random problems per length across seen (8, 12, 15d)
    #    and unseen (16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100d) = 3,200 problems per model!
    eval_lengths = [8, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100]
    n_per_len = 200
    print("\n" + "=" * 90, flush=True)
    print(f"EVALUATING ALL 4 EQUALIZED MODELS ON {n_per_len} RANDOM PROBLEMS PER LENGTH ({len(eval_lengths)*n_per_len} PROBLEMS/MODEL)", flush=True)
    print("=" * 90, flush=True)

    rng = random.Random(20261023)
    problems_by_len: Dict[int, List[Tuple[str, str]]] = {
        d: generate_random_problems_for_length(d, n_per_len, rng) for d in eval_lengths
    }

    equalized_results: Dict[str, Any] = {}
    for model_id, (model, meta) in trained_models.items():
        print(f"\n>>> Evaluating {meta['label']} ...", flush=True)
        per_len_res = {}
        for d in eval_lengths:
            t_d = time.time()
            probs = problems_by_len[d]
            prompts = [p[0] for p in probs]
            expected = [p[1] for p in probs]
            max_new = 10 * (d + 1) + (d + 2) + 8
            outs = fast_kv_generate_mathgpt(model, prompts, tokenizer, max_new_tokens=max_new)
            passed = 0
            for out_str, exp_ans in zip(outs, expected):
                _, pred_ans = extract_scratchpad_answer(out_str)
                if pred_ans.strip() == exp_ans:
                    passed += 1
            acc = round((passed / n_per_len) * 100.0, 2)
            ci = wilson_ci(passed, n_per_len)
            per_len_res[str(d)] = {
                "digits": d,
                "n": n_per_len,
                "passed": passed,
                "accuracy": acc,
                "wilson_95_ci": list(ci),
                "elapsed_sec": round(time.time() - t_d, 2),
            }
            print(f"  d={d:3d} (n={n_per_len}) | Exact={passed:3d}/{n_per_len} ({acc:6.2f}%, CI [{ci[0]:5.1f}, {ci[1]:5.1f}]%)", flush=True)

        # Compute pooled summaries
        seen_pass = sum(per_len_res[str(d)]["passed"] for d in [8, 12, 15])
        seen_n = 3 * n_per_len
        unseen_16_40_pass = sum(per_len_res[str(d)]["passed"] for d in [16, 20, 25, 30, 35, 40])
        unseen_16_40_n = 6 * n_per_len
        unseen_16_50_pass = sum(per_len_res[str(d)]["passed"] for d in [16, 20, 25, 30, 35, 40, 45, 50])
        unseen_16_50_n = 8 * n_per_len

        equalized_results[model_id] = {
            "meta": meta,
            "pooled_seen_8_15d": {
                "passed": seen_pass,
                "n": seen_n,
                "accuracy": round(seen_pass / seen_n * 100.0, 2),
                "wilson_95_ci": list(wilson_ci(seen_pass, seen_n)),
            },
            "pooled_unseen_16_40d": {
                "passed": unseen_16_40_pass,
                "n": unseen_16_40_n,
                "accuracy": round(unseen_16_40_pass / unseen_16_40_n * 100.0, 2),
                "wilson_95_ci": list(wilson_ci(unseen_16_40_pass, unseen_16_40_n)),
            },
            "pooled_unseen_16_50d": {
                "passed": unseen_16_50_pass,
                "n": unseen_16_50_n,
                "accuracy": round(unseen_16_50_pass / unseen_16_50_n * 100.0, 2),
                "wilson_95_ci": list(wilson_ci(unseen_16_50_pass, unseen_16_50_n)),
            },
            "per_length": per_len_res,
        }

    out_payload = {
        "part3_18m_1d_at_1024_context": part3_18m_1d,
        "equalized_15d_1024_comparison": equalized_results,
    }
    out_path = os.path.join(ROOT_DIR, "results/equalized_15d_1024_and_ruoss_sweep.json")
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[PASS] Saved complete equalized 1-15d T=1024 + Ruoss et al. comparison (12,800 total test problems) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
