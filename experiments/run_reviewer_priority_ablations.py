"""
Reviewer Priority Ablations:
1. p-Only Ablation (Decoupled 2D-RoPE with tau micro-rotation disabled).
2. k-Shift Sweep (k=0, k=16, k=32, k=64) to show the cliff shifting with p_max = N_train + k.
3. 1D Baseline Retraining with extended budget / tuned LR to isolate optimization vs extrapolation.
All evaluated on random multi-carry addition problems across lengths 8 to 80 digits.
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
    apply_rope,
)
from src.tokenizer import MathTokenizer
from src.arithmetic import format_addition_char_scratchpad, extract_scratchpad_answer
from experiments.run_equalized_15d_1024_and_large_sweep import (
    wilson_ci,
    apply_1d_rope_custom_pos,
    generate_1_to_15d_curriculum,
    preencode_dataset_with_coords,
    generate_random_problems_for_length,
)


def forward_ablation(
    model: MathGPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
    place_coords: torch.Tensor,
    step_coords: torch.Tensor,
    mode: str,
    k_shift: int = 0,
) -> torch.Tensor:
    """
    Forward pass supporting:
    - '2d_rope': standard decoupled (p, tau) with shift k_shift
    - 'p_only': place-only RoPE (tau forced to 0) with shift k_shift
    - '1d_rope': standard 1D RoPE (0..T-1)
    """
    B, T = idx.size()
    device = idx.device
    x = model.transformer.wte(idx)

    if mode == "1d_rope":
        pos_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)
        p_c = None
        s_c = None
    elif mode == "2d_rope":
        if k_shift > 0:
            shifts = torch.randint(0, k_shift + 1, (B, 1), device=device)
            place_coords = place_coords + shifts * (place_coords > 0).long()
        p_c = place_coords
        s_c = step_coords
        pos_ids = None
    elif mode == "p_only":
        if k_shift > 0:
            shifts = torch.randint(0, k_shift + 1, (B, 1), device=device)
            place_coords = place_coords + shifts * (place_coords > 0).long()
        p_c = place_coords
        s_c = torch.zeros_like(step_coords)
        pos_ids = None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for block in model.transformer.h:
        attn = block.attn
        ln_x = block.ln_1(x)
        qkv = attn.c_attn(ln_x)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        k = k.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        v = v.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)

        if mode == "1d_rope":
            q = apply_1d_rope_custom_pos(q, pos_ids, attn.rope_cos, attn.rope_sin)
            k = apply_1d_rope_custom_pos(k, pos_ids, attn.rope_cos, attn.rope_sin)
        else:
            q = apply_2d_rope(q, p_c, s_c, attn.rope_2d_cos, attn.rope_2d_sin)
            k = apply_2d_rope(k, p_c, s_c, attn.rope_2d_cos, attn.rope_2d_sin)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, attn.n_embd)
        y = attn.c_proj(y)
        x = x + y
        x = x + block.mlp(block.ln_2(x))

    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    return loss


def train_ablation_model(
    model: MathGPT,
    dataset: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
    mode: str,
    k_shift: int,
    steps: int = 3000,
    batch_size: int = 32,
    lr: float = 2e-3,
    seed: int = 42,
) -> Tuple[float, float]:
    """Trains equalized 100k ablation model."""
    torch.manual_seed(seed)
    random.seed(seed)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

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
        loss = forward_ablation(
            model=model,
            idx=x_b,
            targets=y_b,
            place_coords=p_b,
            step_coords=s_b,
            mode=mode,
            k_shift=k_shift,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())

        if step % 500 == 0 or step == steps:
            print(
                f"  [Step {step:5d}/{steps}] loss={last_loss:.5f} | elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    return last_loss, round(time.time() - t0, 2)


@torch.no_grad()
def fast_kv_generate_ablation(
    model: MathGPT,
    prompts: List[str],
    tokenizer: MathTokenizer,
    mode: str,
    max_new_tokens: int,
) -> List[str]:
    """Fast KV-cached batched decoding matching verified reference generation."""
    model.eval()
    B = len(prompts)
    encoded = [tokenizer.encode(p + "<THINK>", add_bos=True, add_eos=False) for p in prompts]
    idx = torch.tensor(encoded, dtype=torch.long)
    T_prefill = idx.size(1)

    n_layer = model.config.n_layer
    is_2d = (mode in ["2d_rope", "p_only"])
    is_p_only = (mode == "p_only")

    eos_id = tokenizer.eos_id

    if is_2d:
        p_coords, s_coords = model.compute_tensor_2d_coords(idx)
        if is_p_only:
            s_coords = torch.zeros_like(s_coords)
    else:
        p_coords = torch.arange(T_prefill, dtype=torch.long).unsqueeze(0).expand(B, T_prefill)
        s_coords = None

    kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layer

    def run_step(
        tokens_step: torch.Tensor,
        p_step: torch.Tensor,
        s_step: Optional[torch.Tensor],
        is_prefill: bool,
    ) -> torch.Tensor:
        bsz, t_step = tokens_step.size()
        x = model.transformer.wte(tokens_step)
        for layer_idx, block in enumerate(model.transformer.h):
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

            if kv_cache[layer_idx] is not None:
                k_prev, v_prev = kv_cache[layer_idx]
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
            kv_cache[layer_idx] = (k, v)

            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_prefill
            )
            y = y.transpose(1, 2).contiguous().view(bsz, t_step, attn.n_embd)
            y = attn.c_proj(y)
            x = x + y
            x = x + block.mlp(block.ln_2(x))
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
                    if t_val == 5:  # </THINK>
                        in_think[b] = False
                        col_count[b] = cur_col[b] + 1
                    else:
                        if t_val == 22:  # (
                            cur_col[b] += 1
                            cur_step[b] = 0
                        if cur_col[b] >= 0:
                            p_new[b] = cur_col[b]
                            s_new[b] = min(int(cur_step[b].item()), 9)
                            cur_step[b] += 1
                else:
                    if not after_think_eq[b]:
                        if t_val == 24:  # =
                            after_think_eq[b] = True
                    else:
                        if 6 <= t_val <= 15:
                            p_new[b] = max(0, int(col_count[b].item()) - 1 - int(ans_d_idx[b].item()))
                            s_new[b] = 0
                            ans_d_idx[b] += 1
            if is_p_only:
                s_new = torch.zeros_like(s_new)
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
            tok_list = tok_list[:tok_list.index(eos_id)]
        out_strs.append(tokenizer.decode(tok_list, skip_special=False))
    return out_strs


def main():
    torch.set_num_threads(16)
    tokenizer = MathTokenizer()
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints/reviewer_ablations"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)

    print("=" * 90)
    print("REVIEWER PRIORITY ABLATIONS: p-ONLY, k-SWEEP (0, 16, 64), & RETRAINED 1D")
    print("=" * 90, flush=True)

    # 1. Dataset Generation
    train_texts = generate_1_to_15d_curriculum(n_samples=6000, seed=42)
    helper_cfg = GPTConfig.create_100k(vocab_size=tokenizer.vocab_size, block_size=1024, pos_emb_type="rope_2d")
    helper_model = MathGPT(helper_cfg)
    dataset = preencode_dataset_with_coords(train_texts, tokenizer, helper_model, block_size=1024)
    print(f"[OK] Pre-encoded {len(dataset)} 1-15d equations.", flush=True)

    # 2. Experiment Configurations
    configs = [
        # (id, label, mode, pos_emb_type, k_shift, steps, lr)
        ("p_only_k32", "p-Only (tau rotation disabled, k=32)", "p_only", "rope_2d", 32, 3000, 2e-3),
        ("2d_k0", "2D-RoPE Shift k=0 (No Shift, p_max=15)", "2d_rope", "rope_2d", 0, 3000, 2e-3),
        ("2d_k16", "2D-RoPE Shift k=16 (p_max=31)", "2d_rope", "rope_2d", 16, 3000, 2e-3),
        ("2d_k64", "2D-RoPE Shift k=64 (p_max=79)", "2d_rope", "rope_2d", 64, 3000, 2e-3),
        ("1d_retrained", "1D-RoPE Retrained (6,000 Steps, LR=8e-4)", "1d_rope", "rope", 0, 6000, 8e-4),
    ]

    trained_models = {}

    for exp_id, label, mode, pos_emb_type, k_shift, steps, lr in configs:
        ckpt_path = os.path.join(ROOT_DIR, f"checkpoints/reviewer_ablations/{exp_id}.pt")
        print("\n" + "-" * 90)
        print(f"EXPERIMENT: {label} ({steps} steps, lr={lr})")
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
            n_recur=1,
            k_rand_shift=k_shift,
        )
        model = MathGPT(cfg)
        for block in model.transformer.h:
            if pos_emb_type == "rope":
                cos, sin = precompute_rope_freqs(block.attn.head_dim, max_seq_len=4096)
                block.attn.rope_cos = cos
                block.attn.rope_sin = sin

        if os.path.exists(ckpt_path):
            print(f"Loading existing checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            final_loss = 0.0
            train_sec = 0.0
        else:
            final_loss, train_sec = train_ablation_model(
                model=model,
                dataset=dataset,
                pad_id=tokenizer.pad_id,
                mode=mode,
                k_shift=k_shift,
                steps=steps,
                batch_size=32,
                lr=lr,
                seed=42,
            )
            torch.save(model.state_dict(), ckpt_path)

        trained_models[exp_id] = (
            model,
            {
                "id": exp_id,
                "label": label,
                "mode": mode,
                "k_shift": k_shift,
                "steps": steps,
                "lr": lr,
                "final_loss": round(final_loss, 5),
                "train_sec": train_sec,
            },
        )

    # 3. Add existing equalized 2D k=32 model for comparison
    k32_ckpt = os.path.join(ROOT_DIR, "checkpoints/equalized_15d_1024/2d_shift32_15d_1024.pt")
    if os.path.exists(k32_ckpt):
        cfg_k32 = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=1024,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.0,
            bias=False,
            pos_emb_type="rope_2d",
            max_coords=4096,
            n_recur=1,
            k_rand_shift=32,
        )
        model_k32 = MathGPT(cfg_k32)
        model_k32.load_state_dict(torch.load(k32_ckpt, map_location="cpu"))
        trained_models["2d_k32"] = (
            model_k32,
            {
                "id": "2d_k32",
                "label": "Decoupled 2D-RoPE Shift k=32 (p_max=47, Baseline)",
                "mode": "2d_rope",
                "k_shift": 32,
                "steps": 3000,
                "lr": 2e-3,
                "final_loss": 0.00597,
                "train_sec": 352.24,
            },
        )

    # 4. Evaluation Suite
    # Test lengths from in-distribution up to 80 digits
    eval_lengths = [8, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80]
    n_per_len = 200
    print("\n" + "=" * 90)
    print(f"EVALUATING ALL ABLATION MODELS ON {n_per_len} PROBLEMS PER LENGTH")
    print("=" * 90, flush=True)

    rng = random.Random(20261024)
    problems_by_len = {
        d: generate_random_problems_for_length(d, n_per_len, rng) for d in eval_lengths
    }

    eval_results = {}
    for exp_id, (model, meta) in trained_models.items():
        mode = meta["mode"]
        print(f"\n>>> Evaluating {meta['label']} ...", flush=True)
        per_len_res = {}
        for d in eval_lengths:
            t_d = time.time()
            probs = problems_by_len[d]
            prompts = [p[0] for p in probs]
            expected = [p[1] for p in probs]
            max_new = 10 * (d + 1) + (d + 2) + 8
            outs = []
            chunk_size = 50
            for c_i in range(0, len(prompts), chunk_size):
                p_chunk = prompts[c_i : c_i + chunk_size]
                outs.extend(fast_kv_generate_ablation(model, p_chunk, tokenizer, mode=mode, max_new_tokens=max_new))
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

        eval_results[exp_id] = {
            "meta": meta,
            "per_length": per_len_res,
        }

    out_path = os.path.join(ROOT_DIR, "results/reviewer_priority_ablations.json")
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[PASS] Saved reviewer priority ablation results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
