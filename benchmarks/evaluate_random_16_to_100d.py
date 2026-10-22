"""
Expanded Random & Multi-Carry Arithmetic Extrapolation Benchmark (16 to 100 Digits).
Evaluates CBT-18.95M (`champion_final.pt`) using KV-cached batched greedy decoding
across unseen digit lengths from 16 to 100 digits, logging per-length accuracy,
internal scratchpad accuracy, 95% Wilson confidence intervals, and failure modes.
"""

import json
import math
import random
import time
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F

from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig, apply_2d_rope
from champion_model.scaled_t4.tokenizer import ScaledTokenizer


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


@torch.no_grad()
def fast_generate_batch(
    model: ScaledGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
) -> torch.Tensor:
    """KV-cached batched greedy decoding exact to ScaledGPT.generate(temperature=0.0)."""
    B, _ = idx.size()
    n_recur = max(1, model.config.n_recur)
    scale_factor = 1.0 / math.sqrt(n_recur) if n_recur > 1 else 1.0
    n_layer = model.config.n_layer
    total_layers = n_recur * n_layer

    k_cache: List[Any] = [None] * total_layers
    v_cache: List[Any] = [None] * total_layers

    p_coords, s_coords = model.compute_2d_coords(idx)
    x = model.transformer.wte(idx)
    layer_idx = 0
    for _ in range(n_recur):
        for block in model.transformer.h:
            x_ln = block.ln_1(x)
            attn = block.attn
            qkv = attn.c_attn(x_ln)
            q, k, v = qkv.chunk(3, dim=2)
            q = q.view(B, -1, attn.n_head, attn.head_dim).transpose(1, 2)
            k = k.view(B, -1, attn.n_head, attn.head_dim).transpose(1, 2)
            v = v.view(B, -1, attn.n_head, attn.head_dim).transpose(1, 2)
            q = apply_2d_rope(q, p_coords, s_coords, attn.rope_cos, attn.rope_sin)
            k = apply_2d_rope(k, p_coords, s_coords, attn.rope_cos, attn.rope_sin)
            k_cache[layer_idx] = k
            v_cache[layer_idx] = v
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, -1, attn.n_embd)
            x_attn = x + attn.c_proj(y)
            x_res = x_attn + block.mlp(block.ln_2(x_attn))
            x = x + scale_factor * (x_res - x) if n_recur > 1 else x_res
            layer_idx += 1

    logits = model.lm_head(model.transformer.ln_f(x[:, -1:, :]))
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat([idx, next_token], dim=1)

    tok = model.tokenizer
    lparen_id = tok.lparen_id
    think_e_id = tok.think_e_id
    eq_id = tok.eq_id
    digit_ids = set(tok.digit_ids)

    cur_col = torch.full((B,), -1, dtype=torch.long)
    cur_step = torch.zeros((B,), dtype=torch.long)
    in_think = torch.ones((B,), dtype=torch.bool)
    after_think_eq = torch.zeros((B,), dtype=torch.bool)
    col_count = torch.zeros((B,), dtype=torch.long)
    ans_d_idx = torch.zeros((B,), dtype=torch.long)
    finished = (next_token.squeeze(-1) == eos_id)

    for _ in range(max_new_tokens - 1):
        if finished.all():
            break
        tok_vec = idx[:, -1]
        p_step = torch.zeros((B, 1), dtype=torch.long)
        s_step = torch.zeros((B, 1), dtype=torch.long)
        for b in range(B):
            t_id = tok_vec[b].item()
            if in_think[b]:
                if t_id == think_e_id:
                    in_think[b] = False
                    col_count[b] = cur_col[b] + 1
                else:
                    if t_id == lparen_id:
                        cur_col[b] += 1
                        cur_step[b] = 0
                    if cur_col[b] >= 0:
                        p_step[b, 0] = cur_col[b]
                        s_step[b, 0] = min(int(cur_step[b].item()), 9)
                        cur_step[b] += 1
            else:
                if not after_think_eq[b]:
                    if t_id == eq_id:
                        after_think_eq[b] = True
                else:
                    if t_id in digit_ids:
                        p_step[b, 0] = max(0, int(col_count[b].item()) - 1 - int(ans_d_idx[b].item()))
                        s_step[b, 0] = 0
                        ans_d_idx[b] += 1

        x = model.transformer.wte(idx[:, -1:])
        layer_idx = 0
        for _ in range(n_recur):
            for block in model.transformer.h:
                x_ln = block.ln_1(x)
                attn = block.attn
                qkv = attn.c_attn(x_ln)
                q, k, v = qkv.chunk(3, dim=2)
                q = q.view(B, 1, attn.n_head, attn.head_dim).transpose(1, 2)
                k = k.view(B, 1, attn.n_head, attn.head_dim).transpose(1, 2)
                v = v.view(B, 1, attn.n_head, attn.head_dim).transpose(1, 2)
                q = apply_2d_rope(q, p_step, s_step, attn.rope_cos, attn.rope_sin)
                k = apply_2d_rope(k, p_step, s_step, attn.rope_cos, attn.rope_sin)
                k_cache[layer_idx] = torch.cat([k_cache[layer_idx], k], dim=2)
                v_cache[layer_idx] = torch.cat([v_cache[layer_idx], v], dim=2)
                y = F.scaled_dot_product_attention(q, k_cache[layer_idx], v_cache[layer_idx], is_causal=False)
                y = y.transpose(1, 2).contiguous().view(B, 1, attn.n_embd)
                x_attn = x + attn.c_proj(y)
                x_res = x_attn + block.mlp(block.ln_2(x_attn))
                x = x + scale_factor * (x_res - x) if n_recur > 1 else x_res
                layer_idx += 1

        logits = model.lm_head(model.transformer.ln_f(x))
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
        finished |= (next_token.squeeze(-1) == eos_id)
    return idx


def generate_expected_scratchpad(a: int, b: int) -> str:
    """Computes the exact expected scratchpad string inside <THINK>...</THINK>."""
    sa, sb = str(a)[::-1], str(b)[::-1]
    max_len = max(len(sa), len(sb))
    carry = 0
    steps = []
    for i in range(max_len):
        da = int(sa[i]) if i < len(sa) else 0
        db = int(sb[i]) if i < len(sb) else 0
        total = carry + da + db
        c_out = total // 10
        s_out = total % 10
        steps.append(f"({carry}+{da}+{db})={c_out}{s_out}")
        carry = c_out
    if carry > 0:
        steps.append(f"({carry}+0+0)=0{carry}")
    return "".join(steps)


def sample_problems_for_length(d: int, n_samples: int, rng: random.Random) -> List[Tuple[int, int, str]]:
    """
    Generates n_samples diverse d-digit addition problems with varied carry patterns:
    - 70% uniform random d-digit + d-digit (random carry chains)
    - 15% dense carry-heavy d-digit sums (digits biased toward 5-9 so carries cascade across multiple columns)
    - 15% alternating / sparse carry chains (mix of 9-runs and 0-runs)
    """
    problems = []
    for i in range(n_samples):
        r = i / n_samples
        if r < 0.70:
            a = rng.randint(10 ** (d - 1), 10**d - 1)
            b = rng.randint(10 ** (d - 1), 10**d - 1)
            kind = "uniform_random"
        elif r < 0.85:
            a_str = str(rng.randint(5, 9)) + "".join(str(rng.randint(5, 9)) for _ in range(d - 1))
            b_str = str(rng.randint(5, 9)) + "".join(str(rng.randint(5, 9)) for _ in range(d - 1))
            a, b = int(a_str), int(b_str)
            kind = "dense_carry"
        else:
            # Partial cascading carries: e.g., blocks of 9s and random digits
            a_digits = [str(rng.choice([0, 9, rng.randint(1, 8)])) for _ in range(d)]
            if a_digits[0] == "0":
                a_digits[0] = str(rng.randint(1, 9))
            b_digits = [str(9 - int(x)) if rng.random() < 0.6 else str(rng.randint(0, 9)) for x in a_digits]
            if b_digits[0] == "0":
                b_digits[0] = str(rng.randint(1, 9))
            a, b = int("".join(a_digits)), int("".join(b_digits))
            kind = "cascade_mix"
        problems.append((a, b, kind))
    return problems


def main() -> None:
    tok = ScaledTokenizer()
    ckpt = torch.load(
        "champion_model/scaled_t4/checkpoints/champion_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = ScaledGPTConfig(
        vocab_size=tok.vocab_size,
        block_size=2048,
        n_layer=6,
        n_head=8,
        n_embd=512,
        n_recur=3,
        max_coords=4096,
        k_rand_shift=32,
    )
    model = ScaledGPT(config, tok)
    state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    lengths = [16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100]
    samples_per_len = {
        16: 30,
        20: 30,
        25: 30,
        30: 30,
        35: 30,
        40: 30,
        45: 30,
        50: 30,
        55: 20,
        60: 20,
        70: 15,
        80: 10,
        100: 10,
    }

    rng = random.Random(20261021)
    all_results: Dict[str, Any] = {"per_length": {}, "failures": []}
    t_global = time.time()

    print("=" * 95, flush=True)
    print("EXPANDED RANDOM EXTRAPOLATION SWEEP (16 TO 100 DIGITS) ON CBT-18.95M (champion_final.pt)", flush=True)
    print("=" * 95, flush=True)

    for d in lengths:
        n_s = samples_per_len[d]
        probs = sample_problems_for_length(d, n_s, rng)
        prompts_tok = [tok.encode(f"{a}+{b}=<THINK>", add_bos=True, add_eos=False) for a, b, _ in probs]
        x = torch.tensor(prompts_tok, dtype=torch.long)
        max_new = d * 12 + 25

        t0 = time.time()
        out = fast_generate_batch(model, x, max_new_tokens=max_new, eos_id=tok.eos_id)
        dt = time.time() - t0

        exact_ans = 0
        exact_think = 0
        lparen_errors = 0
        off_by_one_errors = 0

        for idx_b, (a, b, kind) in enumerate(probs):
            text = tok.decode(out[idx_b].tolist(), skip_special=False)
            exp_ans = str(a + b)
            exp_think = generate_expected_scratchpad(a, b)

            if "<THINK>" in text and "</THINK>=" in text:
                think_part = text.split("<THINK>", 1)[1].split("</THINK>=", 1)[0]
                ans_part = text.split("</THINK>=", 1)[1].split("<EOS>", 1)[0].strip()
            else:
                think_part = ""
                ans_part = "NO_STOP"

            think_ok = (think_part == exp_think)
            ans_ok = (ans_part == exp_ans)
            if think_ok:
                exact_think += 1
            if ans_ok:
                exact_ans += 1
            else:
                # Diagnose failure mode
                if think_part.count("(") != exp_think.count("(") and ans_part != "NO_STOP":
                    lparen_errors += 1
                if think_ok and abs(len(ans_part) - len(exp_ans)) == 1:
                    off_by_one_errors += 1
                if len(all_results["failures"]) < 40:
                    all_results["failures"].append({
                        "digits": d,
                        "kind": kind,
                        "expected_len": len(exp_ans),
                        "pred_len": len(ans_part),
                        "think_ok": think_ok,
                        "expected_head": exp_ans[:20],
                        "pred_head": ans_part[:20],
                    })

        lo, hi = wilson_ci(exact_ans, n_s)
        think_lo, think_hi = wilson_ci(exact_think, n_s)
        acc = (exact_ans / n_s) * 100.0
        t_acc = (exact_think / n_s) * 100.0
        tok_s = (n_s * out.size(1)) / dt

        all_results["per_length"][str(d)] = {
            "digits": d,
            "n_samples": n_s,
            "exact_match": exact_ans,
            "accuracy_pct": round(acc, 1),
            "wilson_95_ci": [lo, hi],
            "think_exact": exact_think,
            "think_acc_pct": round(t_acc, 1),
            "think_wilson_95_ci": [think_lo, think_hi],
            "lparen_errors": lparen_errors,
            "off_by_one_readout_errors": off_by_one_errors,
            "seq_tokens": int(out.size(1)),
            "elapsed_sec": round(dt, 2),
            "tok_per_sec": round(tok_s, 1),
        }

        print(
            f"d={d:3d} (n={n_s:2d}, T~{out.size(1):4d}) | "
            f"Exact={exact_ans:2d}/{n_s:2d} ({acc:5.1f}%, CI [{lo:4.1f}, {hi:5.1f}]%) | "
            f"Think={exact_think:2d}/{n_s:2d} ({t_acc:5.1f}%) | "
            f"time={dt:5.1f}s ({tok_s:5.1f} tok/s)",
            flush=True,
        )

    # Aggregate 16-50d stats
    d_16_50 = [d for d in lengths if 16 <= d <= 50]
    tot_n_50 = sum(all_results["per_length"][str(d)]["n_samples"] for d in d_16_50)
    tot_k_50 = sum(all_results["per_length"][str(d)]["exact_match"] for d in d_16_50)
    tot_t_50 = sum(all_results["per_length"][str(d)]["think_exact"] for d in d_16_50)
    lo50, hi50 = wilson_ci(tot_k_50, tot_n_50)
    tlo50, thi50 = wilson_ci(tot_t_50, tot_n_50)

    all_results["summary_16_50"] = {
        "total_samples": tot_n_50,
        "exact_match": tot_k_50,
        "accuracy_pct": round((tot_k_50 / tot_n_50) * 100.0, 2),
        "wilson_95_ci": [lo50, hi50],
        "think_exact": tot_t_50,
        "think_acc_pct": round((tot_t_50 / tot_n_50) * 100.0, 2),
        "think_wilson_95_ci": [tlo50, thi50],
    }
    all_results["total_elapsed_sec"] = round(time.time() - t_global, 2)

    out_path = "results/random_extrapolation_16_to_100d.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 95, flush=True)
    print(
        f"POOLED 16-50 DIGITS: {tot_k_50}/{tot_n_50} ({all_results['summary_16_50']['accuracy_pct']}%, "
        f"95% Wilson CI [{lo50}, {hi50}]%) | Think: {tot_t_50}/{tot_n_50} ({all_results['summary_16_50']['think_acc_pct']}%)",
        flush=True,
    )
    print(f"Saved detailed results to {out_path} in {all_results['total_elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
