"""
Training and Evaluation runner for ~100k parameter Dense MathGPT.
Trains on column scratchpad arithmetic from shared math pairs and
evaluates across the 1-20 digit length extrapolation benchmark suite.
"""

import os
import sys
import time
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure workspace root and dense_model package are on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dense_model.model import MathGPT, GPTConfig
from src.tokenizer import MathTokenizer
from src.arithmetic import format_addition_char_scratchpad, extract_scratchpad_answer
from src.dataset import get_or_generate_shared_math_pairs


EXTRAPOLATION_BENCHMARK: Dict[str, List[Tuple[str, str]]] = {
    "In-Distribution (1-12 Digits)": [
        ("15+25=", "40"),
        ("1234+5678=", "6912"),
        ("12345678+87654321=", "99999999"),
        ("10000000+10000000=", "20000000"),
        ("123456789012+876543210987=", "999999999999"),
        ("500000000000+500000000000=", "1000000000000"),
    ],
    "Moderate Extrapolation (14 & 16 Digits)": [
        ("12345678901234+87654321098765=", "99999999999999"),
        ("50000000000000+50000000000000=", "100000000000000"),
        ("1111222233334444+8888777766665555=", "9999999999999999"),
    ],
    "Extreme Extrapolation (18 & 20 Digits)": [
        ("100000000000000000+100000000000000000=", "200000000000000000"),
        ("12345678901234567890+87654321098765432109=", "99999999999999999999"),
        ("55555555555555555555+44444444444444444445=", "100000000000000000000"),
    ],
    "20-Digit Zero Counting & Cascade Carry": [
        ("10000000000000000000+10000000000000000000=", "20000000000000000000"),
        ("99999999999999999999+1=", "100000000000000000000"),
    ],
}


def load_training_equations(
    n_samples: int,
    root_dir: str = ROOT_DIR,
    seed: int = 42,
    curriculum: bool = False,
) -> List[str]:
    """Loads n_samples equations from shared invariant pairs (or balanced curriculum) and formats as scratchpads."""
    if curriculum:
        rng = random.Random(seed)
        texts: List[str] = []
        for _ in range(n_samples):
            p_case = rng.random()
            # 10% adversarial cascading carries: 99..99 + 1 or 99..99 + 99..99
            if p_case < 0.10:
                d = rng.randint(2, 12)
                if rng.random() < 0.5:
                    a_val = 10**d - 1
                    b_val = 1
                else:
                    a_val = 10**d - 1
                    b_val = rng.randint(1, 10**d - 1)
            # 10% adversarial repeated zero chains: 100..00 + 100..00 or 500..00 + 500..00
            elif p_case < 0.20:
                d = rng.randint(2, 12)
                digit_lead_a = rng.choice([1, 2, 4, 5, 8])
                digit_lead_b = rng.choice([1, 2, 4, 5, 8])
                a_val = digit_lead_a * (10 ** (d - 1))
                b_val = digit_lead_b * (10 ** (d - 1))
            # 80% standard balanced curriculum
            else:
                tier = rng.random()
                if tier < 0.25:
                    la, lb = rng.randint(1, 2), rng.randint(1, 2)
                elif tier < 0.60:
                    la, lb = rng.randint(2, 4), rng.randint(2, 4)
                elif tier < 0.85:
                    la, lb = rng.randint(4, 8), rng.randint(4, 8)
                else:
                    la, lb = rng.randint(8, 12), rng.randint(8, 12)

                min_a = 0 if la == 1 else 10 ** (la - 1)
                max_a = 10 ** la - 1
                min_b = 0 if lb == 1 else 10 ** (lb - 1)
                max_b = 10 ** lb - 1
                a_val = rng.randint(min_a, max_a)
                b_val = rng.randint(min_b, max_b)

            texts.append(format_addition_char_scratchpad(a_val, b_val))
        return texts

    shared_path = os.path.join(root_dir, "data", "shared_math_pairs.pt")
    if os.path.exists(shared_path):
        data = torch.load(shared_path, map_location="cpu")
        a_tensor, b_tensor = data["a"], data["b"]
    else:
        a_tensor, b_tensor = get_or_generate_shared_math_pairs(root_dir)

    total_avail = len(a_tensor)
    take_n = min(n_samples, total_avail)

    texts = []
    for i in range(take_n):
        a_val = int(a_tensor[i].item() if isinstance(a_tensor[i], torch.Tensor) else a_tensor[i])
        b_val = int(b_tensor[i].item() if isinstance(b_tensor[i], torch.Tensor) else b_tensor[i])
        texts.append(format_addition_char_scratchpad(a_val, b_val))

    return texts


def encode_batch(
    texts: List[str],
    tokenizer: MathTokenizer,
    block_size: int = 512,
    device: Optional[torch.device] = None,
    mask_prompt: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes texts into input and target tensors with prompt loss masking."""
    B = len(texts)
    pad_id = tokenizer.pad_id
    eq_tokens = tokenizer.encode("=", add_bos=False, add_eos=False)
    eq_token_id = eq_tokens[0] if eq_tokens else None

    x_tensor = torch.full((B, block_size), pad_id, dtype=torch.long)
    y_tensor = torch.full((B, block_size), -100, dtype=torch.long)

    for i, txt in enumerate(texts):
        tokens = tokenizer.encode(txt, add_bos=True, add_eos=True)
        if len(tokens) > block_size + 1:
            tokens = tokens[: block_size + 1]

        n = len(tokens) - 1
        t_tok = torch.tensor(tokens, dtype=torch.long)
        x_tensor[i, :n] = t_tok[:-1]
        y_tensor[i, :n] = t_tok[1:]

        if mask_prompt and eq_token_id is not None:
            eq_matches = (t_tok[:-1] == eq_token_id).nonzero(as_tuple=True)[0]
            if len(eq_matches) > 0:
                eq_idx = eq_matches[0].item()
                y_tensor[i, :eq_idx + 1] = -100

    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    return x_tensor, y_tensor


def train_dense_100k(
    model: MathGPT,
    tokenizer: MathTokenizer,
    train_texts: List[str],
    device: torch.device,
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    block_size: int = 512,
    log_interval: int = 50,
) -> List[float]:
    """Trains the 100k Dense model with AdamW and Cosine Annealing."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=1e-5)

    n_samples = len(train_texts)
    effective_bs = min(batch_size, n_samples)
    loss_history: List[float] = []

    indices = list(range(n_samples))
    curr_idx = 0
    epoch_seed = 42

    t0 = time.time()
    total_tokens_processed = 0
    for step in range(1, steps + 1):
        if curr_idx + effective_bs > n_samples:
            epoch_seed += 1
            rng = random.Random(epoch_seed)
            rng.shuffle(indices)
            curr_idx = 0

        batch_indices = indices[curr_idx : curr_idx + effective_bs]
        curr_idx += effective_bs
        batch_texts = [train_texts[i] for i in batch_indices]

        # Dynamic batch length: pad only to the longest sequence in the batch (capped by block_size)
        raw_lengths = [len(tokenizer.encode(t, add_bos=True, add_eos=True)) for t in batch_texts]
        batch_len = min(max(raw_lengths) - 1, block_size)

        x, y = encode_batch(
            texts=batch_texts,
            tokenizer=tokenizer,
            block_size=batch_len,
            device=device,
            mask_prompt=True,
        )

        # Count tokens trained (unmasked target tokens where y != -100)
        num_target_tokens = int((y != -100).sum().item())
        total_tokens_processed += num_target_tokens

        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if step % log_interval == 0 or step == steps:
            elapsed = time.time() - t0
            lr_curr = optimizer.param_groups[0]["lr"]
            tok_per_sec = total_tokens_processed / max(elapsed, 1e-4)
            print(
                f"[INFO] Step {step:5d}/{steps:5d} | "
                f"Loss: {loss_val:.4f} | "
                f"Tokens: {total_tokens_processed:,} ({tok_per_sec:.0f} tok/s) | "
                f"LR: {lr_curr:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    return loss_history


def evaluate_benchmarks(
    model: MathGPT,
    tokenizer: MathTokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Evaluates the model across the 1-20 digit length extrapolation benchmark suite."""
    model.eval()
    summary: Dict[str, Any] = {}
    total_cases = 0
    total_passed = 0

    print("\n" + "=" * 70)
    print("100K DENSE MODEL EXTRAPOLATION BENCHMARK EVALUATION")
    print("=" * 70)

    for tier_name, cases in EXTRAPOLATION_BENCHMARK.items():
        print(f"\n--- {tier_name} ({len(cases)} cases) ---")
        tier_passed = 0
        tier_results = []

        for prompt, expected in cases:
            eq_prompt = prompt if prompt.endswith("<THINK>") else (prompt + "<THINK>")
            tokens = tokenizer.encode(eq_prompt, add_bos=True, add_eos=False)
            x = torch.tensor([tokens], dtype=torch.long, device=device)

            t_start = time.time()
            with torch.no_grad():
                out = model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    eos_id=tokenizer.eos_id,
                )
            elapsed = time.time() - t_start

            gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
            reasoning, answer = extract_scratchpad_answer(gen_text)
            is_pass = (answer.strip() == expected.strip())

            if is_pass:
                tier_passed += 1
                total_passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"

            total_cases += 1
            tier_results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": answer,
                "reasoning_sample": reasoning[:60] if reasoning else "",
                "is_correct": is_pass,
                "latency_sec": round(elapsed, 3),
            })

            ans_preview = answer[:25] + ("..." if len(answer) > 25 else "")
            print(f"  {prompt:<35} Expected: {expected:<22} Pred: {ans_preview:<22} {status} ({elapsed:.2f}s)")

        tier_acc = (tier_passed / len(cases)) * 100.0 if cases else 0.0
        summary[tier_name] = {
            "passed": tier_passed,
            "total": len(cases),
            "accuracy_pct": round(tier_acc, 2),
            "cases": tier_results,
        }
        print(f"  --> Tier Accuracy: {tier_acc:.1f}% ({tier_passed}/{len(cases)})")

    overall_acc = (total_passed / total_cases) * 100.0 if total_cases > 0 else 0.0
    summary["overall"] = {
        "passed": total_passed,
        "total": total_cases,
        "accuracy_pct": round(overall_acc, 2),
    }

    print("\n" + "=" * 70)
    print(f"OVERALL BENCHMARK ACCURACY: {overall_acc:.2f}% ({total_passed}/{total_cases})")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ~100k Dense MathGPT")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of training equation pairs to sample")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--block-size", type=int, default=256, help="Maximum sequence context length")
    parser.add_argument("--save-ckpt", type=str, default="", help="Checkpoint save path (defaults to dense_model/checkpoints/dense_100k[_2d].pt)")
    parser.add_argument("--pos-emb-type", type=str, default="rope", choices=["rope", "rope_2d", "absolute"], help="Position embedding type")
    parser.add_argument("--curriculum", action="store_true", help="Use balanced digit curriculum (1-12 digits)")
    parser.add_argument("--threads", type=int, default=12, help="Number of CPU threads to use")
    parser.add_argument("--log-interval", type=int, default=50, help="Log interval in steps")
    parser.add_argument("--n-recur", type=int, default=1, help="Number of recurrent unrolling passes across blocks")
    parser.add_argument("--k-rand-shift", type=int, default=0, help="Place shift data augmentation magnitude")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only using checkpoint")
    args = parser.parse_args()

    if not args.save_ckpt:
        suffix = "_2d" if args.pos_emb_type == "rope_2d" else ""
        if args.n_recur > 1:
            suffix += f"_rec{args.n_recur}"
        args.save_ckpt = f"dense_model/checkpoints/dense_100k{suffix}.pt"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.threads)

    tokenizer = MathTokenizer()
    config = GPTConfig.create_100k(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        pos_emb_type=args.pos_emb_type,
        n_recur=args.n_recur,
        k_rand_shift=args.k_rand_shift,
    )
    model = MathGPT(config)

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"[INFO] Initialized MathGPT (~100k Dense):")
    print(f"       Total Parameters:          {total_params:,}")
    print(f"       Non-Embedding Parameters:  {non_emb_params:,}")
    print(f"       Layers: {config.n_layer}, Heads: {config.n_head}, Dim: {config.n_embd}, RoPE: {config.pos_emb_type}")
    print(f"[INFO] Active Compute Device: {device} (CPU threads={torch.get_num_threads() if device.type == 'cpu' else 'N/A'})")
    model.to(device)

    if not args.eval_only:
        print(f"\n[INFO] Loading {args.n_samples} training equations (curriculum={args.curriculum})...")
        train_texts = load_training_equations(args.n_samples, root_dir=ROOT_DIR, curriculum=args.curriculum)
        print(f"[INFO] Loaded {len(train_texts)} equations. Sample text:\n       {train_texts[0][:80]}...")

        print(f"\n[INFO] Starting training for {args.steps} steps (batch_size={args.batch_size}, lr={args.lr})...")
        t_start = time.time()
        loss_hist = train_dense_100k(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            block_size=args.block_size,
            log_interval=args.log_interval,
        )
        t_elapsed = time.time() - t_start
        print(f"[INFO] Training completed in {t_elapsed:.1f}s. Final loss: {loss_hist[-1]:.4f}")

        if args.save_ckpt:
            os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, args.save_ckpt)), exist_ok=True)
            ckpt_path = os.path.join(ROOT_DIR, args.save_ckpt)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "final_loss": loss_hist[-1],
                "steps": args.steps,
            }, ckpt_path)
            print(f"[INFO] Saved model checkpoint to {ckpt_path}")
    else:
        ckpt_path = os.path.join(ROOT_DIR, args.save_ckpt)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[INFO] Loaded checkpoint from {ckpt_path}")
        else:
            print(f"[WARN] Checkpoint {ckpt_path} not found. Running with initialized weights.")

    summary = evaluate_benchmarks(model, tokenizer, device=device)
    return summary


if __name__ == "__main__":
    main()
