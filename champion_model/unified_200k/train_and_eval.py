"""
Unified Training and Benchmark Suite for UnifiedGPT (~200k Parameters).
Trains simultaneously on:
1. Multi-digit column scratchpad arithmetic (1 to 12 digits + edge case curriculum).
2. Natural language tool-calling dialogues across 6 functional categories.

Evaluates on:
1. Arithmetic Length Extrapolation Benchmark (1-20 digits, 14 test cases).
2. Structured Tool Calling Benchmark (50 test cases).
"""

import os
import sys
import time
import random
import argparse
from typing import List, Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.unified_200k.tokenizer import UnifiedTokenizer
from champion_model.unified_200k.model import UnifiedGPT, UnifiedGPTConfig
from champion_model.unified_200k.dataset import generate_joint_dataset, encode_joint_batch
from champion_model.math_model.arithmetic import extract_scratchpad_answer
from champion_model.tool_model.dataset import generate_dataset as generate_tool_eval_dataset


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


def train_unified_gpt(
    model: UnifiedGPT,
    tokenizer: UnifiedTokenizer,
    math_dataset: List[Dict[str, str]],
    tool_dataset: List[Dict[str, str]],
    steps: int = 3500,
    anneal_steps: int = 1500,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    weight_decay: float = 0.01,
    log_interval: int = 250,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Two-Stage Frontier Training Pipeline:
    Phase 1: Broad Joint Pre-training (50% Math / 50% Tools)
    Phase 2: Reasoning & State-Machine Annealing (75% Math / 25% Tools) with LR cooling to 1e-5.
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=1e-5)

    n_math = len(math_dataset)
    n_tools = len(tool_dataset)

    math_idx = 0
    tool_idx = 0
    rng_math = random.Random(42)
    rng_tool = random.Random(1337)

    loss_history: List[float] = []
    t0 = time.time()
    total_tokens_processed = 0

    phase1_steps = max(1, steps - anneal_steps)

    for step in range(1, steps + 1):
        is_phase2 = (step > phase1_steps)
        if is_phase2:
            # Phase 2 Reasoning Annealing: 75% math, 25% tools (24 math, 8 tools for bs=32)
            n_m = int(batch_size * 0.75)
            n_t = batch_size - n_m
        else:
            # Phase 1 Broad Pre-training: 50% math, 50% tools (16 math, 16 tools for bs=32)
            n_m = batch_size // 2
            n_t = batch_size - n_m

        if math_idx + n_m > n_math:
            rng_math.shuffle(math_dataset)
            math_idx = 0
        if tool_idx + n_t > n_tools:
            rng_tool.shuffle(tool_dataset)
            tool_idx = 0

        math_batch = math_dataset[math_idx:math_idx + n_m]
        tool_batch = tool_dataset[tool_idx:tool_idx + n_t]
        math_idx += n_m
        tool_idx += n_t

        x_b, y_b = encode_joint_batch(math_batch, tool_batch, tokenizer, device=device)
        total_tokens_processed += (y_b != -100).sum().item()

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x_b, targets=y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if step % log_interval == 0 or step == steps:
            elapsed = time.time() - t0
            tok_per_sec = total_tokens_processed / max(elapsed, 1e-4)
            lr_curr = optimizer.param_groups[0]["lr"]
            stage_name = "Phase 2 (Reasoning Anneal)" if is_phase2 else "Phase 1 (Joint Pretrain)"
            print(
                f"[INFO] Step {step:5d}/{steps:5d} [{stage_name}] | "
                f"Loss: {loss_val:.4f} | "
                f"Tokens: {total_tokens_processed:,} ({tok_per_sec:.0f} tok/s) | "
                f"LR: {lr_curr:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    return loss_history


def evaluate_math_extrapolation(
    model: UnifiedGPT,
    tokenizer: UnifiedTokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Evaluates multi-digit arithmetic extrapolation up to 20 digits."""
    model.eval()
    summary: Dict[str, Any] = {}
    total_cases = 0
    total_passed = 0

    print("\n" + "=" * 75)
    print("UNIFIED 200K MODEL: ARITHMETIC LENGTH EXTRAPOLATION (1-20 DIGITS)")
    print("=" * 75)

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
                stop_ids = [tokenizer.eos_id]
                out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.0, stop_tokens=stop_ids)
            elapsed = time.time() - t_start

            gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
            reasoning, answer = extract_scratchpad_answer(gen_text)
            is_pass = (answer.strip() == expected.strip())

            if is_pass:
                tier_passed += 1
                total_passed += 1
            total_cases += 1

            status = "[PASS]" if is_pass else "[FAIL]"
            tier_results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": answer,
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
    summary["total_passed"] = total_passed
    summary["total_cases"] = total_cases

    print("\n" + "=" * 75)
    print(f"OVERALL ARITHMETIC EXTRAPOLATION ACCURACY: {overall_acc:.2f}% ({total_passed}/{total_cases})")
    print("=" * 75)
    return summary


def evaluate_tool_benchmarks(
    model: UnifiedGPT,
    tokenizer: UnifiedTokenizer,
    device: torch.device,
    n_eval_cases: int = 50,
    seed: int = 999,
) -> Dict[str, Any]:
    """Evaluates tool selection, call syntax, and direct conversational routing."""
    model.eval()
    eval_dataset = generate_tool_eval_dataset(n_samples=n_eval_cases, seed=seed)

    print("\n" + "=" * 75)
    print("UNIFIED 200K MODEL: TOOL CALLING & ROUTING BENCHMARK (50 CASES)")
    print("=" * 75)

    category_stats: Dict[str, Dict[str, int]] = {}
    total_passed = 0
    total_cases = len(eval_dataset)

    for i, item in enumerate(eval_dataset):
        cat = item["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "pass": 0}
        category_stats[cat]["total"] += 1

        prompt = f"<USER>{item['user_prompt']}</USER><THINK>"
        tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([tokens], dtype=torch.long, device=device)

        t_start = time.time()
        with torch.no_grad():
            stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
            out = model.generate(x, max_new_tokens=160, temperature=0.0, stop_tokens=stop_ids)
        elapsed = time.time() - t_start

        gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
        expected_call = item["call"]

        is_pass = False
        if expected_call != "none":
            tool_name = expected_call.split("(")[0]
            if f"<CALL>{tool_name}(" in gen_text:
                is_pass = True
        else:
            if "<CALL>" not in gen_text and "<ANSWER>" in gen_text:
                is_pass = True

        status = "[PASS]" if is_pass else "[FAIL]"
        if is_pass:
            category_stats[cat]["pass"] += 1
            total_passed += 1

        if i < 10 or not is_pass:
            print(f"[{status}] Case {i+1:2d} ({cat:12s}): {item['user_prompt'][:40]}")
            if not is_pass:
                print(f"       Expected: {expected_call}")
                print(f"       Gen Trace: {gen_text[len(prompt):120]}...")

    print("\n--- Summary By Category ---")
    for cat, stats in sorted(category_stats.items()):
        acc = (stats["pass"] / stats["total"]) * 100.0
        print(f"  {cat:15s}: {stats['pass']:2d}/{stats['total']:2d} ({acc:5.1f}%)")

    overall_acc = (total_passed / total_cases) * 100.0
    print("\n" + "=" * 75)
    print(f"OVERALL TOOL CALLING ACCURACY: {overall_acc:.2f}% ({total_passed}/{total_cases})")
    print("=" * 75)

    return {
        "overall_acc": overall_acc,
        "total_passed": total_passed,
        "total_cases": total_cases,
        "category_stats": category_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Unified ~200k Model (Math + Tools)")
    parser.add_argument("--steps", type=int, default=3500, help="Training steps")
    parser.add_argument("--anneal-steps", type=int, default=1500, help="Number of Phase 2 reasoning annealing steps")
    parser.add_argument("--n-math", type=int, default=2000, help="Number of math curriculum samples")
    parser.add_argument("--n-tools", type=int, default=2000, help="Number of tool dialogue samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Peak learning rate")
    parser.add_argument("--threads", type=int, default=12, help="CPU threads")
    parser.add_argument("--log-interval", type=int, default=250, help="Log interval")
    parser.add_argument("--save-ckpt", type=str, default="champion_model/unified_200k/checkpoints/unified_200k.pt", help="Checkpoint output path")
    parser.add_argument("--eval-only", action="store_true", help="Run benchmark evaluations only")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.threads)

    tokenizer = UnifiedTokenizer()
    config = UnifiedGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=4,
        n_head=4,
        n_embd=64,
        n_recur=3,
        k_rand_shift=16,
    )
    model = UnifiedGPT(config, tokenizer=tokenizer)

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"[INFO] Initialized UnifiedGPT (~200k Recurrent 2D-RoPE):")
    print(f"       Total Parameters:          {total_params:,}")
    print(f"       Non-Embedding Parameters:  {non_emb_params:,}")
    print(f"       Layers: {config.n_layer} (Physical), Recurrence: {config.n_recur} (Effective Depth = {config.n_layer * config.n_recur})")
    print(f"       Heads: {config.n_head}, Dim: {config.n_embd}, Vocab: {config.vocab_size}")
    print(f"[INFO] Active Compute Device: {device} (CPU threads={torch.get_num_threads() if device.type == 'cpu' else 'N/A'})")
    model.to(device)

    ckpt_path = os.path.join(ROOT_DIR, args.save_ckpt)

    if not args.eval_only:
        print(f"\n[INFO] Generating joint datasets ({args.n_math} math + {args.n_tools} tools)...")
        math_ds, tool_ds = generate_joint_dataset(n_math=args.n_math, n_tools=args.n_tools, seed=42)
        print(f"[INFO] Joint dataset ready. Math samples: {len(math_ds)}, Tool samples: {len(tool_ds)}")

        print(f"\n[INFO] Starting Two-Stage Frontier Training for {args.steps} steps (anneal_steps={args.anneal_steps}, batch_size={args.batch_size}, lr={args.lr})...")
        t0 = time.time()
        loss_hist = train_unified_gpt(
            model=model,
            tokenizer=tokenizer,
            math_dataset=math_ds,
            tool_dataset=tool_ds,
            steps=args.steps,
            anneal_steps=args.anneal_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            log_interval=args.log_interval,
            device=device,
        )
        t_elapsed = time.time() - t0
        print(f"[INFO] Training completed in {t_elapsed:.1f}s. Final loss: {loss_hist[-1]:.4f}")

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "final_loss": loss_hist[-1],
            "steps": args.steps,
        }, ckpt_path)
        print(f"[INFO] Saved unified model checkpoint to: {ckpt_path}")
    else:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[INFO] Loaded checkpoint from: {ckpt_path}")
        else:
            print(f"[WARN] Checkpoint not found at {ckpt_path}. Evaluating with initialized weights.")

    math_res = evaluate_math_extrapolation(model, tokenizer, device=device)
    tool_res = evaluate_tool_benchmarks(model, tokenizer, device=device, n_eval_cases=50)

    print("\n" + "=" * 75)
    print("FINAL UNIFIED ~200K MODEL BENCHMARK SUMMARY")
    print("=" * 75)
    m_pass = math_res.get("total_passed", 0)
    m_tot = math_res.get("total_cases", 14)
    t_pass = tool_res.get("total_passed", 0)
    t_tot = tool_res.get("total_cases", 50)
    print(f"  Arithmetic Length Extrapolation (1-20d): {m_pass}/{m_tot} ({100.0 * m_pass / max(1, m_tot):.2f}%)")
    print(f"  Structured Tool Calling & Routing:       {t_pass}/{t_tot} ({100.0 * t_pass / max(1, t_tot):.2f}%)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
