"""
HRM-RoPE Inference & Benchmark Runner.

Usage:
  python inference.py --checkpoint path/to/hrm_rope_c1_final.pt --benchmark
  python inference.py --checkpoint path/to/hrm_rope_c1_final.pt --prompt "12345678901234567890+87654321098765432109="
  python inference.py --interactive
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple
import torch

# Ensure local and root imports work cleanly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from model import HRMRoPE, HRMRoPEConfig
from tokenizer import MathTokenizer
from arithmetic import extract_scratchpad_answer, format_addition_char_scratchpad


EXTRAPOLATION_BENCHMARK = {
    "In-Distribution (1-12 Digits - Trained)": [
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


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[HRMRoPE, MathTokenizer]:
    """Loads HRMRoPE model and tokenizer from saved checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    tokenizer = MathTokenizer()
    cfg_raw = checkpoint.get("config")
    if cfg_raw is None:
        cfg = HRMRoPEConfig(vocab_size=tokenizer.vocab_size)
    elif isinstance(cfg_raw, HRMRoPEConfig):
        cfg = cfg_raw
    elif hasattr(cfg_raw, "__dict__"):
        kwargs = {k: v for k, v in cfg_raw.__dict__.items() if k in HRMRoPEConfig.__dataclass_fields__}
        cfg = HRMRoPEConfig(**kwargs)
    elif isinstance(cfg_raw, dict):
        kwargs = {k: v for k, v in cfg_raw.items() if k in HRMRoPEConfig.__dataclass_fields__}
        cfg = HRMRoPEConfig(**kwargs)
    else:
        cfg = HRMRoPEConfig(vocab_size=tokenizer.vocab_size)

    model = HRMRoPE(cfg).to(device)
    state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
    if isinstance(state, dict):
        model.load_state_dict(state, strict=True)
    model.eval()

    return model, tokenizer


def solve_equation(
    model: HRMRoPE,
    tokenizer: MathTokenizer,
    equation_prompt: str,
    device: torch.device,
    max_new_tokens: int = 1024,
) -> Tuple[str, str, float]:
    """Runs autoregressive scratchpad generation on input equation."""
    prompt = equation_prompt if equation_prompt.endswith("<THINK>") else (equation_prompt + "<THINK>")
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_id=tokenizer.eos_id,
            tokenizer=tokenizer,
        )
    elapsed = time.time() - t0

    gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
    reasoning, answer = extract_scratchpad_answer(gen_text)
    return reasoning, answer, elapsed


def run_benchmark(model: HRMRoPE, tokenizer: MathTokenizer, device: torch.device):
    """Executes the complete 1-20 digit length extrapolation benchmark suite."""
    print("=" * 80)
    print("HRM-RoPE LENGTH EXTRAPOLATION BENCHMARK (1 TO 20 DIGITS)")
    print(f"Device: {device} | Non-Embedding Parameters: {model.get_num_params(non_embedding=True):,}")
    print("=" * 80)

    total_cases = 0
    total_passed = 0

    for cat_name, cases in EXTRAPOLATION_BENCHMARK.items():
        print(f"\n[{cat_name.upper()}] ({len(cases)} cases)")
        cat_passed = 0

        for prompt, expected in cases:
            reasoning, answer, elapsed = solve_equation(model, tokenizer, prompt, device)
            is_correct = (answer == expected)
            status = "[PASS]" if is_correct else "[FAIL]"

            if is_correct:
                cat_passed += 1
                total_passed += 1
            total_cases += 1

            print(f"  {status} {prompt} -> Answer: '{answer}' (Expected: '{expected}') [{elapsed:.2f}s]")

        acc = (cat_passed / len(cases)) * 100
        print(f"  Category Accuracy: {acc:.1f}% ({cat_passed}/{len(cases)})")

    overall_acc = (total_passed / total_cases) * 100
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED: Overall Accuracy = {overall_acc:.2f}% ({total_passed}/{total_cases})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="HRM-RoPE Inference CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hrm_rope_c1_final.pt")
    parser.add_argument("--benchmark", action="store_true", help="Run full 1-20 digit extrapolation suite")
    parser.add_argument("--prompt", type=str, default=None, help="Input equation e.g. '15+25='")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda:0)")
    parser.add_argument("--interactive", action="store_true", help="Interactive terminal prompt")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print(f"[WARN] Checkpoint '{args.checkpoint}' not found.")
        print("[INFO] Initializing untrained HRM-RoPE model for structure verification...")
        model = HRMRoPE(HRMRoPEConfig()).to(device)
        tokenizer = MathTokenizer()
    else:
        print(f"[INFO] Loading checkpoint from {args.checkpoint}...")
        model, tokenizer = load_model(args.checkpoint, device)

    if args.benchmark:
        run_benchmark(model, tokenizer, device)
    elif args.prompt:
        reasoning, answer, elapsed = solve_equation(model, tokenizer, args.prompt, device)
        print(f"Prompt:    {args.prompt}")
        print(f"Reasoning: {reasoning}")
        print(f"Answer:    {answer}")
        print(f"Latency:   {elapsed:.2f}s")
    elif args.interactive:
        print("HRM-RoPE Interactive Mode (type 'exit' or 'quit' to stop):")
        while True:
            try:
                inp = input("Equation (e.g. 1234+5678=): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if inp.lower() in ["exit", "quit"]:
                break
            if not inp.endswith("="):
                inp += "="
            reasoning, answer, elapsed = solve_equation(model, tokenizer, inp, device)
            print(f"  Reasoning: {reasoning}")
            print(f"  Answer:    {answer} [{elapsed:.2f}s]\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
