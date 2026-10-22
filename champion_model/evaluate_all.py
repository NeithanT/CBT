"""
Unified Evaluation Runner for Champion Models (~100k Recurrent 2D-RoPE).
Evaluates both:
1. Arithmetic Length Extrapolation (1-20 Digits across 14 benchmark test cases).
2. ToolGPT Structured Tool Calling (50 test cases across 6 functional categories).
"""

import os
import sys
import argparse
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.math_model.model import MathGPT, GPTConfig
from champion_model.math_model.train_and_eval import evaluate_benchmarks
from champion_model.tool_model.model import ToolGPT, ToolGPTConfig
from champion_model.tool_model.tokenizer import ToolTokenizer
from champion_model.tool_model.train_and_eval import evaluate_tool_benchmarks
from champion_model.math_model.tokenizer import MathTokenizer


def run_math_eval(device: torch.device) -> bool:
    print("\n" + "=" * 75)
    print("RUNNING CHAMPION ARITHMETIC BENCHMARK (1-20 DIGITS EXTRAPOLATION)")
    print("=" * 75)

    ckpt_path = os.path.join(ROOT_DIR, "champion_model/math_model/checkpoints/dense_100k_2d_rec3_full.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, "dense_model/checkpoints/dense_100k_2d_rec3_full.pt")

    tok = MathTokenizer()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = MathGPT(config)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded checkpoint from: {ckpt_path}")
    else:
        config = GPTConfig.create_100k(
            vocab_size=tok.vocab_size,
            block_size=256,
            pos_emb_type="rope_2d",
            n_recur=3,
            k_rand_shift=16,
        )
        model = MathGPT(config)
        print(f"[WARN] Checkpoint not found at {ckpt_path}, running initialized weights.")

    print(f"[INFO] Initialized MathGPT (~100k Recurrent 2D-RoPE):")
    print(f"       Total Parameters:          {model.get_num_params(non_embedding=False):,}")
    print(f"       Non-Embedding Parameters:  {model.get_num_params(non_embedding=True):,}")
    print(f"       Effective Depth:           {config.n_layer * config.n_recur} layers ({config.n_layer} physical x {config.n_recur} recurrent)")

    model.to(device)
    summary = evaluate_benchmarks(model, tok, device=device)
    overall = summary.get("overall", {})
    total_cases = overall.get("total", 14)
    total_passed = overall.get("passed", 0)
    print(f"\n[RESULT] Arithmetic Benchmark: {total_passed}/{total_cases} ({100.0 * total_passed / max(1, total_cases):.2f}%)")
    return total_passed == total_cases


def run_tools_eval(device: torch.device) -> bool:
    print("\n" + "=" * 75)
    print("RUNNING CHAMPION TOOLGPT BENCHMARK (50 STRUCTURED TOOL ROUTING CASES)")
    print("=" * 75)

    ckpt_path = os.path.join(ROOT_DIR, "champion_model/tool_model/checkpoints/tool_gpt_100k.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, "tool_model/checkpoints/tool_gpt_100k.pt")

    tok = ToolTokenizer()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = ToolGPT(config, tokenizer=tok)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded checkpoint from: {ckpt_path}")
    else:
        config = ToolGPTConfig(
            vocab_size=tok.vocab_size,
            block_size=512,
            n_layer=2,
            n_head=4,
            n_embd=64,
            n_recur=3,
            pos_emb_type="rope_2d",
        )
        model = ToolGPT(config, tokenizer=tok)
        print(f"[WARN] Checkpoint not found at {ckpt_path}, running initialized weights.")

    print(f"[INFO] Initialized ToolGPT (~100k Recurrent 2D-Discourse RoPE):")
    print(f"       Total Parameters:          {model.get_num_params(non_embedding=False):,}")
    print(f"       Non-Embedding Parameters:  {model.get_num_params(non_embedding=True):,}")
    print(f"       Effective Depth:           {config.n_layer * config.n_recur} layers ({config.n_layer} physical x {config.n_recur} recurrent)")

    model.to(device)
    res = evaluate_tool_benchmarks(model, tok, device=device, n_eval_cases=50)
    passed = res.get("total_passed", 0)
    total = res.get("total_cases", 50)
    print(f"\n[RESULT] ToolGPT Benchmark: {passed}/{total} ({100.0 * passed / max(1, total):.2f}%)")
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Evaluate Champion 100k Models (Math & Tools)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "math", "tools"], help="Evaluation target")
    parser.add_argument("--device", type=str, default="", help="Device: cuda or cpu")
    parser.add_argument("--threads", type=int, default=12, help="CPU threads")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.threads)

    print(f"[INFO] Device: {device}")

    math_ok = True
    tools_ok = True

    if args.mode in ["all", "math"]:
        math_ok = run_math_eval(device)

    if args.mode in ["all", "tools"]:
        tools_ok = run_tools_eval(device)

    print("\n" + "=" * 75)
    print("CHAMPION MODEL SUITE SUMMARY")
    print("=" * 75)
    if args.mode in ["all", "math"]:
        print(f"  Arithmetic Length Extrapolation (1-20d): {'[PASS] 100.0% (14/14)' if math_ok else '[FAIL]'}")
    if args.mode in ["all", "tools"]:
        print(f"  ToolGPT Tool Calling & Routing (50 cases): {'[PASS] 100.0% (50/50)' if tools_ok else '[FAIL]'}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
