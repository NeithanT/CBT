#!/usr/bin/env python3
"""
Unified Evaluation Benchmark Suite for Scaled CBT (~18.95M Parameters).
Evaluates:
1. Multi-Tool Phrasing & Edge Cases (26 test cases across all 9 schemas).
2. Zero-Shot Out-of-Distribution (OOD) Generalization Suite (18 unseen prompts).
3. Random Multi-Carry Arithmetic Extrapolation (16 to 50 digits, plus cascade carries).

Zero Emojis Policy: strictly adheres to [OK], [PASS], [FAIL], [WARN], [INFO].
"""

import os
import sys
import re
import json
import time
import math
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig, apply_2d_rope
from champion_model.scaled_t4.tokenizer import ScaledTokenizer
try:
    from champion_model.scaled_t4.dataset import format_addition_char_scratchpad, extract_scratchpad_answer
except ImportError:
    from champion_model.math_model.arithmetic import format_addition_char_scratchpad, extract_scratchpad_answer


# -----------------------------------------------------------------------------
# 1. Benchmark Definitions
# -----------------------------------------------------------------------------

# Table 4: 26 Phrasing and Edge Cases across 9 tool schemas
PHRASING_26_CASES = [
    {"label": "Weather (on)", "prompt": "<USER>What is the weather on Rome</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Rome"}},
    {"label": "Weather (in)", "prompt": "<USER>What is the weather in Rome?</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Rome"}},
    {"label": "Weather (short)", "prompt": "<USER>Weather in Tokyo</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Tokyo"}},
    {"label": "Weather (How is)", "prompt": "<USER>How is the weather in Paris</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Paris"}},
    {"label": "Weather (Check)", "prompt": "<USER>Check the weather for London</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "London"}},
    {"label": "Weather (Forecast)", "prompt": "<USER>Forecast for Berlin.</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Berlin"}},
    {"label": "Weather (Nonce)", "prompt": "<USER>What is the weather in Bloumplik?</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Bloumplik"}},
    {"label": "Time (in EST)", "prompt": "<USER>What is the time in EST</USER><THINK>", "expected_tool": "get_time", "expected_args": {"timezone": "EST"}},
    {"label": "Time (What time)", "prompt": "<USER>What time is it in JST</USER><THINK>", "expected_tool": "get_time", "expected_args": {"timezone": "JST"}},
    {"label": "Time (Current)", "prompt": "<USER>Current time in UTC</USER><THINK>", "expected_tool": "get_time", "expected_args": {"timezone": "UTC"}},
    {"label": "Unit (Convert)", "prompt": "<USER>Convert 10 km to miles</USER><THINK>", "expected_tool": "unit_convert", "expected_args": {"value": 10, "from_unit": "km", "to_unit": "miles"}},
    {"label": "Unit (How many)", "prompt": "<USER>How many miles is 25 km</USER><THINK>", "expected_tool": "unit_convert", "expected_args": {"value": 25, "from_unit": "km", "to_unit": "miles"}},
    {"label": "Search (Real 1)", "prompt": "<USER>What are tomatoes</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "tomatoes"}},
    {"label": "Search (Real 2)", "prompt": "<USER>Who is Alexander Graham Bell?</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "Alexander Graham Bell"}},
    {"label": "Search (Nonce 1)", "prompt": "<USER>What is a bloumplik</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "bloumplik"}},
    {"label": "Search (Nonce 2)", "prompt": "<USER>Define zaimgromp</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "zaimgromp"}},
    {"label": "Search (Lookup)", "prompt": "<USER>Look up quantum physics</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "quantum physics"}},
    {"label": "Calculator (Mult)", "prompt": "<USER>Calculate 15 * 24</USER><THINK>", "expected_tool": "calculator", "expected_match": ["15 * 24", "15*24"]},
    {"label": "Calculator (Add)", "prompt": "<USER>Compute 128 + 492</USER><THINK>", "expected_tool": "calculator", "expected_match": ["128 + 492", "128+492"]},
    {"label": "Calculator (Div)", "prompt": "<USER>Solve 500 / 25</USER><THINK>", "expected_tool": "calculator", "expected_match": ["500 / 25", "500/25"]},
    {"label": "Nested DB Query", "prompt": "<USER>Query top 5 active records in engineering.</USER><THINK>", "expected_tool": "database_query", "expected_match": ["department", "engineering"]},
    {"label": "Fetch API", "prompt": "<USER>Fetch latency metrics for EU from /v2/metrics.</USER><THINK>", "expected_tool": "fetch_api", "expected_match": ["/v2/metrics", "latency"]},
    {"label": "Create Event", "prompt": "<USER>Create a calendar event for Team Standup at 10:00.</USER><THINK>", "expected_tool": "create_event", "expected_match": ["Team Standup", "10:00"]},
    {"label": "File Manager", "prompt": "<USER>Save report.txt with content Hello World</USER><THINK>", "expected_tool": "file_manager", "expected_match": ["report.txt", "Hello World"]},
    {"label": "Direct Chat 1", "prompt": "<USER>Thank you!</USER><THINK>", "expected_tool": "no_tool", "expected_match": ["<ANSWER>"]},
    {"label": "Direct Chat 2", "prompt": "<USER>Hello, how are you today?</USER><THINK>", "expected_tool": "no_tool", "expected_match": ["<ANSWER>"]},
]

# Table 5: 18 Zero-Shot Out-of-Distribution Prompts
OOD_18_CASES = [
    {"domain": "OOD Science", "prompt": "<USER>What is quantum physics</USER><THINK>", "expected_call": 'search(query="quantum physics")'},
    {"domain": "OOD Film/Term", "prompt": "<USER>What is interstellar</USER><THINK>", "expected_call": 'search(query="interstellar")'},
    {"domain": "OOD Economics", "prompt": "<USER>Define hyperinflation</USER><THINK>", "expected_call": 'search(query="hyperinflation")'},
    {"domain": "OOD Cuisine", "prompt": "<USER>What is tiramisu?</USER><THINK>", "expected_call": 'search(query="tiramisu")'},
    {"domain": "OOD Astronomy", "prompt": "<USER>What are neutron stars</USER><THINK>", "expected_call": 'search(query="neutron stars")'},
    {"domain": "OOD Weather (City)", "prompt": "<USER>How is the weather in Reykjavik</USER><THINK>", "expected_call": 'get_weather(city="Reykjavik")'},
    {"domain": "OOD Unit (Inverted)", "prompt": "<USER>How many km is 42 miles</USER><THINK>", "expected_call": 'unit_convert(value=42, from_unit="miles", to_unit="km")'},
    {"domain": "OOD Timezone", "prompt": "<USER>What time is it in GMT</USER><THINK>", "expected_call": 'get_time(timezone="GMT")'},
    {"domain": "OOD Philosophy", "prompt": "<USER>What is existentialism?</USER><THINK>", "expected_call": 'search(query="existentialism")'},
    {"domain": "OOD Mythology", "prompt": "<USER>Tell me about Ragnarok</USER><THINK>", "expected_call": 'search(query="Ragnarok")'},
    {"domain": "OOD Calculator", "prompt": "<USER>Compute 84 * 19</USER><THINK>", "expected_call_candidates": ['calculator(expr="84 * 19")', 'calculator(expr="84*19")']},
    {"domain": "OOD Deep Learning", "prompt": "<USER>What are transformers in deep learning</USER><THINK>", "expected_call": 'search(query="transformers in deep learning")'},
    {"domain": "OOD Music/Instrument", "prompt": "<USER>What is a theremin</USER><THINK>", "expected_call": 'search(query="theremin")'},
    {"domain": "OOD History (2-Word)", "prompt": "<USER>Who was Marcus Aurelius?</USER><THINK>", "expected_call": 'search(query="Marcus Aurelius")'},
    {"domain": "OOD Weather (2-Word)", "prompt": "<USER>What is the weather in Buenos Aires?</USER><THINK>", "expected_call": 'get_weather(city="Buenos Aires")'},
    {"domain": "OOD Weather (2-Word)", "prompt": "<USER>Weather on Kuala Lumpur</USER><THINK>", "expected_call": 'get_weather(city="Kuala Lumpur")'},
    {"domain": "OOD Bio (With Digit)", "prompt": "<USER>Explain crispr cas9</USER><THINK>", "expected_call": 'search(query="crispr cas9")'},
    {"domain": "OOD Architecture", "prompt": "<USER>Search for brutalist architecture</USER><THINK>", "expected_call": 'search(query="brutalist architecture")'},
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes 95% Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    lo = max(0.0, (center - half) * 100.0)
    hi = min(100.0, (center + half) * 100.0)
    return (round(lo, 1), round(hi, 1))


def parse_call(resp: str) -> Optional[str]:
    """Extracts contents within <CALL>...</CALL>."""
    if "<CALL>" in resp and "</CALL>" in resp:
        return resp.split("<CALL>")[1].split("</CALL>")[0].strip()
    return None


@torch.no_grad()
def evaluate_phrasing_suite(model: ScaledGPT, tokenizer: ScaledTokenizer, device: torch.device) -> Dict[str, Any]:
    print("=" * 80)
    print(f"SUITE 1: MULTI-TOOL PHRASING & EDGE CASES ({len(PHRASING_26_CASES)} CASES)")
    print("=" * 80)

    routing_passed = 0
    exact_args_passed = 0
    total = len(PHRASING_26_CASES)

    for item in PHRASING_26_CASES:
        toks = tokenizer.encode(item["prompt"], add_bos=True, add_eos=False)
        x = torch.tensor([toks], dtype=torch.long, device=device)
        stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
        out = model.generate(x, max_new_tokens=140, temperature=0.0, stop_tokens=stop_ids)
        new_toks = out[0][len(toks):].tolist()
        resp = tokenizer.decode(new_toks, skip_special=False).strip()

        call_content = parse_call(resp)
        exp_tool = item["expected_tool"]

        # Check routing
        route_ok = False
        args_ok = False

        if exp_tool == "no_tool":
            route_ok = ("<ANSWER>" in resp and "<CALL>" not in resp)
            args_ok = route_ok
        elif call_content is not None:
            if call_content.startswith(f"{exp_tool}("):
                route_ok = True
                # Check arguments
                if "expected_args" in item:
                    args_ok = True
                    for k, v in item["expected_args"].items():
                        target_str = f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                        if target_str not in call_content:
                            args_ok = False
                            break
                elif "expected_match" in item:
                    args_ok = any(m in call_content for m in item["expected_match"])
                else:
                    args_ok = True

        if route_ok:
            routing_passed += 1
        if args_ok:
            exact_args_passed += 1

        status_tag = "[PASS]" if args_ok else ("[ROUTE_OK]" if route_ok else "[FAIL]")
        disp_resp = f"<CALL>{call_content}</CALL>" if call_content else resp[:50]
        print(f"  {status_tag:<10} {item['label']:<24} -> {disp_resp}")

    route_pct = 100.0 * routing_passed / total
    args_pct = 100.0 * exact_args_passed / total
    route_ci = wilson_ci(routing_passed, total)
    args_ci = wilson_ci(exact_args_passed, total)

    print("-" * 80)
    print(f"[SUMMARY] Tool Routing:  {routing_passed}/{total} ({route_pct:.1f}%, 95% CI {route_ci}%)")
    print(f"[SUMMARY] Exact Args:    {exact_args_passed}/{total} ({args_pct:.1f}%, 95% CI {args_ci}%)")
    print("=" * 80 + "\n")

    return {
        "total": total,
        "routing_passed": routing_passed,
        "routing_pct": route_pct,
        "exact_args_passed": exact_args_passed,
        "exact_args_pct": args_pct,
    }


@torch.no_grad()
def evaluate_ood_suite(model: ScaledGPT, tokenizer: ScaledTokenizer, device: torch.device) -> Dict[str, Any]:
    print("=" * 80)
    print(f"SUITE 2: ZERO-SHOT OUT-OF-DISTRIBUTION (OOD) BENCHMARK ({len(OOD_18_CASES)} PROMPTS)")
    print("=" * 80)

    routed = 0
    exact_matches = 0
    total = len(OOD_18_CASES)

    for item in OOD_18_CASES:
        toks = tokenizer.encode(item["prompt"], add_bos=True, add_eos=False)
        x = torch.tensor([toks], dtype=torch.long, device=device)
        stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
        out = model.generate(x, max_new_tokens=140, temperature=0.0, stop_tokens=stop_ids)
        new_toks = out[0][len(toks):].tolist()
        resp = tokenizer.decode(new_toks, skip_special=False).strip()

        call_content = parse_call(resp)
        candidate_calls = item.get("expected_call_candidates", [item.get("expected_call")])

        # Check expected tool name
        first_call = candidate_calls[0]
        exp_tool = first_call.split("(")[0]

        is_routed = (call_content is not None and call_content.startswith(f"{exp_tool}("))
        is_exact = False
        if call_content is not None:
            is_exact = any(c == call_content for c in candidate_calls)

        if is_routed:
            routed += 1
        if is_exact:
            exact_matches += 1

        status_tag = "[EXACT]" if is_exact else ("[ROUTED]" if is_routed else "[DRIFT]")
        disp_resp = f"<CALL>{call_content}</CALL>" if call_content else resp[:50]
        print(f"  {status_tag:<8} {item['domain']:<22} -> {disp_resp}")

    route_pct = 100.0 * routed / total
    exact_pct = 100.0 * exact_matches / max(1, routed)
    headline_pct = 100.0 * exact_matches / total
    route_ci = wilson_ci(routed, total)
    exact_ci = wilson_ci(exact_matches, routed)

    print("-" * 80)
    print(f"[SUMMARY] OOD Tool Selection:      {routed}/{total} ({route_pct:.1f}%, 95% CI {route_ci}%)")
    print(f"[SUMMARY] OOD Exact Argument Copy: {exact_matches}/{routed} ({exact_pct:.1f}%, 95% CI {exact_ci}%)")
    print(f"[SUMMARY] OOD Headline Exact:      {exact_matches}/{total} ({headline_pct:.1f}%)")
    print("=" * 80 + "\n")

    return {
        "total": total,
        "routed": routed,
        "route_pct": route_pct,
        "exact_matches": exact_matches,
        "exact_pct": exact_pct,
    }


@torch.no_grad()
def evaluate_arithmetic_suite(
    model: ScaledGPT,
    tokenizer: ScaledTokenizer,
    device: torch.device,
    samples_per_length: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"SUITE 3: RANDOM MULTI-CARRY ARITHMETIC EXTRAPOLATION (16 TO 50 DIGITS)")
    print(f"Samples per length: {samples_per_length} | Seed: {seed}")
    print("=" * 80)

    lengths = [16, 20, 25, 30, 35, 40, 45, 50]
    rng = random.Random(seed)

    per_length_stats = {}
    pooled_total = 0
    pooled_passed = 0

    for digits in lengths:
        passed = 0
        t0 = time.time()
        for i in range(samples_per_length):
            # 70% uniform random, 15% dense carry (sums >= 9), 15% cascade mix (999... + 1)
            p_type = rng.random()
            if p_type < 0.15:
                a = int("9" * digits)
                b = rng.randint(1, 9)
            elif p_type < 0.30:
                # Dense carry
                a_str = "".join(str(rng.randint(5, 9)) for _ in range(digits))
                b_str = "".join(str(rng.randint(5, 9)) for _ in range(digits))
                a, b = int(a_str), int(b_str)
            else:
                min_v = 10 ** (digits - 1)
                max_v = (10 ** digits) - 1
                a = rng.randint(min_v, max_v)
                b = rng.randint(min_v, max_v)

            expected_sum = a + b
            prompt = f"{a}+{b}=<THINK>"
            toks = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            x = torch.tensor([toks], dtype=torch.long, device=device)
            max_gen = (digits + 2) * 15 + 40
            out = model.generate(x, max_new_tokens=max_gen, temperature=0.0, stop_tokens=[tokenizer.eos_id])
            gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)

            pred_ans = extract_scratchpad_answer(gen_text)
            if pred_ans == expected_sum:
                passed += 1

        elapsed = time.time() - t0
        pct = 100.0 * passed / samples_per_length
        ci = wilson_ci(passed, samples_per_length)
        pooled_total += samples_per_length
        pooled_passed += passed
        per_length_stats[digits] = {"passed": passed, "total": samples_per_length, "accuracy_pct": pct, "ci": ci}

        status = "[PASS]" if passed == samples_per_length else "[FAIL]"
        print(f"  {status} {digits:2d} Digits: {passed:2d}/{samples_per_length:2d} ({pct:5.1f}%, 95% CI {ci}%) | {elapsed:.1f}s")

    pooled_pct = 100.0 * pooled_passed / pooled_total
    pooled_ci = wilson_ci(pooled_passed, pooled_total)

    print("-" * 80)
    print(f"[SUMMARY] Pooled 16-50d Arithmetic: {pooled_passed}/{pooled_total} ({pooled_pct:.2f}%, 95% CI {pooled_ci}%)")
    print("=" * 80 + "\n")

    return {
        "per_length": per_length_stats,
        "pooled_passed": pooled_passed,
        "pooled_total": pooled_total,
        "pooled_pct": pooled_pct,
        "pooled_ci": pooled_ci,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scaled CBT Champion V2 Model")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "phrasing", "ood", "arithmetic"],
        help="Target benchmark suite",
    )
    parser.add_argument("--samples-per-length", type=int, default=5, help="Samples per arithmetic length (default 5, use 30 for full sweep)")
    parser.add_argument("--device", type=str, default="", help="Inference device: cuda or cpu")
    parser.add_argument("--full", action="store_true", help="Run full 30-sample-per-length arithmetic sweep")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print("SCALED CBT CHAMPION EVALUATION RUNNER")
    print(f"Device: {device}")
    print("=" * 80)

    # Resolve checkpoint
    resolved_ckpt = ""
    candidates = [
        args.ckpt,
        "champion_model/scaled_t4/checkpoints/champion_v2_final.pt",
        "champion_model/scaled_t4/checkpoints/champion_final.pt",
        "champion_model/checkpoints/champion_final.pt",
        "champion_model/scaled_t4/checkpoints/scaled_t4_sft_all_tools.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            resolved_ckpt = c
            break

    if not resolved_ckpt:
        print("[FAIL] No valid checkpoint found. Specify --ckpt <path>")
        sys.exit(1)

    print(f"[INFO] Loading checkpoint: {resolved_ckpt}")
    tokenizer = ScaledTokenizer()
    ckpt_data = torch.load(resolved_ckpt, map_location=device, weights_only=False)
    config = ckpt_data.get("config", ScaledGPTConfig(vocab_size=tokenizer.vocab_size))
    model = ScaledGPT(config, tokenizer=tokenizer).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    print("[PASS] Model successfully initialized and loaded.\n")

    results = {}
    if args.suite in ["all", "phrasing"]:
        results["phrasing"] = evaluate_phrasing_suite(model, tokenizer, device)

    if args.suite in ["all", "ood"]:
        results["ood"] = evaluate_ood_suite(model, tokenizer, device)

    if args.suite in ["all", "arithmetic"]:
        n_samples = 30 if args.full else args.samples_per_length
        results["arithmetic"] = evaluate_arithmetic_suite(model, tokenizer, device, samples_per_length=n_samples)

    print("=" * 80)
    print("ALL EVALUATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
