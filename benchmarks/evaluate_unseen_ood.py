"""
Empirical Evaluation of Zero-Shot Out-of-Distribution (OOD) Copying
on 30 Strictly Unseen Real and Synthetic Entities.
None of these entities exist in the training dictionaries of Curriculum V1-V4.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig
from champion_model.scaled_t4.tokenizer import ScaledTokenizer


# 30 strictly unseen prompts across 5 sub-categories
UNSEEN_OOD_30_CASES = [
    # 1. Unseen Historical & Notable Figures (Multi-word)
    {"id": 1, "domain": "Unseen Figure", "prompt": "<USER>Who was Baruch Spinoza?</USER><THINK>", "expected_tool": "search", "expected_arg": "Baruch Spinoza"},
    {"id": 2, "domain": "Unseen Figure", "prompt": "<USER>Who was Hypatia of Alexandria?</USER><THINK>", "expected_tool": "search", "expected_arg": "Hypatia of Alexandria"},
    {"id": 3, "domain": "Unseen Figure", "prompt": "<USER>Who was Ludwig van Beethoven?</USER><THINK>", "expected_tool": "search", "expected_arg": "Ludwig van Beethoven"},
    {"id": 4, "domain": "Unseen Figure", "prompt": "<USER>Who was Johannes Vermeer?</USER><THINK>", "expected_tool": "search", "expected_arg": "Johannes Vermeer"},
    {"id": 5, "domain": "Unseen Figure", "prompt": "<USER>Who is Ada Yonath?</USER><THINK>", "expected_tool": "search", "expected_arg": "Ada Yonath"},
    {"id": 6, "domain": "Unseen Figure", "prompt": "<USER>Who was Srinivasa Ramanujan?</USER><THINK>", "expected_tool": "search", "expected_arg": "Srinivasa Ramanujan"},

    # 2. Unseen Global Cities & Places (Multi-word & Single)
    {"id": 7, "domain": "Unseen City", "prompt": "<USER>What is the weather in Addis Ababa?</USER><THINK>", "expected_tool": "get_weather", "expected_arg": "Addis Ababa"},
    {"id": 8, "domain": "Unseen City", "prompt": "<USER>What is the weather in Ulaanbaatar?</USER><THINK>", "expected_tool": "get_weather", "expected_arg": "Ulaanbaatar"},
    {"id": 9, "domain": "Unseen City", "prompt": "<USER>Weather on Antananarivo</USER><THINK>", "expected_tool": "get_weather", "expected_arg": "Antananarivo"},
    {"id": 10, "domain": "Unseen City", "prompt": "<USER>How is the weather in Dar es Salaam</USER><THINK>", "expected_tool": "get_weather", "expected_arg": "Dar es Salaam"},
    {"id": 11, "domain": "Unseen City", "prompt": "<USER>Check the weather for Belo Horizonte</USER><THINK>", "expected_tool": "get_weather", "expected_arg": "Belo Horizonte"},
    {"id": 12, "domain": "Unseen City", "prompt": "<USER>Forecast for Chiang Mai.</USER><THINK>", "expected_tool": "get_weather", "expected_arg": "Chiang Mai"},

    # 3. Unseen Scientific Concepts (Multi-word & Single)
    {"id": 13, "domain": "Unseen Science", "prompt": "<USER>What are quarks and leptons</USER><THINK>", "expected_tool": "search", "expected_arg": "quarks and leptons"},
    {"id": 14, "domain": "Unseen Science", "prompt": "<USER>Define epigenetics</USER><THINK>", "expected_tool": "search", "expected_arg": "epigenetics"},
    {"id": 15, "domain": "Unseen Science", "prompt": "<USER>Explain superconductivity</USER><THINK>", "expected_tool": "search", "expected_arg": "superconductivity"},
    {"id": 16, "domain": "Unseen Science", "prompt": "<USER>What is the mitochondrial matrix</USER><THINK>", "expected_tool": "search", "expected_arg": "mitochondrial matrix"},
    {"id": 17, "domain": "Unseen Science", "prompt": "<USER>Explain bose einstein condensate</USER><THINK>", "expected_tool": "search", "expected_arg": "bose einstein condensate"},
    {"id": 18, "domain": "Unseen Science", "prompt": "<USER>What is asymptotic freedom?</USER><THINK>", "expected_tool": "search", "expected_arg": "asymptotic freedom"},

    # 4. Unseen Cuisine & Artifacts
    {"id": 19, "domain": "Unseen Concept", "prompt": "<USER>What is a croissant?</USER><THINK>", "expected_tool": "search", "expected_arg": "croissant"},
    {"id": 20, "domain": "Unseen Concept", "prompt": "<USER>Define baklava</USER><THINK>", "expected_tool": "search", "expected_arg": "baklava"},
    {"id": 21, "domain": "Unseen Concept", "prompt": "<USER>What is shakshuka?</USER><THINK>", "expected_tool": "search", "expected_arg": "shakshuka"},
    {"id": 22, "domain": "Unseen Concept", "prompt": "<USER>Tell me about the astrolabe</USER><THINK>", "expected_tool": "search", "expected_arg": "astrolabe"},
    {"id": 23, "domain": "Unseen Concept", "prompt": "<USER>What is a clavichord</USER><THINK>", "expected_tool": "search", "expected_arg": "clavichord"},
    {"id": 24, "domain": "Unseen Concept", "prompt": "<USER>Explain origami</USER><THINK>", "expected_tool": "search", "expected_arg": "origami"},

    # 5. Completely Unseen Synthetic Phonotactic Nonces
    {"id": 25, "domain": "Unseen Nonce", "prompt": "<USER>Who is Vornik Plaxen?</USER><THINK>", "expected_tool": "search", "expected_arg": "Vornik Plaxen"},
    {"id": 26, "domain": "Unseen Nonce", "prompt": "<USER>Define Zyntral Krelvok</USER><THINK>", "expected_tool": "search", "expected_arg": "Zyntral Krelvok"},
    {"id": 27, "domain": "Unseen Nonce", "prompt": "<USER>Weather on Brixendale</USER><THINK>", "expected_tool": "get_weather", "expected_arg": "Brixendale"},
    {"id": 28, "domain": "Unseen Nonce", "prompt": "<USER>What is Glarndoxification?</USER><THINK>", "expected_tool": "search", "expected_arg": "Glarndoxification"},
    {"id": 29, "domain": "Unseen Nonce", "prompt": "<USER>What is a Trundleblump</USER><THINK>", "expected_tool": "search", "expected_arg": "Trundleblump"},
    {"id": 30, "domain": "Unseen Nonce", "prompt": "<USER>Tell me about a Skrevling</USER><THINK>", "expected_tool": "search", "expected_arg": "Skrevling"},
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


def exact_binomial_pvalue(k: int, n: int, p0: float = 0.50) -> float:
    """Computes one-sided upper-tail binomial p-value testing H0: p <= p0 vs H1: p > p0."""
    p_val = 0.0
    for i in range(k, n + 1):
        comb = math.comb(n, i)
        p_val += comb * (p0 ** i) * ((1.0 - p0) ** (n - i))
    return p_val


def parse_call(resp: str) -> Optional[str]:
    """Extracts contents within <CALL>...</CALL>."""
    if "<CALL>" in resp and "</CALL>" in resp:
        return resp.split("<CALL>")[1].split("</CALL>")[0].strip()
    return None


@torch.no_grad()
def evaluate_unseen_suite(
    model: ScaledGPT,
    tokenizer: ScaledTokenizer,
    device: torch.device,
) -> Dict[str, Any]:
    print("=" * 95)
    print(f"STRICT UNSEEN ZERO-SHOT OOD COPYING BENCHMARK ({len(UNSEEN_OOD_30_CASES)} CASES)")
    print("Zero overlap with training CITIES, SEARCH_ENTITIES, or SEARCH_FACTS")
    print("=" * 95)

    routing_passed = 0
    exact_copy_passed = 0
    total = len(UNSEEN_OOD_30_CASES)
    itemized = []

    for item in UNSEEN_OOD_30_CASES:
        toks = tokenizer.encode(item["prompt"], add_bos=True, add_eos=False)
        x = torch.tensor([toks], dtype=torch.long, device=device)
        stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
        out = model.generate(x, max_new_tokens=140, temperature=0.0, stop_tokens=stop_ids)
        new_toks = out[0][len(toks):].tolist()
        resp = tokenizer.decode(new_toks, skip_special=False).strip()

        call_content = parse_call(resp)
        exp_tool = item["expected_tool"]
        exp_arg = item["expected_arg"]

        route_ok = False
        exact_copy_ok = False

        if call_content is not None:
            if call_content.startswith(f"{exp_tool}("):
                route_ok = True
                # Check exact argument content inside parameter quotes
                # Expected format: search(query="<arg>") or get_weather(city="<arg>")
                if f'query="{exp_arg}"' in call_content or f'city="{exp_arg}"' in call_content:
                    exact_copy_ok = True

        if route_ok:
            routing_passed += 1
        if exact_copy_ok:
            exact_copy_passed += 1

        status_tag = "[PASS: EXACT]" if exact_copy_ok else ("[ROUTE_OK]" if route_ok else "[FAIL]")
        disp_resp = f"<CALL>{call_content}</CALL>" if call_content else resp[:60]
        print(f"  #{item['id']:2d} {status_tag:<14} {item['domain']:<16} | Target: '{exp_arg}' -> {disp_resp}", flush=True)

        itemized.append({
            "id": item["id"],
            "domain": item["domain"],
            "prompt": item["prompt"],
            "expected_tool": exp_tool,
            "expected_arg": exp_arg,
            "raw_response": resp,
            "parsed_call": call_content,
            "route_ok": route_ok,
            "exact_copy_ok": exact_copy_ok,
        })

    route_pct = 100.0 * routing_passed / total
    exact_pct = 100.0 * exact_copy_passed / total
    copy_given_routed_pct = (100.0 * exact_copy_passed / routing_passed) if routing_passed > 0 else 0.0

    route_ci = wilson_ci(routing_passed, total)
    exact_ci = wilson_ci(exact_copy_passed, total)
    copy_given_routed_ci = wilson_ci(exact_copy_passed, routing_passed)

    binom_p = exact_binomial_pvalue(exact_copy_passed, total, p0=0.50)

    print("-" * 95)
    print(f"[SUMMARY] Tool Routing:              {routing_passed:2d} / {total:2d} ({route_pct:5.1f}%, 95% CI {route_ci}%)")
    print(f"[SUMMARY] Exact Copy (Given Routed): {exact_copy_passed:2d} / {routing_passed:2d} ({copy_given_routed_pct:5.1f}%, 95% CI {copy_given_routed_ci}%)")
    print(f"[SUMMARY] Headline Exact Match:      {exact_copy_passed:2d} / {total:2d} ({exact_pct:5.1f}%, 95% CI {exact_ci}%)")
    print(f"[SUMMARY] One-Sided Binomial vs 50%: p = {binom_p:.6e} ({'Statistically > 50% (p < 0.05)' if binom_p < 0.05 else 'NOT statistically > 50%'})")
    print("=" * 95 + "\n")

    return {
        "total": total,
        "routing_passed": routing_passed,
        "routing_pct": route_pct,
        "routing_ci": route_ci,
        "exact_copy_passed": exact_copy_passed,
        "headline_exact_pct": exact_pct,
        "headline_exact_ci": exact_ci,
        "copy_given_routed_pct": copy_given_routed_pct,
        "copy_given_routed_ci": copy_given_routed_ci,
        "binomial_p_value_vs_50pct": binom_p,
        "is_statistically_gt_50pct": bool(binom_p < 0.05),
        "itemized": itemized,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CBT on Unseen Zero-Shot OOD Benchmark")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--out-json", type=str, default="", help="Optional output JSON path")
    args = parser.parse_args()

    device = torch.device("cpu")
    tok = ScaledTokenizer()

    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    if "config" in ckpt and ckpt["config"] is not None:
        config = ckpt["config"]
    else:
        config = ScaledGPTConfig(
            vocab_size=tok.vocab_size,
            block_size=1024,
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
    model.to(device)
    model.eval()

    results = evaluate_unseen_suite(model, tok, device)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved results to {args.out_json}")


if __name__ == "__main__":
    main()
