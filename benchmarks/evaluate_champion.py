#!/usr/bin/env python3
"""
Contamination-Audited Evaluation Suite for Scaled CBT Champion (~18.95M Parameters).

Suites:
1. Multi-Tool Phrasing & Edge Cases (26 cases, 9 tool schemas).
2. Zero-Shot OOD Copying (18 cases, fresh entities).
3. Random Multi-Carry Arithmetic (seen 20/25 digits, unseen 26-70 digits).

Contamination guard: every free-text expected value is checked against the champion SFT training
cache (sft_dataset_v4.pt) and the dataset source. Any hit aborts the run. Closed-vocabulary
schema enumerations (units, timezones) and template words are exempt and disclosed in the report.

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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig
from champion_model.scaled_t4.tokenizer import ScaledTokenizer
from champion_model.scaled_t4.dataset import format_addition_char_scratchpad, extract_scratchpad_answer
from benchmarks.evaluate_unseen_ood import UNSEEN_OOD_30_CASES

DEFAULT_CKPT = "checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sft.pt"
TRAIN_CACHE = "champion_model/scaled_t4/data/sft_dataset_v4.pt"
DATASET_SRC = "champion_model/scaled_t4/dataset.py"

# Seen arithmetic lengths (<= max training operand digits) vs. unseen extrapolation lengths.
ARITH_SEEN_DIGITS = [20, 25]
ARITH_UNSEEN_DIGITS = [26, 30, 35, 40, 45, 50, 55, 60, 70]

# Closed-vocabulary schema keys: enumerations defined by the tool schema, exempt from the audit.
CLOSED_KEYS = {"timezone", "from_unit", "to_unit", "unit"}


# -----------------------------------------------------------------------------
# 1. Benchmark Definitions
# -----------------------------------------------------------------------------

PHRASING_26_CASES = [
    {"label": "Weather (on)", "prompt": "<USER>What is the weather on Lagos</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Lagos"}},
    {"label": "Weather (in)", "prompt": "<USER>What is the weather in Nairobi?</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Nairobi"}},
    {"label": "Weather (short)", "prompt": "<USER>Weather in Hanoi</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Hanoi"}},
    {"label": "Weather (How is)", "prompt": "<USER>How is the weather in Accra</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Accra"}},
    {"label": "Weather (Check)", "prompt": "<USER>Check the weather for Lima</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Lima"}},
    {"label": "Weather (Forecast)", "prompt": "<USER>Forecast for Yerevan.</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Yerevan"}},
    {"label": "Weather (Nonce)", "prompt": "<USER>What is the weather in Bloumplik?</USER><THINK>", "expected_tool": "get_weather", "expected_args": {"city": "Bloumplik"}},
    {"label": "Time (in NZST)", "prompt": "<USER>What is the time in NZST</USER><THINK>", "expected_tool": "get_time", "expected_args": {"timezone": "NZST"}},
    {"label": "Time (What time)", "prompt": "<USER>What time is it in SAST</USER><THINK>", "expected_tool": "get_time", "expected_args": {"timezone": "SAST"}},
    {"label": "Time (Current)", "prompt": "<USER>Current time in WAT</USER><THINK>", "expected_tool": "get_time", "expected_args": {"timezone": "WAT"}},
    {"label": "Unit (Convert)", "prompt": "<USER>Convert 10 km to miles</USER><THINK>", "expected_tool": "unit_convert", "expected_args": {"value": 10, "from_unit": "km", "to_unit": "miles"}},
    {"label": "Unit (How many)", "prompt": "<USER>How many miles is 25 km</USER><THINK>", "expected_tool": "unit_convert", "expected_args": {"value": 25, "from_unit": "km", "to_unit": "miles"}},
    {"label": "Search (Real 1)", "prompt": "<USER>What are tardigrades</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "tardigrades"}},
    {"label": "Search (Real 2)", "prompt": "<USER>Who is Johannes Kepler?</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "Johannes Kepler"}},
    {"label": "Search (Nonce 1)", "prompt": "<USER>What is a velmorant</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "velmorant"}},
    {"label": "Search (Nonce 2)", "prompt": "<USER>Define zaimgromp</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "zaimgromp"}},
    {"label": "Search (Lookup)", "prompt": "<USER>Look up mycelium</USER><THINK>", "expected_tool": "search", "expected_args": {"query": "mycelium"}},
    {"label": "Calculator (Mult)", "prompt": "<USER>Calculate 37 * 113</USER><THINK>", "expected_tool": "calculator", "expected_match": ["37 * 113", "37*113"]},
    {"label": "Calculator (Add)", "prompt": "<USER>Compute 419 + 877</USER><THINK>", "expected_tool": "calculator", "expected_match": ["419 + 877", "419+877"]},
    {"label": "Calculator (Div)", "prompt": "<USER>Solve 845 / 13</USER><THINK>", "expected_tool": "calculator", "expected_match": ["845 / 13", "845/13"]},
    {"label": "Nested DB Query", "prompt": "<USER>Query top 5 active records in logistics.</USER><THINK>", "expected_tool": "database_query", "expected_match": ["logistics"]},
    {"label": "Fetch API", "prompt": "<USER>Fetch throughput metrics for APAC from /v3/telemetry.</USER><THINK>", "expected_tool": "fetch_api", "expected_match": ["/v3/telemetry", "throughput"]},
    {"label": "Create Event", "prompt": "<USER>Create a calendar event for Quarterly Review on Friday.</USER><THINK>", "expected_tool": "create_event", "expected_match": ["Quarterly Review"]},
    {"label": "File Manager", "prompt": "<USER>Save report.txt with content Hello World</USER><THINK>", "expected_tool": "file_manager", "expected_match": ["report.txt", "Hello World"]},
    {"label": "Direct Chat 1", "prompt": "<USER>Thank you!</USER><THINK>", "expected_tool": "no_tool", "expected_match": ["<ANSWER>"]},
    {"label": "Direct Chat 2", "prompt": "<USER>Hello, how are you today?</USER><THINK>", "expected_tool": "no_tool", "expected_match": ["<ANSWER>"]},
]

OOD_18_CASES = [
    {"domain": "OOD Science", "prompt": "<USER>What is bioluminescence</USER><THINK>", "expected_call": 'search(query="bioluminescence")'},
    {"domain": "OOD Film/Term", "prompt": "<USER>What is aerogel</USER><THINK>", "expected_call": 'search(query="aerogel")'},
    {"domain": "OOD Economics", "prompt": "<USER>Define stagflation</USER><THINK>", "expected_call": 'search(query="stagflation")'},
    {"domain": "OOD Cuisine", "prompt": "<USER>What is ceviche?</USER><THINK>", "expected_call": 'search(query="ceviche")'},
    {"domain": "OOD Astronomy", "prompt": "<USER>What are quasars</USER><THINK>", "expected_call": 'search(query="quasars")'},
    {"domain": "OOD Weather (City)", "prompt": "<USER>How is the weather in Tbilisi</USER><THINK>", "expected_call": 'get_weather(city="Tbilisi")'},
    {"domain": "OOD Unit (Inverted)", "prompt": "<USER>How many km is 42 miles</USER><THINK>", "expected_call": 'unit_convert(value=42, from_unit="miles", to_unit="km")'},
    {"domain": "OOD Timezone", "prompt": "<USER>What time is it in WAT</USER><THINK>", "expected_call": 'get_time(timezone="WAT")'},
    {"domain": "OOD Philosophy", "prompt": "<USER>What is nihilism?</USER><THINK>", "expected_call": 'search(query="nihilism")'},
    {"domain": "OOD Mythology", "prompt": "<USER>Tell me about Yggdrasil</USER><THINK>", "expected_call": 'search(query="Yggdrasil")'},
    {"domain": "OOD Calculator", "prompt": "<USER>Compute 61 * 27</USER><THINK>", "expected_call_candidates": ['calculator(expr="61 * 27")', 'calculator(expr="61*27")']},
    {"domain": "OOD Deep Learning", "prompt": "<USER>What are convolutional networks</USER><THINK>", "expected_call": 'search(query="convolutional networks")'},
    {"domain": "OOD Music/Instrument", "prompt": "<USER>What is a harpsichord</USER><THINK>", "expected_call": 'search(query="harpsichord")'},
    {"domain": "OOD History (2-Word)", "prompt": "<USER>Who was Hatshepsut?</USER><THINK>", "expected_call": 'search(query="Hatshepsut")'},
    {"domain": "OOD Weather (2-Word)", "prompt": "<USER>What is the weather in Santo Domingo?</USER><THINK>", "expected_call": 'get_weather(city="Santo Domingo")'},
    {"domain": "OOD Weather (3-Word)", "prompt": "<USER>Weather on Ho Chi Minh City</USER><THINK>", "expected_call": 'get_weather(city="Ho Chi Minh City")'},
    {"domain": "OOD Bio (With Digit)", "prompt": "<USER>Explain crispr cas12</USER><THINK>", "expected_call": 'search(query="crispr cas12")'},
    {"domain": "OOD Architecture", "prompt": "<USER>Search for art nouveau</USER><THINK>", "expected_call": 'search(query="art nouveau")'},
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes 95% Wilson score confidence interval (percent)."""
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


# -----------------------------------------------------------------------------
# 2. Contamination Audit
# -----------------------------------------------------------------------------

def load_training_text() -> Tuple[str, int]:
    """Returns the champion SFT training corpus as one string and the max training operand length."""
    data = torch.load(os.path.join(ROOT_DIR, TRAIN_CACHE), map_location="cpu", weights_only=False)
    blob = "\n".join(str(x) for split in ("search", "math", "tools") for x in data[split])
    max_digits = 0
    for item in data["math"]:
        m = re.match(r"^(\d+)\+(\d+)=$", item["prompt"])
        if m:
            max_digits = max(max_digits, len(m.group(1)), len(m.group(2)))
    src = open(os.path.join(ROOT_DIR, DATASET_SRC), encoding="utf-8").read()
    return blob + "\n" + src, max_digits


def case_free_values(case: Dict[str, Any]) -> List[str]:
    """Free-text expected values of a case (closed-vocabulary schema keys excluded)."""
    values: List[str] = []
    for k, v in case.get("expected_args", {}).items():
        if k not in CLOSED_KEYS and isinstance(v, str):
            values.append(v)
    for tok in case.get("expected_match", []):
        if not tok.startswith("<"):
            values.append(tok)
    for call in case.get("expected_call_candidates", []) or [case.get("expected_call", "")]:
        for k, v in re.findall(r'(\w+)="([^"]+)"', call):
            if k not in CLOSED_KEYS:
                values.append(v)
    if "expected_arg" in case:
        values.append(case["expected_arg"])
    return values


def audit_contamination(corpus: str) -> List[str]:
    """Returns violation strings for every free-text expected value present in the training corpus."""
    suites = {
        "phrasing": PHRASING_26_CASES,
        "ood": OOD_18_CASES,
        "unseen": [{"expected_arg": c["expected_arg"]} for c in UNSEEN_OOD_30_CASES],
    }
    violations = []
    for name, cases in suites.items():
        for case in cases:
            for value in case_free_values(case):
                if value and value in corpus:
                    violations.append(f"{name}: '{value}' present in training corpus")
    return violations


# -----------------------------------------------------------------------------
# 3. Inference Suites
# -----------------------------------------------------------------------------

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
        resp = tokenizer.decode(out[0][len(toks):].tolist(), skip_special=False).strip()

        call_content = parse_call(resp)
        exp_tool = item["expected_tool"]
        route_ok = False
        args_ok = False

        if exp_tool == "no_tool":
            route_ok = ("<ANSWER>" in resp and "<CALL>" not in resp)
            args_ok = route_ok
        elif call_content is not None and call_content.startswith(f"{exp_tool}("):
            route_ok = True
            if "expected_args" in item:
                args_ok = all(
                    (f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}") in call_content
                    for k, v in item["expected_args"].items()
                )
            else:
                args_ok = any(m in call_content for m in item["expected_match"])

        routing_passed += int(route_ok)
        exact_args_passed += int(args_ok)

        status_tag = "[PASS]" if args_ok else ("[ROUTE_OK]" if route_ok else "[FAIL]")
        disp_resp = f"<CALL>{call_content}</CALL>" if call_content else resp[:50]
        print(f"  {status_tag:<10} {item['label']:<24} -> {disp_resp}", flush=True)

    route_ci = wilson_ci(routing_passed, total)
    args_ci = wilson_ci(exact_args_passed, total)
    print("-" * 80)
    print(f"[SUMMARY] Tool Routing:  {routing_passed}/{total} ({100.0 * routing_passed / total:.1f}%, 95% CI {route_ci}%)")
    print(f"[SUMMARY] Exact Args:    {exact_args_passed}/{total} ({100.0 * exact_args_passed / total:.1f}%, 95% CI {args_ci}%)")
    print("=" * 80 + "\n")

    return {
        "total": total,
        "routing_passed": routing_passed,
        "routing_ci": route_ci,
        "exact_args_passed": exact_args_passed,
        "exact_args_ci": args_ci,
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
        resp = tokenizer.decode(out[0][len(toks):].tolist(), skip_special=False).strip()

        call_content = parse_call(resp)
        candidate_calls = item.get("expected_call_candidates", [item.get("expected_call")])
        exp_tool = candidate_calls[0].split("(")[0]

        is_routed = call_content is not None and call_content.startswith(f"{exp_tool}(")
        is_exact = call_content is not None and call_content in candidate_calls

        routed += int(is_routed)
        exact_matches += int(is_exact)

        status_tag = "[EXACT]" if is_exact else ("[ROUTED]" if is_routed else "[DRIFT]")
        disp_resp = f"<CALL>{call_content}</CALL>" if call_content else resp[:50]
        print(f"  {status_tag:<8} {item['domain']:<22} -> {disp_resp}", flush=True)

    route_ci = wilson_ci(routed, total)
    exact_ci = wilson_ci(exact_matches, total)
    print("-" * 80)
    print(f"[SUMMARY] OOD Tool Selection:  {routed}/{total} ({100.0 * routed / total:.1f}%, 95% CI {route_ci}%)")
    print(f"[SUMMARY] OOD Headline Exact:  {exact_matches}/{total} ({100.0 * exact_matches / total:.1f}%, 95% CI {exact_ci}%)")
    print("=" * 80 + "\n")

    return {
        "total": total,
        "routed": routed,
        "routing_ci": route_ci,
        "exact_matches": exact_matches,
        "exact_ci": exact_ci,
    }


def make_arithmetic_pair(rng: random.Random, digits: int) -> Tuple[int, int]:
    """Samples a pair: 70% uniform random, 15% dense carry (digits 5-9), 15% cascade (9...9 + b)."""
    p_type = rng.random()
    if p_type < 0.15:
        return int("9" * digits), rng.randint(1, 9)
    if p_type < 0.30:
        a_str = "".join(str(rng.randint(5, 9)) for _ in range(digits))
        b_str = "".join(str(rng.randint(5, 9)) for _ in range(digits))
        return int(a_str), int(b_str)
    lo, hi = 10 ** (digits - 1), (10 ** digits) - 1
    return rng.randint(lo, hi), rng.randint(lo, hi)


@torch.no_grad()
def teacher_forced_match(model: ScaledGPT, tokenizer: ScaledTokenizer, device: torch.device, a: int, b: int) -> bool:
    """True iff greedy decoding reproduces the gold scratchpad + EOS exactly (argmax matches gold at every post-prompt position)."""
    prompt_len = len(tokenizer.encode(f"{a}+{b}=<THINK>", add_bos=True, add_eos=False))
    gold = tokenizer.encode(format_addition_char_scratchpad(a, b), add_bos=True, add_eos=True)
    if len(gold) > model.config.block_size:
        raise ValueError(f"Sequence length {len(gold)} exceeds block_size {model.config.block_size}")
    x = torch.tensor([gold[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x)
    pred = logits[0].argmax(dim=-1)
    target = torch.tensor(gold[1:], dtype=torch.long, device=device)
    return bool((pred[prompt_len - 1:] == target[prompt_len - 1:]).all().item())


@torch.no_grad()
def evaluate_arithmetic_suite(
    model: ScaledGPT,
    tokenizer: ScaledTokenizer,
    device: torch.device,
    digits_list: List[int],
    samples_per_length: int,
    seed: int,
    mode: str,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"SUITE 3: RANDOM MULTI-CARRY ARITHMETIC ({digits_list} digits, mode={mode})")
    print(f"Samples per length: {samples_per_length} | Seed: {seed} | Seen <= {max(ARITH_SEEN_DIGITS)} digits")
    print("=" * 80)

    per_length: Dict[str, Any] = {}
    for digits in digits_list:
        rng = random.Random(seed + digits)
        exact = 0
        think_exact = 0
        gen_tokens = 0
        t0 = time.time()
        for _ in range(samples_per_length):
            a, b = make_arithmetic_pair(rng, digits)
            if mode == "teacher_forced":
                ok = teacher_forced_match(model, tokenizer, device, a, b)
                exact += int(ok)
                think_exact += int(ok)
                continue

            prompt = f"{a}+{b}=<THINK>"
            toks = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            x = torch.tensor([toks], dtype=torch.long, device=device)
            max_gen = (digits + 2) * 15 + 40
            out = model.generate(x, max_new_tokens=max_gen, temperature=0.0, stop_tokens=[tokenizer.eos_id])
            gen_tokens += out.shape[1] - len(toks)
            gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)

            _, pred_ans = extract_scratchpad_answer(gen_text)
            expected_think = format_addition_char_scratchpad(a, b).split("</THINK>")[0].replace("<THINK>", "")
            exact += int(pred_ans == str(a + b))
            think_exact += int(expected_think in gen_text)

        elapsed = time.time() - t0
        ci = wilson_ci(exact, samples_per_length)
        think_ci = wilson_ci(think_exact, samples_per_length)
        seen = digits <= max(ARITH_SEEN_DIGITS)
        status = "[PASS]" if exact == samples_per_length else "[FAIL]"
        print(
            f"  {status} {digits:2d} Digits ({'seen' if seen else 'unseen'}): "
            f"exact {exact:2d}/{samples_per_length} ({100.0 * exact / samples_per_length:5.1f}%, CI {ci}%) | "
            f"think {think_exact:2d}/{samples_per_length} | {elapsed:7.1f}s",
            flush=True,
        )
        per_length[str(digits)] = {
            "digits": digits,
            "mode": mode,
            "seen_in_training": seen,
            "n": samples_per_length,
            "exact": exact,
            "exact_ci": ci,
            "think_exact": think_exact,
            "think_ci": think_ci,
            "gen_tokens": gen_tokens,
            "elapsed_sec": round(elapsed, 2),
        }

    summary = {}
    for label, flag in (("unseen", False), ("seen", True)):
        rows = [v for v in per_length.values() if v["seen_in_training"] == flag]
        if not rows:
            continue
        n = sum(v["n"] for v in rows)
        k = sum(v["exact"] for v in rows)
        summary[label] = {"n": n, "exact": k, "exact_ci": wilson_ci(k, n)}
        print(f"[SUMMARY] {label.capitalize()} lengths: {k}/{n} ({100.0 * k / n:.2f}%, CI {summary[label]['exact_ci']}%)")
    print("=" * 80 + "\n")

    return {"per_length": per_length, "summary": summary}


# -----------------------------------------------------------------------------
# 4. Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Contamination-Audited Evaluation for Scaled CBT Champion")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to checkpoint (.pt)")
    parser.add_argument("--suite", type=str, default="all", choices=["all", "phrasing", "ood", "arithmetic"])
    parser.add_argument("--samples-per-length", type=int, default=30)
    parser.add_argument(
        "--arith-mode",
        type=str,
        default="teacher_forced",
        choices=["teacher_forced", "free_run"],
        help="teacher_forced: one forward pass per problem (exact greedy-equivalent); free_run: autoregressive generation (no KV cache, slow)",
    )
    parser.add_argument("--arith-digits", type=str, default=",".join(map(str, ARITH_SEEN_DIGITS + ARITH_UNSEEN_DIGITS)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="", help="Inference device: cuda or cpu")
    parser.add_argument("--out-json", type=str, default="", help="Optional output JSON path")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"SCALED CBT CHAMPION EVALUATION | Device: {device}")
    print("=" * 80)

    corpus, max_train_digits = load_training_text()
    violations = audit_contamination(corpus)
    if violations:
        for v in violations:
            print(f"[FAIL] {v}")
        print(f"[FAIL] Contamination audit failed with {len(violations)} violation(s); aborting.")
        sys.exit(2)
    print("[PASS] Contamination audit: 0 free-text overlaps with training corpus.")
    print(f"[INFO] Max training operand digits in SFT math replay: {max_train_digits}")
    if max(ARITH_SEEN_DIGITS) > max_train_digits:
        print(f"[WARN] ARITH_SEEN_DIGITS exceeds training ceiling {max_train_digits}")

    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    tokenizer = ScaledTokenizer()
    ckpt_data = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt_data.get("config", ScaledGPTConfig(vocab_size=tokenizer.vocab_size))
    model = ScaledGPT(config, tokenizer=tokenizer).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    print("[PASS] Model successfully initialized and loaded.\n")

    results: Dict[str, Any] = {"checkpoint": args.ckpt, "max_train_operand_digits": max_train_digits}
    if args.suite in ["all", "phrasing"]:
        results["phrasing"] = evaluate_phrasing_suite(model, tokenizer, device)
    if args.suite in ["all", "ood"]:
        results["ood"] = evaluate_ood_suite(model, tokenizer, device)
    if args.suite in ["all", "arithmetic"]:
        digits_list = [int(d) for d in args.arith_digits.split(",") if d.strip()]
        results["arithmetic"] = evaluate_arithmetic_suite(
            model, tokenizer, device, digits_list, args.samples_per_length, args.seed, args.arith_mode
        )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved results to {args.out_json}")

    print("=" * 80)
    print("ALL EVALUATIONS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
