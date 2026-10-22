"""
HRM-RoPE 100 Unseen Random Arithmetic Benchmark Suite.
Tests across 5 tiers:
1. In-Distribution (1 to 12 digits) - 20 cases
2. Moderate Extrapolation (13 to 16 digits) - 20 cases
3. Extreme Extrapolation (17 to 20 digits) - 20 cases
4. Super-Extreme Extrapolation (21 to 25 digits) - 20 cases
5. Ultra-Long & Adversarial Stress (26 to 30 digits, cascade 9s, repeated zeros) - 20 cases
"""

import os
import sys
import time
import json
import csv
import random
from typing import Dict, List, Tuple, Any

import torch

# Ensure repository root and working_model imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKING_MODEL_DIR = os.path.join(REPO_ROOT, "working_model")
sys.path.insert(0, WORKING_MODEL_DIR)
sys.path.insert(0, REPO_ROOT)

from working_model.model import HRMRoPE, HRMRoPEConfig
from working_model.tokenizer import MathTokenizer
from working_model.arithmetic import extract_scratchpad_answer


def generate_random_number(num_digits: int, rng: random.Random) -> int:
    """Generates an integer with exactly num_digits digits."""
    if num_digits <= 0:
        return 0
    if num_digits == 1:
        return rng.randint(0, 9)
    low = 10 ** (num_digits - 1)
    high = (10 ** num_digits) - 1
    return rng.randint(low, high)


def build_100_unseen_dataset(seed: int = 42) -> Dict[str, List[Tuple[str, str, int, int]]]:
    """
    Generates 100 unseen arithmetic cases divided into 5 tiers of 20 cases each.
    Returns: Dict[category_name, List[(prompt, expected_answer, len_a, len_b)]]
    """
    rng = random.Random(seed)
    dataset: Dict[str, List[Tuple[str, str, int, int]]] = {}

    # Tier 1: In-Distribution (1 to 12 digits) - 20 cases
    tier1 = []
    t1_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 5, 7, 8, 9, 10, 11, 12]
    for i, la in enumerate(t1_lengths):
        lb = rng.randint(1, la) if rng.random() > 0.3 else la
        a = generate_random_number(la, rng)
        b = generate_random_number(lb, rng)
        expected = str(a + b)
        tier1.append((f"{a}+{b}=", expected, la, lb))
    dataset["Tier 1: In-Distribution (1 to 12 Digits)"] = tier1

    # Tier 2: Moderate Extrapolation (13 to 16 digits) - 20 cases
    tier2 = []
    t2_lengths = [13, 14, 15, 16] * 5
    for la in t2_lengths:
        lb = la if rng.random() > 0.4 else rng.randint(10, la)
        a = generate_random_number(la, rng)
        b = generate_random_number(lb, rng)
        expected = str(a + b)
        tier2.append((f"{a}+{b}=", expected, la, lb))
    dataset["Tier 2: Moderate Extrapolation (13 to 16 Digits)"] = tier2

    # Tier 3: Extreme Extrapolation (17 to 20 digits) - 20 cases
    tier3 = []
    t3_lengths = [17, 18, 19, 20] * 5
    for la in t3_lengths:
        lb = la if rng.random() > 0.3 else rng.randint(14, la)
        a = generate_random_number(la, rng)
        b = generate_random_number(lb, rng)
        expected = str(a + b)
        tier3.append((f"{a}+{b}=", expected, la, lb))
    dataset["Tier 3: Extreme Extrapolation (17 to 20 Digits)"] = tier3

    # Tier 4: Super-Extreme Extrapolation (21 to 25 digits) - 20 cases
    tier4 = []
    t4_lengths = [21, 22, 23, 24, 25] * 4
    for la in t4_lengths:
        lb = la if rng.random() > 0.3 else rng.randint(18, la)
        a = generate_random_number(la, rng)
        b = generate_random_number(lb, rng)
        expected = str(a + b)
        tier4.append((f"{a}+{b}=", expected, la, lb))
    dataset["Tier 4: Super-Extreme Extrapolation (21 to 25 Digits)"] = tier4

    # Tier 5: Ultra-Long & Adversarial Stress (26 to 30 digits, cascade 9s, repeated zeros) - 20 cases
    tier5 = []
    # 8 random 26-30 digits cases
    for la in [26, 27, 28, 29, 30, 27, 28, 30]:
        lb = la if rng.random() > 0.3 else rng.randint(22, la)
        a = generate_random_number(la, rng)
        b = generate_random_number(lb, rng)
        expected = str(a + b)
        tier5.append((f"{a}+{b}=", expected, la, lb))

    # 6 cascade carry cases with strings of 9s (lengths 15, 18, 20, 22, 25, 28)
    for n9 in [15, 18, 20, 22, 25, 28]:
        a = int("9" * n9)
        b = rng.randint(1, 9)
        expected = str(a + b)
        tier5.append((f"{a}+{b}=", expected, n9, 1))

    # 6 repeated zero cases (lengths 16, 18, 20, 22, 24, 26)
    for nz in [16, 18, 20, 22, 24, 26]:
        da = rng.randint(1, 9)
        db = rng.randint(1, 9)
        a = da * (10 ** (nz - 1))
        b = db * (10 ** (nz - 1))
        expected = str(a + b)
        tier5.append((f"{a}+{b}=", expected, nz, nz))

    dataset["Tier 5: Ultra-Long & Adversarial Stress (26 to 30 Digits, 9s, Zeros)"] = tier5

    return dataset


def load_hrm_rope_model(checkpoint_path: str, device: torch.device) -> Tuple[HRMRoPE, MathTokenizer]:
    """Loads HRMRoPE model and tokenizer."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    tokenizer = MathTokenizer()
    cfg_raw = ckpt.get("config")
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
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    if isinstance(state, dict):
        model.load_state_dict(state, strict=True)
    model.eval()

    return model, tokenizer


def solve_equation(
    model: HRMRoPE,
    tokenizer: MathTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 1024,
) -> Tuple[str, str, float]:
    """Solves an equation prompt autoregressively."""
    full_prompt = prompt if prompt.endswith("<THINK>") else (prompt + "<THINK>")
    tokens = tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
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


def run_100_benchmark(checkpoint_path: str, output_dir: str, device_name: str = "cpu"):
    device = torch.device(device_name)
    print("=" * 80)
    print("HRM-RoPE RIGOROUS 100-TEST UNSEEN EXTRAPOLATION BENCHMARK")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")
    print("=" * 80)

    model, tokenizer = load_hrm_rope_model(checkpoint_path, device)
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"[INFO] HRM-RoPE Loaded successfully! Non-Embedding Parameters: {non_emb_params:,}")

    dataset = build_100_unseen_dataset(seed=42)
    total_tests = sum(len(v) for v in dataset.values())
    print(f"[INFO] Dataset generated: {total_tests} test cases across {len(dataset)} tiers.\n")

    detailed_results: List[Dict[str, Any]] = []
    tier_scores: Dict[str, Dict[str, Any]] = {}

    global_passed = 0
    global_total = 0
    test_idx = 0

    t_start = time.time()

    for tier_name, cases in dataset.items():
        print(f"\n{'=' * 80}")
        print(f"[{tier_name.upper()}] ({len(cases)} cases)")
        print(f"{'=' * 80}")

        tier_passed = 0
        tier_latencies = []

        for prompt, expected, la, lb in cases:
            test_idx += 1
            reasoning, answer, elapsed = solve_equation(model, tokenizer, prompt, device)
            is_correct = (answer == expected)
            tier_latencies.append(elapsed)

            if is_correct:
                tier_passed += 1
                global_passed += 1
            global_total += 1

            status = "[PASS]" if is_correct else "[FAIL]"
            max_digits = max(la, lb)

            disp_prompt = prompt if len(prompt) <= 35 else f"{prompt[:15]}...+{prompt.split('+')[1][:15]}="
            disp_ans = answer if len(answer) <= 25 else f"{answer[:12]}...{answer[-8:]}"
            disp_exp = expected if len(expected) <= 25 else f"{expected[:12]}...{expected[-8:]}"

            print(
                f"[{test_idx:03d}/100] {status} ({max_digits:02d}d) {disp_prompt:<35} "
                f"-> Ans: '{disp_ans}' (Exp: '{disp_exp}') [{elapsed:.2f}s]"
            )

            detailed_results.append({
                "test_id": test_idx,
                "tier": tier_name,
                "prompt": prompt,
                "expected": expected,
                "generated_answer": answer,
                "is_correct": is_correct,
                "digits_a": la,
                "digits_b": lb,
                "max_digits": max_digits,
                "latency_sec": round(elapsed, 3),
                "reasoning_steps": reasoning,
            })

        tier_acc = (tier_passed / len(cases)) * 100.0
        avg_lat = sum(tier_latencies) / len(tier_latencies)
        tier_scores[tier_name] = {
            "passed": tier_passed,
            "total": len(cases),
            "accuracy_pct": round(tier_acc, 2),
            "avg_latency_sec": round(avg_lat, 2),
        }
        print(f"\n--> {tier_name} Accuracy: {tier_acc:.2f}% ({tier_passed}/{len(cases)}) [Avg Latency: {avg_lat:.2f}s]")

    total_time = time.time() - t_start
    overall_acc = (global_passed / global_total) * 100.0

    extrap_passed = sum(tier_scores[t]["passed"] for t in tier_scores if "Tier 1" not in t)
    extrap_total = sum(tier_scores[t]["total"] for t in tier_scores if "Tier 1" not in t)
    extrap_acc = (extrap_passed / extrap_total) * 100.0

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY (100 UNSEEN RANDOM TESTS)")
    print("=" * 80)
    for t_name, score in tier_scores.items():
        print(f"  * {t_name:<65}: {score['accuracy_pct']:6.2f}% ({score['passed']:02d}/{score['total']:02d})")
    print("-" * 80)
    print(f"  * OVERALL BENCHMARK ACCURACY (1-30 DIGITS)        : {overall_acc:6.2f}% ({global_passed}/{global_total})")
    print(f"  * EXTRAPOLATION-ONLY ACCURACY (13-30 DIGITS)      : {extrap_acc:6.2f}% ({extrap_passed}/{extrap_total})")
    print(f"  * TOTAL BENCHMARK EXECUTION TIME                  : {total_time:.1f}s ({total_time / 60:.2f}m)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "hrm_rope_100_unseen_benchmark.json")
    csv_path = os.path.join(output_dir, "hrm_rope_100_unseen_benchmark.csv")

    summary_payload = {
        "benchmark_title": "HRM-RoPE Rigorous 100-Test Unseen Extrapolation Benchmark",
        "checkpoint": checkpoint_path,
        "device": device_name,
        "non_embedding_parameters": non_emb_params,
        "total_tests": global_total,
        "total_passed": global_passed,
        "overall_accuracy_pct": round(overall_acc, 2),
        "extrapolation_accuracy_pct": round(extrap_acc, 2),
        "total_time_sec": round(total_time, 2),
        "tier_scores": tier_scores,
        "detailed_results": detailed_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_id", "tier", "max_digits", "digits_a", "digits_b",
            "prompt", "expected", "generated_answer", "is_correct", "latency_sec"
        ])
        for row in detailed_results:
            writer.writerow([
                row["test_id"],
                row["tier"],
                row["max_digits"],
                row["digits_a"],
                row["digits_b"],
                row["prompt"],
                row["expected"],
                row["generated_answer"],
                row["is_correct"],
                row["latency_sec"]
            ])

    print(f"\n[OK] Results saved to:\n  JSON: {json_path}\n  CSV:  {csv_path}")
    return summary_payload


if __name__ == "__main__":
    ckpt = os.path.join(WORKING_MODEL_DIR, "checkpoints", "hrm_rope_c1_final.pt")
    out_dir = os.path.join(REPO_ROOT, "results", "benchmark_results")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_100_benchmark(checkpoint_path=ckpt, output_dir=out_dir, device_name=dev)
