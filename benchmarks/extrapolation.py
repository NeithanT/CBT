import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn

try:
    from src.model import MathGPT
    from src.tokenizer import MathTokenizer
    from src.bpe_tokenizer import BPEMathTokenizer
    from src.arithmetic import extract_scratchpad_answer
except ImportError:
    from model import MathGPT
    from tokenizer import MathTokenizer
    from bpe_tokenizer import BPEMathTokenizer
    from arithmetic import extract_scratchpad_answer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results", "benchmark_results")

EXTRAPOLATION_BENCHMARK_CATEGORIES: Dict[str, List[Tuple[str, str]]] = {
    "In-Distribution (1 to 12 Digits - Trained)": [
        ("15+25=", "40"),
        ("1234+5678=", "6912"),
        ("12345678+87654321=", "99999999"),
        ("10000000+10000000=", "20000000"),
        ("123456789012+876543210987=", "999999999999"),
        ("500000000000+500000000000=", "1000000000000"),
    ],
    "Moderate Extrapolation (14 & 16 Digits - Unseen)": [
        ("12345678901234+87654321098765=", "99999999999999"),
        ("50000000000000+50000000000000=", "100000000000000"),
        ("1111222233334444+8888777766665555=", "9999999999999999"),
    ],
    "Extreme Extrapolation (18 & 20 Digits - Unseen)": [
        ("100000000000000000+100000000000000000=", "200000000000000000"),
        ("12345678901234567890+87654321098765432109=", "99999999999999999999"),
        ("55555555555555555555+44444444444444444445=", "100000000000000000000"),
    ],
    "20-Digit Zero Counting & Cascade Carry (Unseen)": [
        ("10000000000000000000+10000000000000000000=", "20000000000000000000"),
        ("99999999999999999999+1=", "100000000000000000000"),
    ],
}


def save_benchmark_report(
    model_name: str,
    category_scores: Dict[str, float],
    detailed_results: List[Dict[str, Any]],
    output_dir: str = DEFAULT_RESULTS_DIR,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("/", "_")
    base_filename = f"{timestamp_str}_{slug}"

    json_path = os.path.join(output_dir, f"{base_filename}.json")
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    leaderboard_path = os.path.join(output_dir, "leaderboard.csv")

    report_payload = {
        "timestamp": timestamp_str,
        "model_name": model_name,
        "metadata": metadata or {},
        "category_scores": category_scores,
        "overall_accuracy": sum(category_scores.values()) / max(len(category_scores), 1),
        "detailed_results": detailed_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    fieldnames = ["category", "prompt", "expected", "generated_answer", "is_correct", "full_generation"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in detailed_results:
            writer.writerow({
                "category": item.get("category", ""),
                "prompt": item.get("prompt", ""),
                "expected": item.get("expected", ""),
                "generated_answer": item.get("generated_answer", ""),
                "is_correct": item.get("is_correct", False),
                "full_generation": item.get("full_generation", "").replace("\n", " "),
            })

    leaderboard_fields = [
        "timestamp",
        "model_name",
        "overall_acc",
        "in_distribution_acc",
        "moderate_extrap_acc",
        "extreme_extrap_acc",
        "zero_counting_acc",
        "json_log",
    ]
    write_header = not os.path.exists(leaderboard_path)
    with open(leaderboard_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=leaderboard_fields)
        if write_header:
            writer.writeheader()

        def _get_score(sub: str) -> str:
            for k, v in category_scores.items():
                if sub.lower() in k.lower():
                    return f"{v * 100:.1f}%"
            return "N/A"

        writer.writerow({
            "timestamp": timestamp_str,
            "model_name": model_name,
            "overall_acc": f"{(sum(category_scores.values()) / max(len(category_scores), 1)) * 100:.1f}%",
            "in_distribution_acc": _get_score("in-distribution"),
            "moderate_extrap_acc": _get_score("moderate"),
            "extreme_extrap_acc": _get_score("extreme"),
            "zero_counting_acc": _get_score("zero counting"),
            "json_log": os.path.basename(json_path),
        })

    return json_path, csv_path


def run_extrapolation_benchmark(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    model_name: str = "MathGPT-Model",
    is_scratchpad: bool = True,
    is_bpe: bool = False,
    max_new_tokens: int = 1024,
    save_results: bool = True,
    output_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    model.eval()
    category_scores: Dict[str, float] = {}
    detailed_results: List[Dict[str, Any]] = []

    print("-" * 75)
    print(f"EVALUATION SUITE: {model_name}")
    print(f"Device: {device} | Scratchpad: {is_scratchpad} | BPE: {is_bpe}")
    print("-" * 75)

    for cat_name, test_cases in EXTRAPOLATION_BENCHMARK_CATEGORIES.items():
        correct_count = 0
        total_count = len(test_cases)
        print(f"\n[{cat_name.upper()}] ({total_count} cases)")

        for prompt, expected in test_cases:
            eval_prompt = (prompt + "<THINK>") if (is_scratchpad and not prompt.endswith("<THINK>")) else prompt
            encoded = tokenizer.encode(eval_prompt, add_bos=True, add_eos=False)
            x_in = torch.tensor([encoded], dtype=torch.long, device=device)

            eos_token_id = tokenizer.eos_id
            start_t = time.time()
            with torch.no_grad():
                out_tokens = model.generate(
                    x_in,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    eos_id=eos_token_id,
                )
            elapsed = time.time() - start_t
            gen_text = tokenizer.decode(out_tokens[0].tolist(), skip_special=False)

            if is_scratchpad:
                reasoning, answer = extract_scratchpad_answer(gen_text)
            else:
                reasoning = ""
                parts = gen_text.split("=")
                answer = parts[-1].replace("<EOS>", "").strip() if len(parts) > 1 else ""

            is_correct = (answer == expected)
            if is_correct:
                correct_count += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"

            print(f"  {status} {prompt} -> Answer: '{answer}' (Expected: '{expected}') [{elapsed:.2f}s]")

            detailed_results.append({
                "category": cat_name,
                "prompt": prompt,
                "expected": expected,
                "generated_answer": answer,
                "reasoning_steps": reasoning,
                "full_generation": gen_text,
                "is_correct": is_correct,
                "elapsed_sec": round(elapsed, 3),
            })

        acc = correct_count / total_count if total_count > 0 else 0.0
        category_scores[cat_name] = acc
        print(f"  Category Accuracy: {acc * 100:.1f}% ({correct_count}/{total_count})")

    overall_acc = sum(category_scores.values()) / max(len(category_scores), 1)
    print("\n" + "=" * 75)
    print(f"BENCHMARK COMPLETED: Overall Accuracy = {overall_acc * 100:.2f}%")
    print("=" * 75)

    json_file, csv_file = "", ""
    if save_results:
        json_file, csv_file = save_benchmark_report(
            model_name=model_name,
            category_scores=category_scores,
            detailed_results=detailed_results,
            output_dir=output_dir,
            metadata={
                "is_scratchpad": is_scratchpad,
                "is_bpe": is_bpe,
                "max_new_tokens": max_new_tokens,
                "device": str(device),
            },
        )
        print(f"Results saved to:\n  JSON: {json_file}\n  CSV:  {csv_file}")

    return {
        "model_name": model_name,
        "category_scores": category_scores,
        "overall_accuracy": overall_acc,
        "json_path": json_file,
        "csv_path": csv_file,
    }


def evaluate_extrapolation_char(
    model: MathGPT,
    tokenizer: Any,
    device: torch.device,
    title: str = "MathGPT-Char-Model",
    save_results: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    res = run_extrapolation_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=title,
        is_scratchpad=True,
        is_bpe=False,
        save_results=save_results,
    )
    return res["category_scores"]


def evaluate_extrapolation_bpe(
    model: MathGPT,
    tokenizer: Any,
    device: torch.device,
    title: str = "MathGPT-BPE-Model",
    save_results: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    res = run_extrapolation_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=title,
        is_scratchpad=True,
        is_bpe=True,
        save_results=save_results,
    )
    return res["category_scores"]
