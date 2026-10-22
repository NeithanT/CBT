"""
Sample Efficiency Benchmark for HRM-RoPE (Decoupled 2D-RoPE).
Trains HRM-RoPE from scratch on N in {10, 100, 1000} examples locally on CPU
to evaluate sample efficiency and algorithmic induction speed.
"""

import os
import sys
import time
import json
import csv
import random
import argparse
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.tokenizer import MathTokenizer
from src.models.hrm_rope import HRMRoPE, HRMRoPEConfig
from src.arithmetic import format_addition_char_scratchpad, extract_scratchpad_answer
from src.dataset import get_or_generate_shared_math_pairs
from training.trainer import encode_texts_to_batch
from champion_model.math_model.train_and_eval import EXTRAPOLATION_BENCHMARK


def load_sample_efficiency_datasets(
    subset_sizes: List[int],
    seed: int = 42,
) -> Dict[int, List[str]]:
    """
    Extracts nested subsets of training examples from data/shared_math_pairs.pt.
    Ensures S_10 is a subset of S_100 is a subset of S_1000.
    """
    shared_path = os.path.join(ROOT_DIR, "data", "shared_math_pairs.pt")
    if os.path.exists(shared_path):
        data = torch.load(shared_path, map_location="cpu")
        a_tensor, b_tensor = data["a"], data["b"]
    else:
        a_tensor, b_tensor = get_or_generate_shared_math_pairs(ROOT_DIR)

    max_req = max(subset_sizes)
    assert len(a_tensor) >= max_req, f"Requested {max_req} samples but dataset has {len(a_tensor)}"

    # Generate canonical scratchpad texts deterministically
    all_texts: List[str] = []
    for i in range(max_req):
        a_val = int(a_tensor[i].item() if isinstance(a_tensor[i], torch.Tensor) else a_tensor[i])
        b_val = int(b_tensor[i].item() if isinstance(b_tensor[i], torch.Tensor) else b_tensor[i])
        all_texts.append(format_addition_char_scratchpad(a_val, b_val))

    datasets: Dict[int, List[str]] = {}
    for n in subset_sizes:
        datasets[n] = all_texts[:n]

    return datasets


def train_hrm_rope_sample_subset(
    model: HRMRoPE,
    tokenizer: MathTokenizer,
    train_texts: List[str],
    device: torch.device,
    steps: int = 300,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    block_size: int = 256,
    log_interval: int = 25,
    checkpoint_path: Optional[str] = None,
) -> List[float]:
    """
    Trains HRMRoPE on a specified subset of examples with AdamW and cosine decay.
    """
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
    for step in range(1, steps + 1):
        if curr_idx + effective_bs > n_samples:
            epoch_seed += 1
            rng = random.Random(epoch_seed)
            rng.shuffle(indices)
            curr_idx = 0

        batch_indices = indices[curr_idx : curr_idx + effective_bs]
        curr_idx += effective_bs
        batch_texts = [train_texts[i] for i in batch_indices]

        x, y = encode_texts_to_batch(
            texts=batch_texts,
            tokenizer=tokenizer,
            block_size=block_size,
            device=device,
            mask_prompt=True,
        )

        optimizer.zero_grad()
        _, loss = model(x, targets=y, tokenizer=tokenizer)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if step % log_interval == 0 or step == steps:
            elapsed = time.time() - t0
            lr_curr = optimizer.param_groups[0]["lr"]
            print(
                f"[INFO] Step {step:4d}/{steps:4d} | "
                f"Loss: {loss_val:.4f} | "
                f"LR: {lr_curr:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model.config,
                "steps": steps,
                "n_samples": n_samples,
                "final_loss": loss_history[-1] if loss_history else 0.0,
            },
            checkpoint_path,
        )
        print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    return loss_history


def evaluate_model_on_cases(
    model: HRMRoPE,
    tokenizer: MathTokenizer,
    cases: List[Tuple[str, str]],
    device: torch.device,
    max_new_tokens: int = 512,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates model on given equation test cases (prompt, expected_answer).
    """
    model.eval()
    correct = 0
    results: List[Dict[str, Any]] = []

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
                tokenizer=tokenizer,
            )
        elapsed = time.time() - t_start

        gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
        reasoning, answer = extract_scratchpad_answer(gen_text)
        is_pass = (answer.strip() == expected.strip())
        if is_pass:
            correct += 1

        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": answer,
            "reasoning": reasoning,
            "is_correct": is_pass,
            "latency": elapsed,
        })

    accuracy = (correct / len(cases)) * 100.0 if cases else 0.0
    return accuracy, results


def run_sample_efficiency_experiment(
    subset_sizes: List[int] = [10, 100, 1000],
    steps_map: Optional[Dict[int, int]] = None,
    batch_size_map: Optional[Dict[int, int]] = None,
    lr: float = 5e-4,
    output_dir: str = "results/sample_efficiency",
    checkpoint_dir: str = "checkpoints/sample_efficiency",
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """
    Executes full sample efficiency benchmarking pipeline across specified subset sizes.
    """
    device = torch.device(device_name)
    tokenizer = MathTokenizer()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if steps_map is None:
        steps_map = {10: 300, 100: 500, 1000: 800}
    if batch_size_map is None:
        batch_size_map = {10: 10, 100: 25, 1000: 50}

    print("=" * 80)
    print("HRM-RoPE SAMPLE EFFICIENCY EXPERIMENT")
    print(f"Device: {device} | Subsets: {subset_sizes}")
    print("=" * 80)

    datasets = load_sample_efficiency_datasets(subset_sizes, seed=42)

    # Standard extrapolation test cases
    standard_categories = EXTRAPOLATION_BENCHMARK

    summary_records: List[Dict[str, Any]] = []
    detailed_results: Dict[str, Any] = {}

    for n in subset_sizes:
        print("\n" + "#" * 60)
        print(f"[CONDITION] Training with N = {n} examples")
        print("#" * 60)

        # Initialize fresh model from scratch
        torch.manual_seed(42 + n)
        cfg = HRMRoPEConfig(vocab_size=tokenizer.vocab_size)
        model = HRMRoPE(cfg).to(device)

        ckpt_path = os.path.join(checkpoint_dir, f"hrm_rope_{n}_examples.pt")
        train_texts = datasets[n]
        n_steps = steps_map.get(n, 400)
        b_size = batch_size_map.get(n, 16)

        t_train_start = time.time()
        losses = train_hrm_rope_sample_subset(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            device=device,
            steps=n_steps,
            batch_size=b_size,
            lr=lr,
            checkpoint_path=ckpt_path,
        )
        train_duration = time.time() - t_train_start

        # 1. Evaluate training set fit (memorization check)
        eval_train_count = min(n, 20)
        train_cases = []
        for txt in train_texts[:eval_train_count]:
            prompt = txt.split("<THINK>")[0]
            expected = txt.split("=")[-1]
            train_cases.append((prompt, expected))

        train_acc, train_eval_details = evaluate_model_on_cases(
            model, tokenizer, train_cases, device, max_new_tokens=256
        )

        # 2. Evaluate on standardized benchmark categories
        cat_accuracies: Dict[str, float] = {}
        cat_details: Dict[str, Any] = {}
        total_eval_cases = 0
        total_eval_passed = 0

        for cat_name, cases in standard_categories.items():
            acc, details = evaluate_model_on_cases(model, tokenizer, cases, device, max_new_tokens=512)
            cat_accuracies[cat_name] = acc
            cat_details[cat_name] = details
            passed_count = sum(1 for d in details if d["is_correct"])
            total_eval_passed += passed_count
            total_eval_cases += len(cases)
            print(f"  [{cat_name}] Accuracy: {acc:.1f}% ({passed_count}/{len(cases)})")

        overall_eval_acc = (total_eval_passed / total_eval_cases) * 100.0 if total_eval_cases else 0.0

        summary = {
            "n_samples": n,
            "steps": n_steps,
            "train_duration_sec": round(train_duration, 1),
            "final_train_loss": round(losses[-1], 4) if losses else None,
            "train_fit_acc": round(train_acc, 2),
            "in_distribution_acc": round(cat_accuracies.get("In-Distribution (1-12 Digits - Trained)", 0.0), 2),
            "moderate_extrap_acc": round(cat_accuracies.get("Moderate Extrapolation (14 & 16 Digits)", 0.0), 2),
            "extreme_extrap_acc": round(cat_accuracies.get("Extreme Extrapolation (18 & 20 Digits)", 0.0), 2),
            "cascade_carry_acc": round(cat_accuracies.get("20-Digit Zero Counting & Cascade Carry", 0.0), 2),
            "overall_test_acc": round(overall_eval_acc, 2),
        }
        summary_records.append(summary)

        detailed_results[f"n_{n}"] = {
            "summary": summary,
            "loss_history": losses,
            "train_eval": train_eval_details,
            "benchmark_eval": cat_details,
        }

    # Save summary CSV
    csv_path = os.path.join(output_dir, "sample_efficiency_summary.csv")
    if summary_records:
        keys = list(summary_records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_records)
        print(f"\n[INFO] Summary CSV written to {csv_path}")

    # Save detailed JSON
    json_path = os.path.join(output_dir, "sample_efficiency_detailed.json")
    with open(json_path, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"[INFO] Detailed JSON written to {json_path}")

    # Print final console summary table
    print("\n" + "=" * 90)
    print("SAMPLE EFFICIENCY BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'N':<6}{'Steps':<8}{'Duration':<12}{'Loss':<10}{'Train Fit':<12}{'In-Dist':<10}{'Extrap-16':<12}{'Extrap-20':<12}{'Cascade':<10}")
    print("-" * 90)
    for r in summary_records:
        print(
            f"{r['n_samples']:<6}"
            f"{r['steps']:<8}"
            f"{r['train_duration_sec']:<12.1f}"
            f"{r['final_train_loss']:<10.4f}"
            f"{r['train_fit_acc']:<12.1f}"
            f"{r['in_distribution_acc']:<10.1f}"
            f"{r['moderate_extrap_acc']:<12.1f}"
            f"{r['extreme_extrap_acc']:<12.1f}"
            f"{r['cascade_carry_acc']:<10.1f}"
        )
    print("=" * 90)

    return {"summary": summary_records, "details": detailed_results}


def main():
    parser = argparse.ArgumentParser(description="HRM-RoPE Sample Efficiency Experiment Runner")
    parser.add_argument("--subsets", nargs="+", type=int, default=[10, 100, 1000], help="Subset sizes")
    parser.add_argument("--steps-10", type=int, default=300, help="Steps for 10 samples")
    parser.add_argument("--steps-100", type=int, default=500, help="Steps for 100 samples")
    parser.add_argument("--steps-1000", type=int, default=800, help="Steps for 1000 samples")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda:0)")
    parser.add_argument("--output-dir", type=str, default="results/sample_efficiency", help="Output directory")
    args = parser.parse_args()

    steps_map = {10: args.steps_10, 100: args.steps_100, 1000: args.steps_1000}
    run_sample_efficiency_experiment(
        subset_sizes=args.subsets,
        steps_map=steps_map,
        lr=args.lr,
        output_dir=args.output_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
