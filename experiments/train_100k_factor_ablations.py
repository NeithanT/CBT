"""
Mechanistic Factor Ablation Runner for ~100k Parameter Models.
Isolates and evaluates each architectural factor that made ScaledGPT successful:
1. Positional Encoding Geometry: 1D-RoPE vs Decoupled 2D-RoPE
2. Recurrent Unrolling Depth: R=1 (Dense depth 2) vs R=3 (Recurrent depth 6)
3. Coordinate Shift Augmentation: k_rand_shift = 0 vs 16
4. Interaction Effects: 1D + R=3 vs 2D + R=1 vs 2D + R=3
"""

import os
import sys
import time
import json
import csv
import random
import argparse
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.math_model.model import MathGPT, GPTConfig
from src.tokenizer import MathTokenizer
from src.arithmetic import extract_scratchpad_answer
from champion_model.math_model.train_and_eval import (
    EXTRAPOLATION_BENCHMARK,
    load_training_equations,
    encode_batch,
)


def train_single_model(
    model: MathGPT,
    tokenizer: MathTokenizer,
    train_texts: List[str],
    device: torch.device,
    steps: int = 1500,
    batch_size: int = 16,
    lr: float = 1.2e-3,
    weight_decay: float = 0.01,
    block_size: int = 256,
    log_interval: int = 100,
) -> Tuple[List[float], float]:
    """Trains a model with AdamW, cosine annealing, and dynamic batch padding."""
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

        raw_lengths = [len(tokenizer.encode(t, add_bos=True, add_eos=True)) for t in batch_texts]
        batch_len = min(max(raw_lengths) - 1, block_size)

        x, y = encode_batch(
            texts=batch_texts,
            tokenizer=tokenizer,
            block_size=batch_len,
            device=device,
            mask_prompt=True,
        )

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
                f"Elapsed: {elapsed:.1f}s",
                flush=True,
            )

    train_time = time.time() - t0
    return loss_history, train_time


def evaluate_model(
    model: MathGPT,
    tokenizer: MathTokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Runs the 14-equation length extrapolation benchmark suite."""
    model.eval()
    results = {}
    total_cases = 0
    total_passed = 0

    tier_scores = {}

    for tier_name, cases in EXTRAPOLATION_BENCHMARK.items():
        tier_passed = 0
        tier_cases_count = len(cases)
        for prompt, expected in cases:
            eq_prompt = prompt if prompt.endswith("<THINK>") else (prompt + "<THINK>")
            tokens = tokenizer.encode(eq_prompt, add_bos=True, add_eos=False)
            x = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                out = model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    eos_id=tokenizer.eos_id,
                )

            gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
            _, answer = extract_scratchpad_answer(gen_text)
            is_pass = (answer.strip() == expected.strip())
            if is_pass:
                tier_passed += 1
                total_passed += 1
            total_cases += 1

        tier_accuracy = (tier_passed / tier_cases_count) * 100.0 if tier_cases_count > 0 else 0.0
        tier_scores[tier_name] = {
            "passed": tier_passed,
            "total": tier_cases_count,
            "accuracy": tier_accuracy,
        }

    overall_accuracy = (total_passed / total_cases) * 100.0 if total_cases > 0 else 0.0
    return {
        "tier_scores": tier_scores,
        "total_passed": total_passed,
        "total_cases": total_cases,
        "overall_accuracy": overall_accuracy,
    }


CKPT_NAMES = {
    "model_1d_dense": "1d_rope_R1_k0",
    "model_1d_rec3": "1d_rope_R3_k0",
    "model_2d_dense": "2d_rope_R1_k0",
    "model_2d_rec3": "2d_rope_R3_k0",
    "model_2d_rec3_shift16": "2d_rope_R3_k16",
}


def run_factor_ablation_suite(
    steps: int = 1500,
    batch_size: int = 16,
    lr: float = 1.2e-3,
    n_samples: int = 2500,
    threads: int = 12,
    save_dir: str = "checkpoints/ablations_100k/short_1500steps",
    results_dir: str = "results",
) -> List[Dict[str, Any]]:
    """Executes the full factor ablation matrix for ~100k parameter models."""
    os.makedirs(os.path.join(ROOT_DIR, save_dir), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, results_dir), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(threads)

    print(f"[INFO] Initializing factor ablation suite on {device} (threads={threads})", flush=True)
    tokenizer = MathTokenizer()

    print(f"[INFO] Generating deterministic curriculum dataset ({n_samples} equations)...", flush=True)
    train_texts = load_training_equations(n_samples=n_samples, root_dir=ROOT_DIR, seed=42, curriculum=True)

    ablation_matrix = [
        {
            "id": "model_1d_dense",
            "name": "Model 1: 1D-RoPE Baseline (R=1, Shift=0)",
            "pos_emb_type": "rope",
            "n_recur": 1,
            "k_rand_shift": 0,
            "hypothesis": "Standard 1D positional encoding; suffers phase drift on lengths > 12 digits.",
        },
        {
            "id": "model_1d_rec3",
            "name": "Model 2: 1D-RoPE + Recurrence (R=3, Shift=0)",
            "pos_emb_type": "rope",
            "n_recur": 3,
            "k_rand_shift": 0,
            "hypothesis": "Adds recurrent depth without 2D geometry; tests if depth alone fixes 1D drift.",
        },
        {
            "id": "model_2d_dense",
            "name": "Model 3: 2D-RoPE Only (R=1, Shift=0)",
            "pos_emb_type": "rope_2d",
            "n_recur": 1,
            "k_rand_shift": 0,
            "hypothesis": "Decoupled 2D-RoPE without recurrence; aligns columns but lacks multi-step unrolling.",
        },
        {
            "id": "model_2d_rec3",
            "name": "Model 4: 2D-RoPE + Recurrence (R=3, Shift=0)",
            "pos_emb_type": "rope_2d",
            "n_recur": 3,
            "k_rand_shift": 0,
            "hypothesis": "Decoupled 2D-RoPE + R=3; tests synergistic carry propagation without coordinate jitter.",
        },
        {
            "id": "model_2d_rec3_shift16",
            "name": "Model 5: ScaledGPT Champion (2D-RoPE + R=3 + Shift=16)",
            "pos_emb_type": "rope_2d",
            "n_recur": 3,
            "k_rand_shift": 16,
            "hypothesis": "Full ScaledGPT inductive biases; enforces shift invariance + column alignment + depth.",
        },
    ]

    leaderboard = []

    for idx, config_meta in enumerate(ablation_matrix, start=1):
        print("\n" + "=" * 76, flush=True)
        print(f"[{idx}/{len(ablation_matrix)}] TRAINING: {config_meta['name']}", flush=True)
        print(f"  Positional: {config_meta['pos_emb_type']} | R: {config_meta['n_recur']} | Shift: {config_meta['k_rand_shift']}", flush=True)
        print(f"  Hypothesis: {config_meta['hypothesis']}", flush=True)
        print("=" * 76, flush=True)

        config = GPTConfig.create_100k(
            vocab_size=tokenizer.vocab_size,
            block_size=256,
            pos_emb_type=config_meta["pos_emb_type"],
            n_recur=config_meta["n_recur"],
            k_rand_shift=config_meta["k_rand_shift"],
        )
        torch.manual_seed(42)
        model = MathGPT(config).to(device)

        n_params = model.get_num_params(non_embedding=False)
        non_emb = model.get_num_params(non_embedding=True)
        print(f"[INFO] Parameters: Total={n_params:,} | Non-Embedding={non_emb:,}", flush=True)

        loss_hist, train_time = train_single_model(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            device=device,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            log_interval=250,
        )

        final_loss = loss_hist[-1]
        ckpt_filename = f"{CKPT_NAMES[config_meta['id']]}.pt"
        ckpt_path = os.path.join(ROOT_DIR, save_dir, ckpt_filename)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "final_loss": final_loss,
                "steps": steps,
                "meta": config_meta,
            },
            ckpt_path,
        )
        print(f"[INFO] Checkpoint saved: {ckpt_path}", flush=True)

        print(f"[INFO] Evaluating benchmark across 1-20 digits...", flush=True)
        eval_metrics = evaluate_model(model, tokenizer, device=device)

        record = {
            "id": config_meta["id"],
            "name": config_meta["name"],
            "pos_emb_type": config_meta["pos_emb_type"],
            "n_recur": config_meta["n_recur"],
            "k_rand_shift": config_meta["k_rand_shift"],
            "total_params": n_params,
            "non_embedding_params": non_emb,
            "train_time_sec": round(train_time, 2),
            "final_loss": round(final_loss, 5),
            "in_distribution": eval_metrics["tier_scores"]["In-Distribution (1-12 Digits)"]["accuracy"],
            "mod_extrapolation": eval_metrics["tier_scores"]["Moderate Extrapolation (14 & 16 Digits)"]["accuracy"],
            "ext_extrapolation": eval_metrics["tier_scores"]["Extreme Extrapolation (18 & 20 Digits)"]["accuracy"],
            "cascade_carry": eval_metrics["tier_scores"]["20-Digit Zero Counting & Cascade Carry"]["accuracy"],
            "overall_accuracy": eval_metrics["overall_accuracy"],
            "passed_cases": eval_metrics["total_passed"],
            "total_cases": eval_metrics["total_cases"],
        }
        leaderboard.append(record)

        print(f"[RESULT] Overall Accuracy: {record['overall_accuracy']:.1f}% ({record['passed_cases']}/{record['total_cases']})", flush=True)
        print(f"         In-Dist: {record['in_distribution']:.1f}% | Mod (14-16d): {record['mod_extrapolation']:.1f}% | Ext (18-20d): {record['ext_extrapolation']:.1f}% | Cascade: {record['cascade_carry']:.1f}%", flush=True)

    # Save outputs
    suffix = "_full" if "full" in save_dir or steps >= 2000 else ""
    json_path = os.path.join(ROOT_DIR, results_dir, f"factor_ablation_100k{suffix}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    csv_path = os.path.join(ROOT_DIR, results_dir, f"factor_ablation_100k{suffix}_leaderboard.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard)

    print("\n" + "=" * 76, flush=True)
    print("FACTOR ABLATION LEADERBOARD SUMMARY (~100k PARAMETERS)", flush=True)
    print("=" * 76, flush=True)
    print(f"{'Model':<42} | {'Pos':<7} | {'R':<2} | {'Shift':<5} | {'Loss':<7} | {'Overall':<7}", flush=True)
    print("-" * 76, flush=True)
    for r in leaderboard:
        print(f"{r['name'][:42]:<42} | {r['pos_emb_type']:<7} | {r['n_recur']:<2} | {r['k_rand_shift']:<5} | {r['final_loss']:<7.4f} | {r['overall_accuracy']:<6.1f}%", flush=True)
    print("=" * 76, flush=True)
    print(f"[OK] Full ablation run complete. Metrics saved to {json_path} and {csv_path}", flush=True)
    return leaderboard


def main():
    parser = argparse.ArgumentParser(description="ScaledGPT Factor Ablation Suite (~100k Parameters)")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps per model (default: 1500)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Learning rate (default: 1.2e-3)")
    parser.add_argument("--n-samples", type=int, default=5000, help="Curriculum sample pool (default: 5000)")
    parser.add_argument("--threads", type=int, default=12, help="CPU threads to use (default: 12)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/ablations_100k/short_1500steps", help="Directory to save checkpoints (the 2,000-step paper run lives in checkpoints/ablations_100k)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    run_factor_ablation_suite(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_samples=args.n_samples,
        threads=args.threads,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
