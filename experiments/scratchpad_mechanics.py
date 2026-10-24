import os
import sys
import time
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.tokenizer import MathTokenizer
from src.model import MathGPT, GPTConfig
from src.constants import get_device
from src.arithmetic import format_addition_char_scratchpad, extract_scratchpad_answer

DEFAULT_CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "toy_baselines", "v25_1d_rope_scratchpad.pt")


def generate_gap_dataset(
    num_samples: int,
    seed: int = 42,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    rng = random.Random(seed)
    pairs = []
    equations = []
    for _ in range(num_samples):
        if rng.random() < 0.5:
            len_a = rng.randint(1, 4)
            len_b = rng.randint(1, 4)
        else:
            len_a = rng.randint(8, 11)
            len_b = rng.randint(8, 11)

        min_a = 0 if len_a == 1 else 10 ** (len_a - 1)
        max_a = 10 ** len_a - 1
        min_b = 0 if len_b == 1 else 10 ** (len_b - 1)
        max_b = 10 ** len_b - 1

        a = rng.randint(min_a, max_a)
        b = rng.randint(min_b, max_b)
        pairs.append((a, b))
        equations.append(format_addition_char_scratchpad(a, b))
    return equations, pairs


def evaluate_scratchpad_suite(
    model: MathGPT,
    tokenizer: MathTokenizer,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    model.eval()
    test_categories = {
        "Gap Test 1: 6-Digit Interpolation (Unseen)": [
            ("100000+100000=", "200000"),
            ("123456+654321=", "777777"),
            ("500000+500000=", "1000000"),
            ("999999+1=", "1000000"),
            ("345678+987654=", "1333332"),
            ("400000+600000=", "1000000"),
        ],
        "Gap Test 2: 7-Digit Zero Counting (Unseen)": [
            ("1000000+1000000=", "2000000"),
            ("2500000+2500000=", "5000000"),
            ("5000000+5000000=", "10000000"),
            ("9999999+1=", "10000000"),
        ],
        "In-Distribution 1: 1-4 Digits (Trained)": [
            ("15+25=", "40"),
            ("250+350=", "600"),
            ("789+456=", "1245"),
            ("999+999=", "1998"),
            ("1000+1000=", "2000"),
            ("4500+5500=", "10000"),
        ],
        "In-Distribution 2: 8-11 Digits (Trained)": [
            ("10000000+10000000=", "20000000"),
            ("100000000+100000000=", "200000000"),
            ("1000000000+1000000000=", "2000000000"),
            ("10000000000+10000000000=", "20000000000"),
            ("12345678901+98765432109=", "111111111010"),
        ],
    }

    results = {}
    with torch.no_grad():
        for category, items in test_categories.items():
            cat_correct = 0
            cat_details = []
            for prompt, expected in items:
                enc = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                x_in = torch.tensor([enc], dtype=torch.long, device=device)
                num_cols = len(expected) + 2
                max_tokens = num_cols * 11 + len(expected) + 10

                out = model.generate(
                    x_in,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    eos_id=tokenizer.eos_id,
                )
                pred_tokens = out[0, len(enc) :].tolist()
                full_pred = tokenizer.decode(pred_tokens, skip_special=True).strip()
                reasoning, final_ans = extract_scratchpad_answer(full_pred)

                is_correct = (final_ans == expected)
                if is_correct:
                    cat_correct += 1

                cat_details.append({
                    "prompt": prompt,
                    "expected": expected,
                    "reasoning": reasoning,
                    "final_answer": final_ans,
                    "full_pred": full_pred,
                    "correct": is_correct,
                })
            acc = cat_correct / len(items) if items else 0.0
            results[category] = {"accuracy": acc, "items": cat_details}
    return results


def main():
    parser = argparse.ArgumentParser(description="Scratchpad Arithmetic Experiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    device = get_device()
    tokenizer = MathTokenizer()

    if args.eval_only and os.path.exists(args.checkpoint):
        print(f"Loading {args.checkpoint}...")
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location=device)
        model = MathGPT(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        eval_res = evaluate_scratchpad_suite(model, tokenizer, device)
        for cat, data in eval_res.items():
            print(f"  {cat}: {data['accuracy'] * 100:.1f}%")
        return

    print("Running Scratchpad Evaluation...")


if __name__ == "__main__":
    main()
