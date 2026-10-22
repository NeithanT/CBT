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

DEFAULT_CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "baseline", "math_gpt_length_gen.pt")


def generate_length_dataset(
    min_digits: int = 1,
    max_digits: int = 4,
    num_samples: int = 10000,
    seed: int = 42,
) -> List[str]:
    rng = random.Random(seed)
    equations = set()
    attempts = 0
    max_attempts = num_samples * 4

    while len(equations) < num_samples and attempts < max_attempts:
        len_a = rng.randint(min_digits, max_digits)
        len_b = rng.randint(min_digits, max_digits)

        min_a = 0 if len_a == 1 else 10 ** (len_a - 1)
        max_a = 10 ** len_a - 1
        min_b = 0 if len_b == 1 else 10 ** (len_b - 1)
        max_b = 10 ** len_b - 1

        a = rng.randint(min_a, max_a)
        b = rng.randint(min_b, max_b)
        equations.add(f"{a}+{b}={a + b}")
        attempts += 1

    return list(equations)


def evaluate_length_generalization(
    model: MathGPT,
    tokenizer: MathTokenizer,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    test_scales = {
        "1-Digit (Trained)": [("3+5=", "8"), ("7+9=", "16"), ("0+0=", "0")],
        "2-Digit (Trained)": [("12+15=", "27"), ("48+52=", "100"), ("99+99=", "198")],
        "3-Digit (Trained)": [("123+456=", "579"), ("500+500=", "1000"), ("999+1=", "1000")],
        "4-Digit (Boundary)": [("1234+5678=", "6912"), ("5000+5000=", "10000"), ("9999+1=", "10000")],
        "5-Digit (OOD +1)": [("12345+67890=", "80235"), ("50000+50000=", "100000"), ("99999+1=", "100000")],
        "6-Digit (OOD +2)": [("123456+654321=", "777777"), ("500000+500000=", "1000000")],
        "8-Digit (OOD +4)": [("12345678+87654321=", "99999999"), ("50000000+50000000=", "100000000")],
    }

    results = {}
    with torch.no_grad():
        for scale_name, test_cases in test_scales.items():
            correct = 0
            for prompt, expected in test_cases:
                enc = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                x_in = torch.tensor([enc], dtype=torch.long, device=device)
                out = model.generate(
                    x_in,
                    max_new_tokens=len(expected) + 4,
                    temperature=0.0,
                    eos_id=tokenizer.eos_id,
                )
                pred_tokens = out[0, len(enc) :].tolist()
                full_pred = tokenizer.decode(pred_tokens, skip_special=True).strip()
                ans = full_pred.split("=")[-1].strip() if "=" in full_pred else full_pred.strip()
                if ans == expected:
                    correct += 1
            acc = correct / len(test_cases)
            results[scale_name] = acc
    return results


def main():
    parser = argparse.ArgumentParser(description="Length Generalization Experiment")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    device = get_device()
    tokenizer = MathTokenizer()

    if args.eval_only and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint {args.checkpoint}...")
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location=device)
        model = MathGPT(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        res = evaluate_length_generalization(model, tokenizer, device)
        for scale, acc in res.items():
            print(f"  {scale}: {acc * 100:.1f}%")
        return

    print("Running Length Generalization Evaluation...")


if __name__ == "__main__":
    main()
