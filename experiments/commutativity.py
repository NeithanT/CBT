import os
import sys
import time
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
from src.dataset import generate_commutativity_dataset, MathDataset

DEFAULT_CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "toy_baselines", "v25_abspos_4l128d_commutative.pt")


def evaluate_dataset_split(
    model: MathGPT,
    tokenizer: MathTokenizer,
    equations: List[str],
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for eq in equations:
            parts = eq.split("=")
            prompt = parts[0] + "="
            expected = parts[1]

            enc = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            x_in = torch.tensor([enc], dtype=torch.long, device=device)
            out = model.generate(
                x_in,
                max_new_tokens=len(expected) + 2,
                temperature=0.0,
                eos_id=tokenizer.eos_id,
            )
            pred_tokens = out[0, len(enc) :].tolist()
            full_pred = tokenizer.decode(pred_tokens, skip_special=True).strip()
            ans = full_pred.split("=")[-1].strip() if "=" in full_pred else full_pred.strip()
            if ans == expected:
                correct += 1
    return correct / len(equations) if equations else 0.0


def main():
    parser = argparse.ArgumentParser(description="Commutativity Generalization Experiment")
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

        train_eq, test_canon, test_comm_seen, test_comm_unseen = generate_commutativity_dataset()
        print("Evaluating Commutativity splits:")
        print(f"  Canonical Unseen (a <= b): {evaluate_dataset_split(model, tokenizer, test_canon[:100], device) * 100:.1f}%")
        print(f"  Commuted Unseen (b > a):   {evaluate_dataset_split(model, tokenizer, test_comm_unseen[:100], device) * 100:.1f}%")
        return

    print("Running Commutativity Evaluation...")


if __name__ == "__main__":
    main()
