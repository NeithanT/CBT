import os
import sys
import glob
import argparse
from typing import Dict, List, Any, Optional
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.model import MathGPT, GPTConfig, create_model
from src.tokenizer import MathTokenizer
from src.bpe_tokenizer import BPEMathTokenizer
from src.constants import get_device
from benchmarks.extrapolation import run_extrapolation_benchmark


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = create_model(config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state_dict is None:
        raise KeyError(f"No state dict found in checkpoint {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()

    is_char = (config.vocab_size <= 64)
    tokenizer = MathTokenizer() if is_char else BPEMathTokenizer()
    model_type = getattr(config, "model_type", "standard").capitalize()
    if is_char:
        model_desc = f"{model_type} (Char)"
    else:
        model_desc = f"{model_type} (BPE Subword)"
    return model, tokenizer, config, model_desc



def benchmark_single_checkpoint(
    checkpoint_path: str, device: torch.device, label: Optional[str] = None
) -> Dict[str, Any]:
    model, tokenizer, config, model_type = load_model_from_checkpoint(checkpoint_path, device)
    model_label = label or os.path.basename(checkpoint_path).replace(".pt", "")
    n_params = model.get_num_params(non_embedding=True)

    print(f"\nEvaluating: {checkpoint_path}")
    print(f"Type: {model_type} | Params: {n_params:,} | Block size: {config.block_size}")

    return run_extrapolation_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_label,
        is_scratchpad=True,
        is_bpe=(not is_char),
        save_results=True,
    )


def find_all_checkpoints(base_dir: str = ROOT_DIR) -> List[str]:
    patterns = [
        os.path.join(base_dir, "checkpoints", "**", "*.pt"),
        os.path.join(base_dir, "checkpoints_*", "*.pt"),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    return sorted(list(set(found)))


def main():
    parser = argparse.ArgumentParser(description="Extrapolation Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--all", action="store_true", help="Benchmark all discovered checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda:0)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Benchmark Device: {device}")

    if args.checkpoint:
        benchmark_single_checkpoint(args.checkpoint, device)
    elif args.all:
        all_pts = find_all_checkpoints(ROOT_DIR)
        if not all_pts:
            print("No checkpoints found.")
            return
        print(f"Discovered {len(all_pts)} checkpoints:")
        for pt in all_pts:
            print(f"  {os.path.relpath(pt, ROOT_DIR)}")
        for pt in all_pts:
            try:
                benchmark_single_checkpoint(pt, device)
            except Exception as e:
                print(f"[ERROR] Failed {pt}: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
