#!/usr/bin/env python3
"""
Download Champion Weights from Hugging Face Hub.
Pulls 'champion_final.pt' from https://huggingface.co/Neithanvarvas/CBT
and stores it locally as 'checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sft.pt'.
"""

import os
import sys
import argparse

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[FAIL] huggingface_hub is not installed. Please run: pip install huggingface_hub")
    sys.exit(1)

DEFAULT_REPO = "Neithanvarvas/CBT"
DEFAULT_FILENAME = "champion_final.pt"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARGET_DIR = os.path.join(REPO_ROOT, "checkpoints", "cbt")
LOCAL_NAME = "cbt_18m_2d_rope_R3_k45_sft.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Download ScaledGPT Champion weights from Hugging Face")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO, help=f"Hugging Face repository ID (default: {DEFAULT_REPO})")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help=f"Filename to download (default: {DEFAULT_FILENAME})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_TARGET_DIR, help="Target directory for weights")
    return parser.parse_args()


def download_champion_weights(repo_id: str = DEFAULT_REPO, filename: str = DEFAULT_FILENAME, output_dir: str = DEFAULT_TARGET_DIR) -> str:
    """Downloads weights from Hugging Face and renames the champion file to its local name."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Pulling '{filename}' from Hugging Face (https://huggingface.co/{repo_id})...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
        )
        if filename == DEFAULT_FILENAME:
            local_path = os.path.join(output_dir, LOCAL_NAME)
            os.replace(downloaded_path, local_path)
            downloaded_path = local_path
        print(f"[PASS] Successfully downloaded weights to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    download_champion_weights(
        repo_id=args.repo_id,
        filename=args.filename,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
