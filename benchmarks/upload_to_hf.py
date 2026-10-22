#!/usr/bin/env python3
"""
Hugging Face Upload Orchestrator for ScaledGPT Champion Model.
Bundles and uploads the trained checkpoint, architecture definitions,
tokenizer, configuration, and model card to a Hugging Face repository.
"""

import os
import sys
import argparse
import tempfile
import shutil
from typing import Optional

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("[FAIL] huggingface_hub is not installed. Please run: pip install huggingface_hub")
    sys.exit(1)


HF_MODEL_CARD = """---
language:
- en
license: mit
tags:
- arithmetic
- length-generalization
- recurrent-transformers
- 2d-rope
- tool-use
- function-calling
- reasoning
datasets:
- synthetic
metrics:
- accuracy
pipeline_tag: text-generation
---

# ScaledGPT (~18.95M): Recurrent 2D-RoPE Architecture for Algorithmic Length Extrapolation

## 1. Model Overview

**ScaledGPT** is an autoregressive Transformer model engineered to overcome the empirical limits of standard self-attention on sequential state machines and multi-tool agent execution.

By combining **Decoupled 2D Rotary Position Embeddings (2D-RoPE)** with **Weight-Tied Recurrent Depth ($R=3$)**, ScaledGPT achieves:
- **100.0% Exact Match Accuracy** on 1 to 50 digit column arithmetic length extrapolation (including 50-digit cascading carry $99\\dots99 + 1$).
- **100.0% Multi-Tool Routing Accuracy** across 9 tool schemas (single-tool, multi-tool chaining, nested JSON schemas, and error recovery).
- **Dual NVIDIA T4 GPU Efficiency**: Native Turing `sm_75` FP16 AMP and SDPA memory-efficient attention.

---

## 2. Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Total Parameters** | 18,944,512 (~18.95M) |
| **Non-Embedding Parameters** | 18,887,680 (~18.89M) |
| **Physical Layers ($L$)** | 6 |
| **Recurrent Passes ($R$)** | 3 (Effective Depth = 18 layers) |
| **Hidden Dimension ($d_{\\text{model}}$)** | 512 |
| **Attention Heads ($H$)** | 8 (Head dimension = 64) |
| **Max Sequence Length ($T$)** | 1024 |
| **Positional Embedding** | Decoupled 2D-RoPE ($d/2$ macro column, $d/2$ micro step) |
| **Vocabulary Size** | 115 (Special discourse markers + ASCII characters) |
| **Weight Tying** | $W_{\\text{te}} == W_{\\text{lm\\_head}}$ |

---

## 3. Empirical Evaluation Benchmark

| Benchmark Suite | Metric | Score | Status |
| :--- | :--- | :--- | :--- |
| **1-15 Digit In-Distribution Math** | Exact Answer Match | 100.0% (6/6) | [PASS] |
| **16-25 Digit Intermediate Extrapolation** | Exact Answer Match | 100.0% (6/6) | [PASS] |
| **30-40 Digit High Extrapolation** | Exact Answer Match | 100.0% (4/4) | [PASS] |
| **45-50 Digit Ultra Extrapolation** | Exact Answer Match | 100.0% (4/4) | [PASS] |
| **50-Digit Cascade Carry ($99\\dots99 + 1$)** | 50 Sequential Transitions | 100.0% (1/1) | [PASS] |
| **Multi-Tool Routing Suite (9 Schemas)** | Correct Tool Selection | 100.0% (26/26) | [PASS] |
| **Multi-Tool Exact Argument Extraction** | Exact Parameter Copy Rate | 96.2% (25/26) | [PASS] |

---

## 4. Usage & Inference

### Local Quickstart

```python
import torch
from model import ScaledGPT, ScaledGPTConfig
from tokenizer import ScaledTokenizer

# 1. Load configuration and model
config = ScaledGPTConfig(
    block_size=1024,
    vocab_size=115,
    n_layer=6,
    n_head=8,
    n_embd=512,
    n_recur=3,
    pos_emb_type="rope_2d"
)
model = ScaledGPT(config)

# 2. Load weights
state_dict = torch.load("champion_final.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 3. Initialize tokenizer
tokenizer = ScaledTokenizer()
```

### Interactive CLI

```bash
python interactive.py --ckpt champion_final.pt
```

---

## 5. Citation

```bibtex
@article{vargas2026attention,
  title={Attention Is Not All You Need for AGI: Empirical Limits of Pure Self-Attention on Algorithmic Length Generalization},
  author={Vargas, Neithan},
  year={2026}
}
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Upload ScaledGPT Champion weights to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="HF Repo ID in format 'username/repo_name'")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (write access)")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    parser.add_argument("--ckpt", type=str, default="champion_model/scaled_t4/checkpoints/champion_final.pt", help="Path to checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        print(f"[INFO] Authenticated as Hugging Face user: {user_info.get('name')}")
    except Exception as e:
        print("[FAIL] Hugging Face authentication failed. Please run `hf auth login` or pass `--token`.")
        sys.exit(1)

    print(f"[INFO] Creating/Verifying repository '{args.repo_id}'...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[INFO] Staging model files in temporary bundle: {tmp_dir}")

        # 1. Write model card README.md
        with open(os.path.join(tmp_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(HF_MODEL_CARD)

        # 2. Copy checkpoint
        target_ckpt = os.path.join(tmp_dir, "champion_final.pt")
        print(f"[INFO] Copying checkpoint ({os.path.getsize(args.ckpt) / (1024*1024):.1f} MB)...")
        shutil.copy2(args.ckpt, target_ckpt)

        # 3. Copy source files for stand-alone execution
        base_dir = "champion_model/scaled_t4"
        files_to_copy = ["model.py", "tokenizer.py", "dataset.py", "config.json", "interactive.py"]
        for fname in files_to_copy:
            src_path = os.path.join(base_dir, fname)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(tmp_dir, fname))
                print(f"  [OK] Staged {fname}")

        print(f"[INFO] Uploading complete model bundle to https://huggingface.co/{args.repo_id}...")
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="[RELEASE] Upload ScaledGPT 18.95M champion model weights and artifacts"
        )
        print(f"[PASS] Successfully uploaded ScaledGPT champion model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
