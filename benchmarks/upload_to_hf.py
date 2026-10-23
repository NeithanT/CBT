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
---

# 2D Embeddings for Length Generalization in Character-Based Transformers (CBT)

- **GitHub Repository**: [https://github.com/NeithanT/CBT](https://github.com/NeithanT/CBT)
- **Research Paper**: [RESEARCH_PAPER.md](https://github.com/NeithanT/CBT/blob/main/RESEARCH_PAPER.md) | [Paper PDF](https://github.com/NeithanT/CBT/blob/main/RESEARCH_PAPER.pdf)

## 1. Model Overview

**Character-Based Transformer (CBT)** is an 18.95M-parameter character-level Transformer whose tokens carry a rule-assigned two-dimensional (2D) coordinate $(p, \\tau)$ used by **Decoupled 2D Rotary Position Embeddings (2D-RoPE)** alongside **Weight-Tied Recurrent Depth ($R = 3$)**.

Trained strictly on 1- to 15-digit column additions (with random coordinate shift masking $k=32$, exposing macro column coordinates up to $p_{\\max} = 15 + 32 = 47$) and multi-tool dialogues, **CBT-18.95M** (`champion_final.pt`) achieves:
- **100.0% (20/20; 95% Wilson CI [83.9, 100.0]%)** on structured 16- to 50-digit sums (including 50-digit cascading carry $99\\dots99 + 1$).
- **100.0% (180/180; 95% Wilson CI [97.9, 100.0]%)** on random multi-carry sums from 16 to 40 digits, and **98.8% (237/240; CI [96.4, 99.6]%)** across 16 to 50 digits (with 238/240 = 99.2% exact `<THINK>` scratchpads and zero `(` delimiter errors).
- **90.0% (18/20)** at 55 digits before degrading beyond the shift-masked coordinate horizon $p_{\\max}=47$ (45.0% at 60d, 6.7% at 70d, 0.0% at 80–100d).
- **Equalized 1–15d, $T=1024$ Control Study (12,800 Random Problems)**: When four equalized 100k models (`1D-RoPE Standard`, `1D-RoPE + Offset`, `1D-RoPE + Ruoss et al. 2023 Randomized Positions`, and `Decoupled 2D-RoPE k=32`) are trained on the exact same 1–15d dataset with $T=1024$ and evaluated on 200 random problems per length up to 100d, all three 1D/Ruoss variants score **0.0% (0/200, CI [0.0, 1.9]%)** across 16–100d, whereas equalized 2D-RoPE achieves **81.4% (977/1,200, CI [79.1, 83.5]%)** across 16–40d (up to 92.0% at 16d and 25d).
- **100.0% (26/26; CI [87.1, 100.0]%)** tool routing and **96.2% (25/26)** exact argument extraction across 9 tool schemas on phrasing templates shared with training.
- **88.9% (16/18)** zero-shot OOD tool routing, with **50.0% (8/16)** exact character argument copying (and 68.8% exact or near-exact).

---

## 2. Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Model Name** | Character-Based Transformer (CBT-18.95M) |
| **Total Parameters** | 18,946,560 (~18.95M) |
| **Non-Embedding Parameters** | 18,887,680 (~18.89M) |
| **Physical Layers (L)** | 6 |
| **Recurrent Passes (R)** | 3 (Effective Depth = 18 layers) |
| **Hidden Dimension (d_model)** | 512 |
| **Attention Heads (H)** | 8 (Head dimension = 64) |
| **Max Sequence Length (T)** | 1024 (Coordinate cache up to 4096) |
| **Positional Embedding** | Decoupled 2D-RoPE (d/2 macro column $p$, d/2 micro step $\\tau$; rule-assigned via `compute_2d_coords`) |
| **Coordinate Shift Masking (k)** | $k = 32$ ($p_{\\max} = 15 + 32 = 47$ during 1–15d training) |
| **Vocabulary Size** | 115 (14 structural control tags + 101 ASCII characters) |
| **Weight Tying** | W_te == W_lm_head |

---

## 3. Empirical Evaluation Benchmarks (Greedy Decoding)

| Benchmark Suite | Scope / Test Cases | Final Score | 95% Wilson CI |
| :--- | :--- | :--- | :--- |
| **1–15 Digit In-Distribution Sums** | Seen in Training (1–15d) | **100.0%** | -- |
| **16–50 Digit Structured Extrapolation** | 20 Cases (incl. $99\\dots99 + 1$) | **100.0% (20/20)** | [83.9, 100.0]% |
| **16–40 Digit Random Multi-Carry Sweep** | 6 Lengths $\\times$ 30 Cases ($n=180$, $p \\le 40$) | **100.0% (180/180)** | [97.9, 100.0]% |
| **45–50 Digit Random Multi-Carry (At $p_{\\max}=47$)** | 2 Lengths $\\times$ 30 Cases ($n=60$; 58/60 Think OK) | **95.0% (57/60)** | [86.3, 98.3]% |
| **Pooled 16–50 Digit Random Multi-Carry** | 8 Lengths $\\times$ 30 Cases ($n=240$; 238/240 Think OK) | **98.8% (237/240)** | [96.4, 99.6]% |
| **55–100 Digit Beyond Horizon ($p > 47$)** | 55d: 90.0% (18/20), 60d: 45.0% (9/20), 70d: 6.7%, 80–100d: 0.0% | Bounded by $p_{\\max}=47$ | -- |
| **Multi-Tool Routing Suite (9 Schemas)** | 26 Template-Matched Cases | **100.0% (26/26)** | [87.1, 100.0]% |
| **Multi-Tool Exact Argument Extraction** | 26 Template-Matched Cases | **96.2% (25/26)** | [81.1, 99.3]% |
| **Zero-Shot OOD Tool Selection** | 18 Unseen Prompts | **88.9% (16/18)** | [67.2, 96.9]% |
| **Zero-Shot OOD Exact Argument Copy** | 16 Correctly Routed OOD Prompts | **50.0% (8/16)** | [28.0, 72.0]% |

---

## 4. Usage & Inference

### Local Quickstart

```python
import torch
from model import ScaledGPT, ScaledConfig
from tokenizer import ScaledTokenizer

# 1. Load configuration and model
config = ScaledConfig()
model = ScaledGPT(config)

# 2. Load weights
state_dict = torch.load("champion_final.pt", map_location="cpu", weights_only=True)
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
@article{vargas2026cbt,
  title={2D Embeddings for Length Generalization in Character-Based Transformers (CBT)},
  author={Vargas Vargas, Neithan},
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

        # 4. Copy research paper and random 16-100d evaluation results
        extra_files = [
            "RESEARCH_PAPER.pdf",
            "RESEARCH_PAPER.md",
            "results/random_extrapolation_16_to_100d.json",
            "results/equalized_15d_1024_and_ruoss_sweep.json",
        ]
        for ef in extra_files:
            if os.path.isfile(ef):
                shutil.copy2(ef, os.path.join(tmp_dir, os.path.basename(ef)))
                print(f"  [OK] Staged {ef}")

        print(f"[INFO] Uploading complete model bundle to https://huggingface.co/{args.repo_id}...")
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="[UPDATE] Update CBT-18.95M paper, model card, and 16-100d random extrapolation results"
        )
        print(f"[PASS] Successfully uploaded ScaledGPT champion model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
