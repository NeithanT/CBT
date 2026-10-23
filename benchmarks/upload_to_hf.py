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

Trained on 1- to 15-digit column additions (random coordinate shift masking $k=45$, exposing macro column coordinates up to $p_{\\max} = 70$) and multi-tool dialogues, **CBT-18.95M** (`champion_v2_final.pt`; 12,000 SFT steps, final loss 0.0034) achieves (single run, greedy decoding):
- **100.0% (90/90; 95% Wilson CI [95.9, 100.0]%)** on random multi-carry sums from 16 to 55 digits (n=10 per length; 80/80 on 16-50 digits).
- **96.7% (29/30; CI [83.3, 99.4]%)** tool routing and **56.7% (17/30; CI [39.2, 72.6]%)** exact argument copying on 30 entity-disjoint unseen prompts (not shown to exceed 50%).
- **80.8% (21/26; CI [62.1, 91.5]%)** tool routing and **73.1% (19/26)** exact argument extraction on the 26-case multi-schema suite.
- Known failures: `database_query`, `fetch_api`, `create_event`, and `file_manager` cases misroute, an embedded digit (`crispr cas9`) misroutes, and a casual greeting can trigger a spurious `get_weather` call.
- Equalized 1-15d, $T=1024$ control study (100k models, 200 random problems per length): all three 1D/Ruoss variants score 0.0% across 16-100d, whereas equalized 2D-RoPE achieves 81.4% (977/1,200, CI [79.1, 83.5]%) across 16-40d.

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
| **Coordinate Shift Masking (k)** | $k = 45$ ($p_{\\max} = 70$ during 1-15d training) |
| **Vocabulary Size** | 115 (14 structural control tags + 101 ASCII characters) |
| **Weight Tying** | W_te == W_lm_head |

---

## 3. Empirical Evaluation Benchmarks (Greedy Decoding)

| Benchmark Suite | Scope / Test Cases | Final Score | 95% Wilson CI |
| :--- | :--- | :--- | :--- |
| **Random Multi-Carry Sums, 16-50d** | 8 Lengths x 10 Cases (n=80) | **100.0% (80/80)** | [95.4, 100.0]% |
| **Random Multi-Carry Sums, 16-55d** | 9 Lengths x 10 Cases (n=90) | **100.0% (90/90)** | [95.9, 100.0]% |
| **Unseen-Entity Tool Routing** | 30 Entity-Disjoint Prompts | **96.7% (29/30)** | [83.3, 99.4]% |
| **Unseen-Entity Exact Argument Copy** | 30 Entity-Disjoint Prompts | **56.7% (17/30)** | [39.2, 72.6]% |
| **Multi-Tool Routing Suite (9 Schemas)** | 26 Phrasing and Edge Cases | **80.8% (21/26)** | [62.1, 91.5]% |
| **Multi-Tool Exact Argument Extraction** | 26 Phrasing and Edge Cases | **73.1% (19/26)** | [53.9, 86.3]% |

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
state_dict = torch.load("champion_v2_final.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# 3. Initialize tokenizer
tokenizer = ScaledTokenizer()
```

### Interactive CLI

```bash
python interactive.py --ckpt champion_v2_final.pt
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
    parser.add_argument("--ckpt", type=str, default="champion_model/scaled_t4/checkpoints/champion_v2_final.pt", help="Path to checkpoint")
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
        target_ckpt = os.path.join(tmp_dir, "champion_v2_final.pt")
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
