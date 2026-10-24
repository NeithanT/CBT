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
- **Research Paper**: [RESEARCH_PAPER.md](https://github.com/NeithanT/CBT/blob/main/RESEARCH_PAPER.md)

## 1. Model Overview

**Character-Based Transformer (CBT-18.95M)** is an 18.94M-parameter character-level Transformer (111-token vocabulary) whose tokens carry a rule-assigned two-dimensional coordinate $(p, \\tau)$ used by **Decoupled 2D Rotary Position Embeddings (2D-RoPE)**, with weight-tied recurrent depth ($R = 3$). This release is the **V5 model** (`champion_final.pt`): pretraining, search-copy and all-tools fine-tuning, V4 fine-tuning, then a 23,927-step V5 fine-tuning on a decontaminated online stream (35% search copy, 20% arithmetic replay of 1-25 digits, the rest multi-tool and chat). Coordinate shift masking uses $k=45$, so $p_{\\max} = 25 + 45 = 70$.

Results (greedy decoding; Wilson 95% CIs are in the paper and in `results/`):
- **Addition**, 30 random problems per length (70% uniform, 15% dense carry, 15% cascades), scored as teacher-forced greedy agreement with the gold scratchpad and answer: **150/150** at 4-16 digits, **60/60** at 20 and 25 digits (seen), **210/210** at 26-55 digits (unseen), **25/30** at 60 and **19/30** at 70 digits. Pooled over the unseen 26-70 digits: **254/270**.
- **Tool use**: **26/26** exact calls on a 26-prompt phrasing suite, **18/18** on 18 out-of-distribution prompts, and **30/30** exact copies of entities absent from the training corpus (unseen-30). The V5 data was broadened after the previous model's failures on the first two suites were seen, so the unseen-30 set is the cleaner evidence.
- **1D-RoPE control**: a twin with the same architecture, V5 data and step count but standard 1D-RoPE scores **0/270** at 26-70 digits and **73/74** exact tool calls. The 2D coordinates account for the arithmetic extrapolation, not for the tool-use results.
- **Baselines**: standard 1D-RoPE models trained on 1-12 digit sums score 0/30 at every length from 14 to 35 digits (character and BPE vocabularies; looped, hierarchical and hybrid variants too).

All results are single training runs. Details, limitations and the failure analysis are in the paper.

---

## 2. Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Total Parameters** | 18,944,512 |
| **Non-Embedding Parameters** | 18,887,680 |
| **Physical Layers (L)** | 6 |
| **Recurrent Passes (R)** | 3 (effective depth 18) |
| **Hidden Dimension** | 512 |
| **Attention Heads** | 8 (head dimension 64) |
| **Context** | 1024 tokens |
| **Positional Embedding** | Decoupled 2D-RoPE (half of each head rotated by place $p$, half by step $\\tau$; rule-assigned) |
| **Coordinate Shift Masking** | $k = 45$ |
| **Vocabulary** | 111 (14 control tokens + 97 characters) |
| **Weight Tying** | input embedding = output projection |

---

## 3. Usage

The file is a `torch.save` dictionary with the config and `model_state_dict`.

```python
import torch
from model import ScaledGPT
from tokenizer import ScaledTokenizer

ckpt = torch.load("champion_final.pt", map_location="cpu", weights_only=False)
tokenizer = ScaledTokenizer()
model = ScaledGPT(ckpt["config"], tokenizer=tokenizer)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

Interactive CLI:

```bash
python interactive.py --ckpt champion_final.pt
```

---

## 4. Files

`champion_final.pt` (V5 model), source files for stand-alone execution, `RESEARCH_PAPER.md`, and the evaluation results in JSON.

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
    parser.add_argument("--ckpt", type=str, default="checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sftv5.pt", help="Path to checkpoint")
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

        # 4. Copy research paper and evaluation results
        extra_files = [
            "RESEARCH_PAPER.md",
            "results/champion_v3_arithmetic.json",
            "results/champion_v3_arithmetic_4_16d.json",
            "results/champion_v3_phrasing.json",
            "results/champion_v3_ood.json",
            "results/champion_v3_unseen_ood.json",
            "results/cbt1d_v5_arithmetic.json",
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
            commit_message="[UPDATE] Replace weights with the V5 model; update paper, model card and evaluation results"
        )
        print(f"[PASS] Successfully uploaded ScaledGPT champion model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
