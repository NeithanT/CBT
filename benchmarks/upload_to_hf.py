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

# CBT-18.95M: a character-level Transformer with two-dimensional rotary coordinates

- **GitHub**: [github.com/NeithanT/CBT](https://github.com/NeithanT/CBT)
- **Paper** (PDF, equations render): [RESEARCH_PAPER.pdf](RESEARCH_PAPER.pdf) | Markdown: [RESEARCH_PAPER.md](RESEARCH_PAPER.md)

## 1. Overview

**CBT-18.95M** is an 18.94M-parameter character-level Transformer with a 111-token vocabulary. Every token carries a rule-assigned two-dimensional coordinate (p, tau): p is the decimal place (or the dialogue segment) and tau is the step within it. Decoupled 2D Rotary Position Embeddings (2D-RoPE) rotate half of each attention head by p and half by tau. The six physical layers are applied three times with shared weights (effective depth 18). Random coordinate shift masking with k = 45 gives a coordinate horizon of p_max = 25 + 45 = 70.

This release (`champion_final.pt`) is the model after supervised fine-tuning: pretraining, search-copy and all-tools fine-tuning, an intermediate 12,000-step round, then a final 23,927-step round on a decontaminated online stream (35% search copy, 20% arithmetic replay of 1-25 digits, the rest multi-tool and chat).

**Main finding.** A matched 1D-RoPE twin (same architecture, data and step count) cannot add sums longer than its training maximum, while CBT-18.95M can. Tool routing and entity copying are the same for both, so they come from the training data, not from the layout.

## 2. Results

Greedy decoding. Addition: 30 random problems per length (70% uniform, 15% dense carries, 15% cascades), scored as teacher-forced greedy agreement with the gold scratchpad and answer. Training covers 1-25 digits.

**Addition: CBT-18.95M (2D-RoPE) against its 1D-RoPE twin**

| Operand digits | Status | CBT-18.95M (2D) | 1D-RoPE twin |
| :--- | :--- | :---: | :---: |
| 4, 8, 12, 14, 16 | seen | 150/150 | 149/150 |
| 20, 25 | seen | 60/60 | 57/60 |
| 26 | unseen (+1) | 30/30 | 0/30 |
| 30, 35, 40, 45, 50, 55 | unseen | 180/180 | 0/180 |
| 60 | unseen | 25/30 | 0/30 |
| 70 | unseen | 19/30 | 0/30 |
| **26-70 pooled** | unseen | **254/270** | **0/270** |

**Tool use (exact calls)**

| Suite | Prompts | CBT-18.95M (2D) | 1D-RoPE twin |
| :--- | :---: | :---: | :---: |
| Phrasing | 26 | 26 | 25 |
| Out-of-distribution entities | 18 | 18 | 18 |
| Unseen-entity copy | 30 | 30 | 30 |

The fine-tuning data was broadened after an earlier model's failures on the phrasing and OOD suites were seen, so the unseen-entity set (entities absent from every training corpus) is the cleaner evidence.

**Baselines trained on 1-12 digit sums with standard 1D-RoPE** (correct answers out of 30)

| Model | 4 digits | 8 digits | 12 digits | 14-35 digits |
| :--- | :---: | :---: | :---: | :---: |
| 18.9M character model | 30 | 30 | 0 | 0 at every length |
| 18.9M BPE model (word-level scratchpad) | 30 | 30 | 14 | 0 at every length |

All results are single training runs with small test sets. Details, limitations and the failure analysis are in the paper.

---

## 2b. Model Specifications

| Parameter | Specification |
| :--- | :--- |
| Total parameters | 18,944,512 |
| Non-embedding parameters | 18,887,680 |
| Physical layers | 6 (applied 3 times, effective depth 18) |
| Hidden dimension | 512 |
| Attention heads | 8 (head dimension 64) |
| Context | 1024 tokens |
| Positional embedding | Decoupled 2D-RoPE (half of each head rotated by place p, half by step tau; rule-assigned) |
| Coordinate shift masking | k = 45 |
| Vocabulary | 111 (14 control tokens + 97 characters) |
| Weight tying | input embedding = output projection |

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

`champion_final.pt`, source files for stand-alone execution, the paper (`RESEARCH_PAPER.pdf`, `RESEARCH_PAPER.md`), and the evaluation results in JSON.

---

## 5. Citation

```bibtex
@article{vargas2026cbt,
  title={From Digits to Characters: Two-Dimensional Rotary Coordinates for Length Generalization in a Compact Character-Level Transformer},
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
    parser.add_argument("--ckpt", type=str, default="checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sft.pt", help="Path to checkpoint")
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
        extra_files = {
            "RESEARCH_PAPER.md": "RESEARCH_PAPER.md",
            "RESEARCH_PAPER.pdf": "RESEARCH_PAPER.pdf",
            "results/champion_v3_arithmetic.json": "cbt_2d_arithmetic_20_70d.json",
            "results/champion_v3_arithmetic_4_16d.json": "cbt_2d_arithmetic_4_16d.json",
            "results/champion_v3_phrasing.json": "cbt_2d_phrasing.json",
            "results/champion_v3_ood.json": "cbt_2d_ood.json",
            "results/champion_v3_unseen_ood.json": "cbt_2d_unseen_copy.json",
            "results/cbt1d_v5_arithmetic.json": "cbt_1d_twin_arithmetic_20_70d.json",
            "results/cbt1d_v5_arithmetic_4_16d.json": "cbt_1d_twin_arithmetic_4_16d.json",
            "results/equalized_15d_1024_and_ruoss_sweep.json": "equalized_15d_1024_and_ruoss_sweep.json",
        }
        for src, dst in extra_files.items():
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp_dir, dst))
                print(f"  [OK] Staged {src} as {dst}")

        print(f"[INFO] Uploading complete model bundle to https://huggingface.co/{args.repo_id}...")
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="[UPDATE] Replace weights with the final SFT model; update paper, model card and evaluation results"
        )
        print(f"[PASS] Successfully uploaded ScaledGPT champion model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
