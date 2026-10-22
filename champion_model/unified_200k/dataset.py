"""
Joint Balanced Dataset & Batch Encoder for Unified Math & Tool Calling.
"""

import os
import sys
import random
from typing import List, Dict, Tuple, Any, Optional
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.unified_200k.tokenizer import UnifiedTokenizer
from champion_model.math_model.arithmetic import format_addition_char_scratchpad
from champion_model.math_model.train_and_eval import load_training_equations
from champion_model.tool_model.dataset import generate_dataset as generate_tool_dataset


def generate_arithmetic_samples(n_samples: int = 2000, seed: int = 42) -> List[Dict[str, str]]:
    """Loads arithmetic equations using the exact balanced 1-12 digit curriculum from the champion model."""
    texts = load_training_equations(n_samples=n_samples, root_dir=ROOT_DIR, seed=seed, curriculum=True)
    samples: List[Dict[str, str]] = []
    for full_text in texts:
        prompt = full_text.split("<THINK>")[0]
        samples.append({
            "task_type": "math",
            "prompt": prompt,
            "full_text": full_text,
        })
    return samples


def generate_joint_dataset(
    n_math: int = 2000,
    n_tools: int = 2000,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Generates dual datasets for math and tools."""
    math_samples = generate_arithmetic_samples(n_samples=n_math, seed=seed)
    raw_tools = generate_tool_dataset(n_samples=n_tools, seed=seed + 1)
    tool_samples = [
        {
            "task_type": "tools",
            "prompt": f"<USER>{item['user_prompt']}</USER>",
            "full_text": item["full_text"],
        }
        for item in raw_tools
    ]
    return math_samples, tool_samples


def encode_joint_batch(
    math_items: List[Dict[str, str]],
    tool_items: List[Dict[str, str]],
    tokenizer: UnifiedTokenizer,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes balanced joint batch of math and tool sequences.
    Applies exact prompt loss masking:
    - Math: masks everything up to and including '<THINK>'
    - Tools: masks everything inside '<USER>...</USER>'
    """
    items = math_items + tool_items
    B = len(items)

    encoded_seqs = [tokenizer.encode(item["full_text"], add_bos=True, add_eos=True) for item in items]
    max_len = max(len(s) for s in encoded_seqs) - 1

    x_tensor = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long)
    y_tensor = torch.full((B, max_len), -100, dtype=torch.long)

    think_s_id = tokenizer.think_s_id
    user_e_id = tokenizer.user_e_id

    for i, (item, tokens) in enumerate(zip(items, encoded_seqs)):
        n = len(tokens) - 1
        t_tok = torch.tensor(tokens, dtype=torch.long)
        x_tensor[i, :n] = t_tok[:-1]
        y_tensor[i, :n] = t_tok[1:].clone()

        if item["task_type"] == "math":
            # Mask prompt up to '=', so that at input '<THINK>', target is '(' (identical to champion MathGPT)
            think_matches = (t_tok[:-1] == think_s_id).nonzero(as_tuple=True)[0]
            if len(think_matches) > 0:
                mask_until = think_matches[0].item()
                y_tensor[i, :mask_until] = -100
        else:
            # Mask prompt up to '</USER>' so at input '</USER>', target is '<THINK>'
            user_e_matches = (t_tok[:-1] == user_e_id).nonzero(as_tuple=True)[0]
            if len(user_e_matches) > 0:
                mask_until = user_e_matches[0].item()
                y_tensor[i, :mask_until] = -100

    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    return x_tensor, y_tensor
