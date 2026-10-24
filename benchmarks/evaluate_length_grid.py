#!/usr/bin/env python3
"""Accuracy grid over (digits of a, digits of b) for a CBT checkpoint, teacher-forced greedy agreement.

Each cell samples uniform random operands with exactly the given digit counts.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from benchmarks.evaluate_champion import teacher_forced_match, wilson_ci
from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig
from champion_model.scaled_t4.tokenizer import ScaledTokenizer

DEFAULT_SIZES = "5,10,15,20,25,30,40,50,60,70"


def sample_operand(rng: random.Random, digits: int) -> int:
    return rng.randint(10 ** (digits - 1), 10 ** digits - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Operand-length grid evaluation for a CBT checkpoint")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sizes", type=str, default=DEFAULT_SIZES)
    parser.add_argument("--samples", type=int, default=10, help="Problems per cell")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out-json", type=str, required=True)
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    tokenizer = ScaledTokenizer()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt.get("config", ScaledGPTConfig(vocab_size=tokenizer.vocab_size))
    model = ScaledGPT(config, tokenizer=tokenizer).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sizes: List[int] = [int(s) for s in args.sizes.split(",") if s.strip()]
    cells: Dict[str, Dict[str, object]] = {}
    for da in sizes:
        for db in sizes:
            rng = random.Random(args.seed + 1000 * da + db)
            t0 = time.time()
            k = 0
            for _ in range(args.samples):
                a, b = sample_operand(rng, da), sample_operand(rng, db)
                k += int(teacher_forced_match(model, tokenizer, device, a, b))
            lo, hi = wilson_ci(k, args.samples)
            cells[f"{da},{db}"] = {"a_digits": da, "b_digits": db, "n": args.samples, "exact": k,
                                    "ci": [lo, hi]}
            print(f"[INFO] a={da:>2} b={db:>2}: {k}/{args.samples} ({time.time() - t0:.1f}s)", flush=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.ckpt, "sizes": sizes, "samples_per_cell": args.samples,
                   "seed": args.seed, "cells": cells}, f, indent=2)
    print(f"[PASS] Saved {args.out_json}")


if __name__ == "__main__":
    main()
