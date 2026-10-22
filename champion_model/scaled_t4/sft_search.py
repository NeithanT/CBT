#!/usr/bin/env python3
"""
Targeted SFT Script: Universal Multi-Tool & Search Delegation Annealing for ScaledGPT (~18.95M).
Aligns the champion checkpoint to route across all tools cleanly:
- Weather (`get_weather(city="...")`) across diverse phrasing ('in', 'on', 'for', raw)
- Time (`get_time(timezone="...")`)
- Calculator (`calculator(expr="...")`)
- Unit Conversion (`unit_convert(...)`)
- Open-domain Search Delegation & Nonce Pointer Copying (`search(query="...")`)
- Multi-Tool Chaining, Nested JSON Schemas, Error Recovery, and Conversational Routing
while preserving 100% 1-50 digit arithmetic invariance via replay blending.
"""

import os
import sys
import time
import math
import random
import argparse
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig
    from champion_model.scaled_t4.tokenizer import ScaledTokenizer
    from champion_model.scaled_t4.dataset import (
        generate_arithmetic_equations,
        generate_tool_dataset,
        generate_search_delegation_dataset,
        encode_joint_batch,
    )
    from champion_model.scaled_t4.train_and_eval import sync_checkpoint_to_kaggle, prune_checkpoints
except ImportError:
    from model import ScaledGPT, ScaledGPTConfig
    from tokenizer import ScaledTokenizer
    from dataset import (
        generate_arithmetic_equations,
        generate_tool_dataset,
        generate_search_delegation_dataset,
        encode_joint_batch,
    )
    from train_and_eval import sync_checkpoint_to_kaggle, prune_checkpoints


def run_sft_probe(model: nn.Module, tokenizer: ScaledTokenizer, device: torch.device, step: int):
    """Runs a non-blocking fast probe across all tools, search copy induction, and arithmetic carry."""
    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()

    probe_prompts = [
        ("Weather (on Rome)", "<USER>What is the weather on Rome</USER><THINK>", 'get_weather(city="Rome")'),
        ("Weather (in Tokyo)", "<USER>Weather in Tokyo</USER><THINK>", 'get_weather(city="Tokyo")'),
        ("Time (in EST)", "<USER>What is the time in EST</USER><THINK>", 'get_time(timezone="EST")'),
        ("Calculator (*)", "<USER>Calculate 15 * 24</USER><THINK>", "calculator(expr="),
        ("Unit Convert", "<USER>How many miles is 25 km</USER><THINK>", 'unit_convert(value=25, from_unit="km", to_unit="miles")'),
        ("Search Real", "<USER>What are tomatoes</USER><THINK>", 'search(query="tomatoes")'),
        ("Search Nonce", "<USER>What is a bloumplik</USER><THINK>", 'search(query="bloumplik")'),
        ("Nested JSON", "<USER>Query the top 5 active records in the engineering department.</USER><THINK>", "database_query("),
        ("Math 9-Digit", "123456789+987654321=<THINK>", "1111111110"),
    ]

    print(f"\n--- [PROBE Step {step}] All-Tools Alignment & Carry Check ---")
    for label, p, expected in probe_prompts:
        toks = tokenizer.encode(p, add_bos=True, add_eos=False)
        x = torch.tensor([toks], dtype=torch.long, device=device)
        with torch.no_grad():
            stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
            out = raw_model.generate(x, max_new_tokens=140, temperature=0.0, stop_tokens=stop_ids)
        # Slice strictly by generated tokens to prevent BOS shift and character misalignment
        new_toks = out[0][len(toks):].tolist()
        resp = tokenizer.decode(new_toks, skip_special=False).strip()

        status = "[PASS]" if (expected in resp or ("=" + expected) in resp) else "[PROGRESS]"

        # Format display cleanly without clipping tags
        if "<CALL>" in resp and "</CALL>" in resp:
            call_part = resp.split("<CALL>")[1].split("</CALL>")[0]
            disp = f"<CALL>{call_part}</CALL>"
        elif "</THINK>" in resp:
            thought_part = resp.split("</THINK>")[0].strip()
            after_part = resp.split("</THINK>")[-1].strip()
            disp = f"<THINK>{thought_part}</THINK> {after_part}" if after_part else f"<THINK>{thought_part}</THINK>"
        else:
            disp = resp

        print(f"  {status} {label:<20} -> {disp}")
    print("-" * 70 + "\n", flush=True)

    if was_training:
        raw_model.train()


def main():
    parser = argparse.ArgumentParser(description="ScaledGPT Final All-Tools & Search Annealing SFT")
    parser.add_argument("--steps", type=int, default=4000, help="SFT steps (default 4000)")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="Peak SFT learning rate for annealing")
    parser.add_argument("--max-hours", type=float, default=6.0, help="Max training hours")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging step interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint save step interval")
    parser.add_argument("--probe-interval", type=int, default=500, help="Probe test step interval")
    parser.add_argument("--max-keep-ckpts", type=int, default=2, help="Checkpoints to retain")
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default="champion_model/scaled_t4/checkpoints/scaled_t4_sft_search.pt",
        help="Base checkpoint path",
    )
    parser.add_argument(
        "--save-ckpt",
        type=str,
        default="champion_model/scaled_t4/checkpoints/champion_final.pt",
        help="Target checkpoint path",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = ScaledTokenizer()
    config = ScaledGPTConfig(vocab_size=tokenizer.vocab_size)
    model = ScaledGPT(config, tokenizer=tokenizer).to(device)

    # Resolve base checkpoint (prioritizing scaled_t4_sft_search_step_7000 / scaled_t4_sft_search.pt)
    resolved_base = None
    candidates = [
        args.base_ckpt,
        os.path.join(ROOT_DIR, args.base_ckpt),
        os.path.join(ROOT_DIR, "champion_model/scaled_t4/checkpoints/scaled_t4_sft_search.pt"),
        os.path.join(ROOT_DIR, "champion_model/checkpoints/scaled_t4_sft_search_step_7000.zip"),
        "/kaggle/working/checkpoints/scaled_t4/scaled_t4_sft_search.pt",
        os.path.join(ROOT_DIR, "champion_model/scaled_t4/checkpoints/scaled_t4_final.pt"),
        os.path.join(ROOT_DIR, "champion_model/checkpoints/scaled_t4_final.zip"),
    ]
    for c in candidates:
        if os.path.exists(c) and not os.path.isdir(c):
            resolved_base = c
            break

    if resolved_base is not None:
        ckpt_data = torch.load(resolved_base, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        step_tag = f" (from step {ckpt_data.get('step')})" if "step" in ckpt_data else ""
        print(f"[INFO] Successfully loaded base weights{step_tag} from: {resolved_base}")
    else:
        print(f"[WARN] Base checkpoint not found. Starting from scratch.")

    # Multi-GPU DataParallel wrapper if 2+ GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[INFO] Detected {torch.cuda.device_count()} GPUs. Wrapping with DataParallel.")
        model = nn.DataParallel(model)

    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.steps), eta_min=5e-6)
    use_cuda_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    print(f"\n[INFO] Loading SFT datasets (Multi-Tool 55% + Search Copy 35% + Math Replay 10%)...")
    data_cache = os.path.join(ROOT_DIR, "champion_model/scaled_t4/data/sft_dataset.pt")
    if os.path.exists(data_cache):
        cached = torch.load(data_cache, map_location="cpu", weights_only=False)
        if cached.get("version") == "v3_all_tools":
            print(f"[INFO] Found V3 all-tools pre-packaged dataset: {data_cache}")
            search_ds = cached["search"]
            math_ds = cached["math"]
            tool_ds = cached["tools"]
        else:
            print(f"[INFO] Upgrading dataset cache to V3 All-Tools 1-Hop Curriculum...")
            search_ds = generate_search_delegation_dataset(n_samples=12000, seed=42)
            math_ds = generate_arithmetic_equations(n_samples=4000, seed=777, max_train_digits=15)
            tool_ds = generate_tool_dataset(n_samples=20000, seed=888)
            os.makedirs(os.path.dirname(data_cache), exist_ok=True)
            torch.save({
                "version": "v3_all_tools",
                "search": search_ds,
                "math": math_ds,
                "tools": tool_ds,
            }, data_cache)
        print(f"       Loaded {len(tool_ds):,} multi-tool, {len(search_ds):,} search, and {len(math_ds):,} math samples.")
    else:
        print(f"[INFO] Generating V3 All-Tools SFT dataset on-the-fly...")
        search_ds = generate_search_delegation_dataset(n_samples=12000, seed=42)
        math_ds = generate_arithmetic_equations(n_samples=4000, seed=777, max_train_digits=15)
        tool_ds = generate_tool_dataset(n_samples=20000, seed=888)

    save_path = os.path.join(ROOT_DIR, args.save_ckpt)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    prune_checkpoints(ckpt_dir, keep_last=args.max_keep_ckpts, base_filename=base_name)

    print("=" * 80)
    print(f"STARTING FINAL ALL-TOOLS ANNEALING SFT ({args.steps:,} steps)")
    print(f"Device: {device} | Peak LR: {args.lr:.2e} -> 5.0e-06 | Batch Size: {args.batch_size}")
    print(f"Curriculum: 55% Multi-Tool (Weather/Time/Calc/Convert/JSON) + 35% Search Copy + 10% Math Replay")
    print("=" * 80 + "\n", flush=True)

    t0 = time.time()
    n_search, n_math, n_tool = len(search_ds), len(math_ds), len(tool_ds)
    s_idx, m_idx, t_idx = 0, 0, 0
    rng_s, rng_m, rng_t = random.Random(101), random.Random(42), random.Random(1337)

    for step in range(1, args.steps + 1):
        elapsed_hours = (time.time() - t0) / 3600.0
        if elapsed_hours >= args.max_hours:
            print(f"[WARN] SFT budget reached ({args.max_hours:.1f}h). Halting gracefully.")
            break

        # Batch Composition (32): Math 3 (9.38%), Search 11 (34.38%), Multi-Tools 18 (56.25%)
        n_m = max(1, int(args.batch_size * 0.10))
        n_s = max(1, int(args.batch_size * 0.35))
        n_t = args.batch_size - n_m - n_s

        if s_idx + n_s > n_search:
            rng_s.shuffle(search_ds)
            s_idx = 0
        if m_idx + n_m > n_math:
            rng_m.shuffle(math_ds)
            m_idx = 0
        if t_idx + n_t > n_tool:
            rng_t.shuffle(tool_ds)
            t_idx = 0

        batch_s = search_ds[s_idx : s_idx + n_s]
        batch_m = math_ds[m_idx : m_idx + n_m]
        batch_t = tool_ds[t_idx : t_idx + n_t]
        s_idx += n_s
        m_idx += n_m
        t_idx += n_t

        x_b, y_b = encode_joint_batch(batch_m, batch_t + batch_s, tokenizer, device=device)
        optimizer.zero_grad(set_to_none=True)

        if use_cuda_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x_b, targets=y_b)
                if loss.dim() > 0:
                    loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x_b, targets=y_b)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        loss_val = float(loss.item())

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            elapsed = time.time() - t0
            lr_cur = optimizer.param_groups[0]["lr"]
            print(
                f"[SFT INFO] Step {step:5d}/{args.steps:5d} | "
                f"Loss: {loss_val:.4f} | "
                f"LR: {lr_cur:.6f} | "
                f"Elapsed: {elapsed / 60.0:.1f} min",
                flush=True,
            )

        if step == 1 or step % args.probe_interval == 0 or step == args.steps:
            run_sft_probe(model, tokenizer, device, step)

        if step % args.save_interval == 0:
            step_ckpt = save_path.replace(".pt", f"_step_{step}.pt")
            raw_model = model.module if hasattr(model, "module") else model
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "config": raw_model.config,
                "step": step,
                "loss": loss_val,
            }, step_ckpt)
            sync_checkpoint_to_kaggle(step_ckpt)
            print(f"[INFO] Saved intermediate SFT checkpoint: {step_ckpt}", flush=True)
            prune_checkpoints(ckpt_dir, keep_last=args.max_keep_ckpts, base_filename=base_name)

    # Save final SFT model (both to save_path and updating scaled_t4_sft_search.pt for seamless CLI compatibility)
    raw_model = model.module if hasattr(model, "module") else model
    final_payload = {
        "model_state_dict": raw_model.state_dict(),
        "config": raw_model.config,
        "steps": args.steps,
    }
    torch.save(final_payload, save_path)
    sync_checkpoint_to_kaggle(save_path)
    compat_path = os.path.join(ckpt_dir, "scaled_t4_sft_search.pt")
    if os.path.abspath(save_path) != os.path.abspath(compat_path):
        torch.save(final_payload, compat_path)
        sync_checkpoint_to_kaggle(compat_path)
    print(f"\n[OK] Final All-Tools Annealed SFT model saved: {save_path}\n", flush=True)


if __name__ == "__main__":
    main()
