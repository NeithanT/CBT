"""
Unified training and benchmark evaluation suite for ~100k ToolGPT.
"""

import os
import sys
import time
import argparse
import random
from typing import List, Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tool_model.tokenizer import ToolTokenizer
from tool_model.model import ToolGPT, ToolGPTConfig
from tool_model.dataset import generate_dataset, generate_tool_example


def encode_tool_batch(
    samples: List[Dict[str, str]],
    tokenizer: ToolTokenizer,
    device: Optional[torch.device] = None,
    mask_prompt: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes batch with prompt loss masking (masks everything inside <USER>...</USER>)."""
    B = len(samples)
    raw_tokens = [tokenizer.encode(s["full_text"], add_bos=True, add_eos=True) for s in samples]
    max_len = max(len(t) for t in raw_tokens) - 1

    x_tensor = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long)
    y_tensor = torch.full((B, max_len), -100, dtype=torch.long)

    user_e_id = tokenizer.user_e_id

    for i, tokens in enumerate(raw_tokens):
        n = len(tokens) - 1
        t_tok = torch.tensor(tokens, dtype=torch.long)
        x_tensor[i, :n] = t_tok[:-1]
        y_tensor[i, :n] = t_tok[1:]

        if mask_prompt:
            matches = (t_tok[:-1] == user_e_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                user_e_pos = matches[0].item()
                y_tensor[i, :user_e_pos + 1] = -100

    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    return x_tensor, y_tensor


def train_tool_gpt(
    model: ToolGPT,
    tokenizer: ToolTokenizer,
    dataset: List[Dict[str, str]],
    steps: int = 1500,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    weight_decay: float = 0.01,
    log_interval: int = 100,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Trains ToolGPT with AdamW, Cosine Annealing, and dynamic sequence batching."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=1e-5)

    n_samples = len(dataset)
    effective_bs = min(batch_size, n_samples)
    indices = list(range(n_samples))
    curr_idx = 0
    epoch_seed = 42

    t0 = time.time()
    total_tokens_processed = 0
    loss_history: List[float] = []

    for step in range(1, steps + 1):
        if curr_idx + effective_bs > n_samples:
            epoch_seed += 1
            rng = random.Random(epoch_seed)
            rng.shuffle(indices)
            curr_idx = 0

        batch_indices = indices[curr_idx : curr_idx + effective_bs]
        curr_idx += effective_bs
        batch_samples = [dataset[i] for i in batch_indices]

        x, y = encode_tool_batch(batch_samples, tokenizer, device=device, mask_prompt=True)
        num_target_tokens = int((y != -100).sum().item())
        total_tokens_processed += num_target_tokens

        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if step % log_interval == 0 or step == steps:
            elapsed = time.time() - t0
            tok_per_sec = total_tokens_processed / max(elapsed, 1e-4)
            lr_curr = optimizer.param_groups[0]["lr"]
            print(
                f"[INFO] Step {step:5d}/{steps:5d} | "
                f"Loss: {loss_val:.4f} | "
                f"Tokens: {total_tokens_processed:,} ({tok_per_sec:.0f} tok/s) | "
                f"LR: {lr_curr:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    return loss_history


def evaluate_tool_benchmarks(
    model: ToolGPT,
    tokenizer: ToolTokenizer,
    device: torch.device,
    n_eval_cases: int = 50,
    seed: int = 999,
) -> Dict[str, Any]:
    """Comprehensive evaluation of tool selection, call syntax, and execution."""
    model.eval()
    rng = random.Random(seed)
    eval_dataset = [generate_tool_example(rng) for _ in range(n_eval_cases)]

    print("\n" + "=" * 70)
    print("TOOLGPT 100K BENCHMARK EVALUATION (TOOL CALLING & ROUTING)")
    print("=" * 70)

    category_stats: Dict[str, Dict[str, int]] = {}
    total_tool_calls_expected = 0
    correct_tool_calls = 0
    correct_no_tools = 0
    total_no_tools_expected = 0

    for i, item in enumerate(eval_dataset):
        cat = item["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "pass": 0}
        category_stats[cat]["total"] += 1

        prompt = f"<USER>{item['user_prompt']}</USER><THINK>"
        tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([tokens], dtype=torch.long, device=device)

        t_start = time.time()
        with torch.no_grad():
            # Stop when either </CALL> or </ANSWER> is reached
            stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
            out = model.generate(x, max_new_tokens=160, temperature=0.0, stop_tokens=stop_ids)
        elapsed = time.time() - t_start

        gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
        expected_call = item["call"]

        is_pass = False
        if expected_call != "none":
            total_tool_calls_expected += 1
            tool_name = expected_call.split("(")[0]
            # Verify tool name and proper call tag
            if f"<CALL>{tool_name}(" in gen_text:
                correct_tool_calls += 1
                is_pass = True
        else:
            total_no_tools_expected += 1
            if "<CALL>" not in gen_text and "<ANSWER>" in gen_text:
                correct_no_tools += 1
                is_pass = True

        status = "[PASS]" if is_pass else "[FAIL]"
        if is_pass:
            category_stats[cat]["pass"] += 1

        if i < 10 or not is_pass:
            print(f"[{status}] Case {i+1:2d} ({cat:12s}): {item['user_prompt'][:40]}")
            if not is_pass:
                print(f"       Expected: {expected_call}")
                print(f"       Gen Trace: {gen_text[len(prompt):120]}...")

    print("\n--- Summary By Category ---")
    total_passed = 0
    total_cases = len(eval_dataset)
    for cat, stats in sorted(category_stats.items()):
        acc = (stats["pass"] / stats["total"]) * 100.0
        total_passed += stats["pass"]
        print(f"  {cat:15s}: {stats['pass']:2d}/{stats['total']:2d} ({acc:5.1f}%)")

    overall_acc = (total_passed / total_cases) * 100.0
    print("\n" + "=" * 70)
    print(f"OVERALL TOOL CALLING ACCURACY: {overall_acc:.2f}% ({total_passed}/{total_cases})")
    print("=" * 70)

    return {
        "overall_acc": overall_acc,
        "total_passed": total_passed,
        "total_cases": total_cases,
        "category_stats": category_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ~100k ToolGPT")
    parser.add_argument("--steps", type=int, default=1500, help="Number of training steps")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of training dialogues")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--n-recur", type=int, default=3, help="Recurrent unrolling passes")
    parser.add_argument("--threads", type=int, default=12, help="CPU threads")
    parser.add_argument("--log-interval", type=int, default=100, help="Log interval")
    parser.add_argument("--pos-emb-type", type=str, default="rope_2d", choices=["rope_2d", "learned_2d"], help="Position embedding type")
    parser.add_argument("--tokenizer-type", type=str, default="char", choices=["char", "bpe_4k"], help="Tokenizer type")
    parser.add_argument("--save-ckpt", type=str, default="", help="Checkpoint path")
    args = parser.parse_args()

    if not args.save_ckpt:
        suffix = ""
        if args.tokenizer_type == "bpe_4k":
            suffix += "_bpe4k"
        if args.pos_emb_type == "learned_2d":
            suffix += "_learned"
        args.save_ckpt = f"tool_model/checkpoints/tool_gpt_100k{suffix}.pt"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.threads)

    if args.tokenizer_type == "bpe_4k":
        from tool_model.bpe_tokenizer import ToolBPETokenizer
        tokenizer = ToolBPETokenizer(vocab_size=4000)
    else:
        tokenizer = ToolTokenizer()

    config = ToolGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_recur=args.n_recur,
        pos_emb_type=args.pos_emb_type,
    )
    model = ToolGPT(config, tokenizer=tokenizer)

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"[INFO] Initialized ToolGPT (~100k Recurrent 2D-Discourse RoPE):")
    print(f"       Total Parameters:          {total_params:,}")
    print(f"       Non-Embedding Parameters:  {non_emb_params:,}")
    print(f"       Layers: {config.n_layer}, Recur: {config.n_recur}, Dim: {config.n_embd}, Heads: {config.n_head}")
    print(f"[INFO] Active Device: {device} (CPU threads={torch.get_num_threads() if device.type == 'cpu' else 'N/A'})")
    model.to(device)

    print(f"\n[INFO] Generating {args.n_samples} synthetic tool calling dialogues...")
    dataset = generate_dataset(n_samples=args.n_samples, seed=42)
    print(f"[INFO] Dataset ready. Sample: {dataset[0]['full_text'][:90]}...")

    print(f"\n[INFO] Starting training for {args.steps} steps...")
    t0 = time.time()
    loss_history = train_tool_gpt(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        device=device,
    )
    t_train = time.time() - t0
    print(f"[INFO] Training completed in {t_train:.1f}s. Final loss: {loss_history[-1]:.4f}")

    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, args.save_ckpt)), exist_ok=True)
    ckpt_path = os.path.join(ROOT_DIR, args.save_ckpt)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "final_loss": loss_history[-1],
        "steps": args.steps,
    }, ckpt_path)
    print(f"[INFO] Saved model checkpoint to {ckpt_path}")

    evaluate_tool_benchmarks(model, tokenizer, device=device)


if __name__ == "__main__":
    main()
