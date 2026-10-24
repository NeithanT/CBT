import os
import sys
import time
import math
import random
import argparse
from typing import Tuple, List, Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.tokenizer import MathTokenizer
from src.model import MathGPT, GPTConfig
from src.arithmetic import format_addition_place_value_thinking, extract_scratchpad_answer
from src.dataset import get_or_generate_shared_math_pairs
from src.distributed import (
    init_distributed_mode,
    cleanup_distributed,
    wrap_model_ddp,
    get_amp_components,
    distributed_barrier,
)
from benchmarks.extrapolation import evaluate_extrapolation_char

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "baselines_18m")
RUN_NAME = "char_1d_rope_placevalue_grpo_12d"
SNAPSHOT_DIR = os.path.join(CHECKPOINT_DIR, "snapshots", RUN_NAME)


def compute_reward(prompt: str, expected_ans: str, full_gen: str) -> Tuple[float, bool, int]:
    reasoning, pred_ans = extract_scratchpad_answer(full_gen)
    is_correct = (pred_ans == expected_ans)

    has_think_tags = ("<THINK>" in full_gen and "</THINK>" in full_gen)
    num_think_lines = len([line for line in reasoning.split("\n") if line.strip()])

    reward = 0.0

    if is_correct:
        reward += 10.0
    else:
        reward -= 10.0

    if has_think_tags:
        reward += 2.0
        if num_think_lines >= 2:
            reward += min(3.0, 0.5 * num_think_lines)
    else:
        reward -= 5.0

    if not is_correct and num_think_lines == 0:
        reward -= 5.0

    return reward, is_correct, num_think_lines


def sample_rollouts(
    model: MathGPT,
    prompt_ids: List[int],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    tokenizer: MathTokenizer,
    device: torch.device,
) -> List[Dict[str, Any]]:
    rollouts = []
    prefix = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(group_size):
        curr = prefix.clone()
        gen_tokens = []
        gen_logprobs = []

        for _ in range(max_new_tokens):
            if curr.size(1) >= model.config.block_size:
                break
            with torch.no_grad():
                logits, _ = model(curr)
            next_logits = logits[0, -1, :]

            if temperature <= 0.0:
                next_tok = torch.argmax(next_logits).item()
                log_prob = F.log_softmax(next_logits, dim=-1)[next_tok].item()
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
                log_prob = torch.log(probs[next_tok] + 1e-12).item()

            gen_tokens.append(next_tok)
            gen_logprobs.append(log_prob)
            curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)

            if next_tok == tokenizer.eos_id:
                break

        full_text = tokenizer.decode(curr[0].tolist(), skip_special=False)
        rollouts.append({
            "tokens": gen_tokens,
            "logprobs": gen_logprobs,
            "full_text": full_text,
        })
    return rollouts


def train_model4_rl(args):
    device, rank, local_rank, world_size, is_master = init_distributed_mode(args)

    if is_master:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        print("=" * 75)
        print("TRAINING MODEL 4: Place-Value Zero-Reduction Thinking with Reinforcement Learning")
        print(f"World Size: {world_size} | Device: {device} | Precision: FP16 AMP")
        print(f"SFT Warmup Steps: {args.sft_steps} | RL Steps: {args.rl_steps} | Group Size: {args.group_size}")
        print("=" * 75)

    tokenizer = MathTokenizer()
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        pos_emb_type="rope",
    )

    policy_model = MathGPT(config).to(device)
    ref_model = MathGPT(config).to(device)

    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        policy_model.load_state_dict(ckpt.get("model_state", ckpt))
        if is_master:
            print(f"Loaded initial checkpoint from {args.init_checkpoint}")

    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.rl_steps + args.sft_steps), eta_min=args.lr * 0.1)
    AutocastCtx, scaler = get_amp_components(device, enabled=args.use_amp)

    if is_master:
        get_or_generate_shared_math_pairs()
    distributed_barrier()

    a_all, b_all = get_or_generate_shared_math_pairs()
    total_equations = len(a_all)

    # Phase 1: SFT Warmup (if requested)
    if args.sft_steps > 0:
        if is_master:
            print(f"\n[PHASE 1: SFT WARMUP] Training {args.sft_steps} steps on place-value format...")
        policy_model.train()
        for sft_step in range(1, args.sft_steps + 1):
            batch_idx = (sft_step * args.batch_size) % (total_equations - args.batch_size)
            a_sub = a_all[batch_idx : batch_idx + args.batch_size].tolist()
            b_sub = b_all[batch_idx : batch_idx + args.batch_size].tolist()
            texts = [format_addition_place_value_thinking(a, b) for a, b in zip(a_sub, b_sub)]

            x_list, y_list = [], []
            for txt in texts:
                toks = tokenizer.encode(txt, add_bos=True, add_eos=True)
                toks = toks[: config.block_size + 1]
                x = toks[:-1]
                y = toks[1:]
                pad_len = config.block_size - len(x)
                if pad_len > 0:
                    x = x + [tokenizer.pad_id] * pad_len
                    y = y + [-100] * pad_len
                x_list.append(torch.tensor(x[: config.block_size], dtype=torch.long))
                y_list.append(torch.tensor(y[: config.block_size], dtype=torch.long))

            x_t = torch.stack(x_list).to(device)
            y_t = torch.stack(y_list).to(device)

            optimizer.zero_grad(set_to_none=True)
            with AutocastCtx():
                _, loss = policy_model(x_t, y_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if is_master and (sft_step % args.log_interval == 0 or sft_step == args.sft_steps):
                print(f"[SFT Step {sft_step:4d}/{args.sft_steps:4d}] Loss: {loss.item():.4f}")

        ref_model.load_state_dict(policy_model.state_dict())
        if is_master:
            print("[SFT Warmup Completed. Reference policy updated.]\n")

    # Phase 2: Reinforcement Learning (GRPO)
    if is_master:
        print(f"[PHASE 2: GRPO REINFORCEMENT LEARNING] Pushing thinking for {args.rl_steps} steps...")

    policy_model.train()
    for rl_step in range(1, args.rl_steps + 1):
        prompt_batch_idx = (rl_step * args.batch_size) % (total_equations - args.batch_size)
        a_sub = a_all[prompt_batch_idx : prompt_batch_idx + args.batch_size].tolist()
        b_sub = b_all[prompt_batch_idx : prompt_batch_idx + args.batch_size].tolist()

        all_advantages = []
        all_loss = 0.0
        step_correct = 0
        step_total_rollouts = 0
        step_think_lines = 0

        for a, b in zip(a_sub, b_sub):
            prompt = f"{a}+{b}="
            expected_ans = str(a + b)
            prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)

            policy_model.eval()
            rollouts = sample_rollouts(
                policy_model,
                prompt_ids,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                tokenizer=tokenizer,
                device=device,
            )
            policy_model.train()

            rewards = []
            for r in rollouts:
                rew, is_corr, n_lines = compute_reward(prompt, expected_ans, r["full_text"])
                rewards.append(rew)
                if is_corr:
                    step_correct += 1
                step_think_lines += n_lines
                step_total_rollouts += 1

            r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            r_mean = r_tensor.mean()
            r_std = r_tensor.std() + 1e-5
            advantages = (r_tensor - r_mean) / r_std

            for i, r in enumerate(rollouts):
                gen_toks = r["tokens"]
                if not gen_toks:
                    continue
                adv = advantages[i].item()

                seq_ids = prompt_ids + gen_toks
                seq_t = torch.tensor([seq_ids], dtype=torch.long, device=device)

                with AutocastCtx():
                    logits, _ = policy_model(seq_t)
                prompt_len = len(prompt_ids)
                gen_logits = logits[0, prompt_len - 1 : -1, :]
                gen_targets = torch.tensor(gen_toks, dtype=torch.long, device=device)

                curr_logprobs = -F.cross_entropy(gen_logits, gen_targets, reduction="none")

                with torch.no_grad():
                    ref_logits, _ = ref_model(seq_t)
                    ref_gen_logits = ref_logits[0, prompt_len - 1 : -1, :]
                    ref_logprobs = -F.cross_entropy(ref_gen_logits, gen_targets, reduction="none")

                ratio = torch.exp(curr_logprobs - ref_logprobs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                kl_div = (torch.exp(ref_logprobs) * (ref_logprobs - curr_logprobs)).mean()
                loss = policy_loss + args.beta_kl * kl_div

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                all_loss += loss.item()

        scheduler.step()

        if is_master and (rl_step % args.log_interval == 0 or rl_step == args.rl_steps):
            acc = (step_correct / max(1, step_total_rollouts)) * 100.0
            avg_think = step_think_lines / max(1, step_total_rollouts)
            print(
                f"[RL Step {rl_step:4d}/{args.rl_steps:4d}] "
                f"Loss: {all_loss / max(1, step_total_rollouts):.4f} | "
                f"Accuracy: {acc:5.1f}% | "
                f"Avg Think Lines: {avg_think:.1f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if is_master and rl_step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(SNAPSHOT_DIR, f"grpo_step{rl_step}.pt")
            torch.save({
                "step": rl_step,
                "model_state": policy_model.state_dict(),
                "config": config,
                "condition": 4,
            }, ckpt_path)
            print(f"[CHECKPOINT] Saved: {ckpt_path}")

    if is_master:
        final_ckpt = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}.pt")
        torch.save({
            "model_state": policy_model.state_dict(),
            "config": config,
            "condition": 4,
        }, final_ckpt)
        print(f"\nSaved Final RL Checkpoint: {final_ckpt}")

        print("\nEvaluating Model 4 on Extrapolation Benchmark...")
        evaluate_extrapolation_char(
            policy_model,
            tokenizer,
            device,
            title="Model 4 (Place-Value RL)",
            save_results=True,
        )

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train Model 4: Place-Value Zero-Reduction with RL")
    parser.add_argument("--sft_steps", type=int, default=100, help="SFT warmup steps")
    parser.add_argument("--rl_steps", type=int, default=500, help="GRPO RL steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Equations per step")
    parser.add_argument("--group_size", type=int, default=4, help="Rollouts per equation (GRPO)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Rollout sampling temperature")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO/GRPO clipping epsilon")
    parser.add_argument("--beta_kl", type=float, default=0.04, help="KL divergence penalty coefficient")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer layers")
    parser.add_argument("--n_head", type=int, default=8, help="Attention heads")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable FP16 AMP")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable AMP")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval in steps")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Checkpoint interval in steps")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Initial model weights path")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    train_model4_rl(args)


if __name__ == "__main__":
    main()
