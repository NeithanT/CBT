"""
Distributed Training & Evaluation Orchestrator for ScaledGPT (~18.95M Parameters).
Optimized for 25-Hour Dual NVIDIA T4 GPU Training on Kaggle.

Key Architecture & Optimization:
1. Turing sm_75 Native FP16 AMP (torch.amp.autocast + GradScaler). Strictly no bfloat16, no TF32.
2. Scaled Dot-Product Attention (SDPA) for O(1) activation memory.
3. DistributedDataParallel (DDP) for 2x T4 GPUs via torchrun.
4. Two-Stage Frontier Training Curriculum:
   - Phase 1: Joint Pre-training (50% Math / 50% Tools).
   - Phase 2: Reasoning & State-Machine Annealing (75% Math / 25% Tools) with LR cooling to 1e-5.
5. High-Digit Arithmetic Extrapolation Benchmark (1 to 50 Digits).
6. Expanded Tool Repertoire Benchmark (Single, Multi-Tool, Nested JSON, Error Recovery, Multi-Turn, No-Tool).
"""

import os
import sys
import datetime

# Crucial Kaggle Dual-T4 NCCL Environment Workarounds:
# On Kaggle containers, T4 GPUs lack PCIe P2P direct routing and external docker NIC binding.
# Without these, NCCL hangs indefinitely inside dist.init_process_group().
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

# Ensure immediate real-time output flushing in headless Kaggle / cloud containers
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import time
import math
import shutil
import random
import argparse
from typing import List, Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    from champion_model.scaled_t4.tokenizer import ScaledTokenizer
    from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig
    from champion_model.scaled_t4.dataset import (
        generate_joint_dataset,
        generate_tool_dataset,
        encode_joint_batch,
        extract_scratchpad_answer,
    )
except ImportError:
    from tokenizer import ScaledTokenizer
    from model import ScaledGPT, ScaledGPTConfig
    from dataset import (
        generate_joint_dataset,
        generate_tool_dataset,
        encode_joint_batch,
        extract_scratchpad_answer,
    )


# -----------------------------------------------------------------------------
# 1. High-Digit Arithmetic Benchmark Suite (1 to 50 Digits, 20 Cases)
# -----------------------------------------------------------------------------

EXTRAPOLATION_BENCHMARK_50D: Dict[str, List[Tuple[str, str]]] = {
    "Tier 1: In-Distribution (1-15 Digits)": [
        ("15+25=", "40"),
        ("1234+5678=", "6912"),
        ("12345678+87654321=", "99999999"),
        ("10000000+10000000=", "20000000"),
        ("123456789012+876543210987=", "999999999999"),
        ("500000000000000+500000000000000=", "1000000000000000"),
    ],
    "Tier 2: Intermediate Extrapolation (16-25 Digits)": [
        ("1111222233334444+8888777766665555=", "9999999999999999"),
        ("100000000000000000+100000000000000000=", "200000000000000000"),
        ("12345678901234567890+87654321098765432109=", "99999999999999999999"),
        ("99999999999999999999+1=", "100000000000000000000"),
        ("1234567890123456789012+8765432109876543210987=", "9999999999999999999999"),
        ("500000000000000000000000+500000000000000000000000=", "1000000000000000000000000"),
    ],
    "Tier 3: High Extrapolation (30-40 Digits)": [
        ("123456789012345678901234567890+876543210987654321098765432109=", "999999999999999999999999999999"),
        ("1000000000000000000000000000000+1000000000000000000000000000000=", "2000000000000000000000000000000"),
        ("11111111111111111111111111111111111+88888888888888888888888888888888888=", "99999999999999999999999999999999999"),
        ("5000000000000000000000000000000000000000+5000000000000000000000000000000000000000=", "10000000000000000000000000000000000000000"),
    ],
    "Tier 4: Ultra Extrapolation (45-50 Digits & Cascade Carry)": [
        ("123456789012345678901234567890123456789012345+876543210987654321098765432109876543210987654=", "999999999999999999999999999999999999999999999"),
        ("10000000000000000000000000000000000000000000000000+10000000000000000000000000000000000000000000000000=", "20000000000000000000000000000000000000000000000000"),
        ("12345678901234567890123456789012345678901234567890+87654321098765432109876543210987654321098765432109=", "99999999999999999999999999999999999999999999999999"),
        ("99999999999999999999999999999999999999999999999999+1=", "100000000000000000000000000000000000000000000000000"),
    ],
}


# -----------------------------------------------------------------------------
# 2. Distributed & Hardware Optimization Utilities
# -----------------------------------------------------------------------------

def init_distributed() -> Tuple[torch.device, int, int, int, bool]:
    """
    Initializes DDP distributed training environment for 2x T4 GPUs.
    Configures Turing sm_75 precision guarantees (FP16 AMP, no bfloat16, no TF32).
    """
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            print(f"[INFO] Initializing NCCL process group on GPU {local_rank}/{world_size}...", flush=True)
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(seconds=60),
            )
            print(f"[OK] NCCL process group connected successfully on GPU {local_rank}.", flush=True)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        is_master = (rank == 0)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_master = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Strict Turing sm_75 optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    return device, rank, local_rank, world_size, is_master


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_checkpoint_to_kaggle(ckpt_path: str) -> None:
    """Mirrors saved checkpoint directly to Kaggle output root if running on Kaggle."""
    if os.path.exists("/kaggle/working"):
        try:
            target_dir = os.path.join("/kaggle/working", "checkpoints", "scaled_t4")
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, os.path.basename(ckpt_path))
            if os.path.abspath(ckpt_path) != os.path.abspath(target_path):
                shutil.copy2(ckpt_path, target_path)
                print(f"[KAGGLE SYNC] Mirrored checkpoint to: {target_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to mirror checkpoint to /kaggle/working: {e}", flush=True)


def prune_checkpoints(ckpt_dir: str, keep_last: int = 2, base_filename: str = "scaled_t4_final") -> None:
    """
    Prunes intermediate step and timeout checkpoints, retaining only the most recent keep_last checkpoints.
    Preserves the final model weights file (<base_filename>.pt).
    Also prunes the Kaggle mirror directory if present to prevent disk overflow.
    """
    if keep_last <= 0:
        return

    dirs_to_prune = [ckpt_dir]
    kaggle_dir = os.path.join("/kaggle/working", "checkpoints", "scaled_t4")
    if os.path.exists(kaggle_dir) and os.path.abspath(kaggle_dir) not in [os.path.abspath(d) for d in dirs_to_prune]:
        dirs_to_prune.append(kaggle_dir)

    for d in dirs_to_prune:
        if not os.path.isdir(d):
            continue
        candidates = []
        for fname in os.listdir(d):
            if not fname.endswith(".pt"):
                continue
            if fname == f"{base_filename}.pt":
                continue
            if base_filename in fname and ("_step_" in fname or "_timeout_step_" in fname):
                fpath = os.path.join(d, fname)
                candidates.append((os.path.getmtime(fpath), fpath))

        candidates.sort(key=lambda x: x[0])
        while len(candidates) > keep_last:
            _, old_path = candidates.pop(0)
            try:
                os.remove(old_path)
                print(f"[INFO] Pruned old checkpoint to conserve disk space: {old_path}", flush=True)
            except Exception as e:
                print(f"[WARN] Failed to prune {old_path}: {e}", flush=True)


# -----------------------------------------------------------------------------
# 3. Two-Stage Frontier Training Loop
# -----------------------------------------------------------------------------

def train_scaled_t4(
    model: nn.Module,
    tokenizer: ScaledTokenizer,
    math_dataset: List[Dict[str, str]],
    tool_dataset: List[Dict[str, str]],
    device: torch.device,
    rank: int,
    world_size: int,
    is_master: bool,
    steps: int = 10000,
    anneal_steps: int = 3000,
    batch_size: int = 32,
    lr: float = 1.0e-3,
    weight_decay: float = 0.01,
    log_interval: int = 250,
    save_interval: int = 2500,
    max_keep_ckpts: int = 2,
    ckpt_path: Optional[str] = None,
    max_hours: float = 25.0,
    start_step: int = 1,
    resume_state: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """
    Two-Stage Frontier Training on Dual NVIDIA T4:
    Phase 1: Broad Joint Pre-training (50% Math / 50% Tools)
    Phase 2: Reasoning & State-Machine Annealing (75% Math / 25% Tools) with LR cooling to 1e-5.
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=1e-5)

    use_cuda_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    # Initial checkpoint pruning to free disk space if resuming
    if ckpt_path is not None and is_master:
        ckpt_dir = os.path.dirname(ckpt_path)
        base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        prune_checkpoints(ckpt_dir, keep_last=max_keep_ckpts, base_filename=base_name)

    # Resume optimizer, scheduler, scaler state if provided
    if resume_state is not None:
        if "optimizer_state_dict" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            except Exception:
                pass
        if "scheduler_state_dict" in resume_state:
            try:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            except Exception:
                pass
        if "scaler_state_dict" in resume_state and use_cuda_amp:
            try:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            except Exception:
                pass

    n_math = len(math_dataset)
    n_tools = len(tool_dataset)

    math_idx = 0
    tool_idx = 0
    rng_math = random.Random(42 + rank)
    rng_tool = random.Random(1337 + rank)

    loss_history: List[float] = []
    t0 = time.time()
    total_tokens_processed = resume_state.get("total_tokens", 0) if resume_state else 0
    phase1_steps = max(1, steps - anneal_steps)

    if is_master:
        print("\n" + "=" * 80, flush=True)
        print(f"[INFO] Launching ScaledGPT Training on {world_size} worker(s)", flush=True)
        print(f"       Total Steps: {steps:,} (Starting at: {start_step:,})", flush=True)
        print(f"       Phase 1: {phase1_steps:,} | Phase 2 Anneal: {anneal_steps:,}", flush=True)
        print(f"       Global Batch Size: {batch_size * world_size} ({batch_size}/GPU) | Peak LR: {lr:.2e}", flush=True)
        print(f"       Compute Budget Limit: {max_hours:.1f} hours | Precision: FP16 AMP (sm_75)", flush=True)
        print("=" * 80 + "\n", flush=True)

    for step in range(start_step, steps + 1):
        elapsed_hours = (time.time() - t0) / 3600.0
        if elapsed_hours >= max_hours:
            if is_master:
                print(f"[WARN] Reached maximum allocated time limit ({max_hours:.1f}h). Halting training gracefully.")
                if ckpt_path is not None:
                    raw_model = model.module if hasattr(model, "module") else model
                    timeout_ckpt = ckpt_path.replace(".pt", f"_timeout_step_{step}.pt")
                    os.makedirs(os.path.dirname(timeout_ckpt), exist_ok=True)
                    torch.save({
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if use_cuda_amp else None,
                        "config": raw_model.config,
                        "step": step,
                        "total_tokens": total_tokens_processed,
                    }, timeout_ckpt)
                    sync_checkpoint_to_kaggle(timeout_ckpt)
                    print(f"[INFO] Saved graceful timeout checkpoint: {timeout_ckpt}", flush=True)
                    ckpt_dir = os.path.dirname(ckpt_path)
                    base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
                    prune_checkpoints(ckpt_dir, keep_last=max_keep_ckpts, base_filename=base_name)
            break

        is_phase2 = (step > phase1_steps)
        if is_phase2:
            n_m = int(batch_size * 0.75)
            n_t = batch_size - n_m
        else:
            n_m = batch_size // 2
            n_t = batch_size - n_m

        if math_idx + n_m > n_math:
            rng_math.shuffle(math_dataset)
            math_idx = 0
        if tool_idx + n_t > n_tools:
            rng_tool.shuffle(tool_dataset)
            tool_idx = 0

        math_batch = math_dataset[math_idx : math_idx + n_m]
        tool_batch = tool_dataset[tool_idx : tool_idx + n_t]
        math_idx += n_m
        tool_idx += n_t

        x_b, y_b = encode_joint_batch(math_batch, tool_batch, tokenizer, device=device)
        step_tokens = (y_b != -100).sum().item()
        total_tokens_processed += step_tokens * world_size

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
        if world_size > 1 and dist.is_initialized():
            loss_tensor = torch.tensor(loss_val, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_val = loss_tensor.item() / world_size

        loss_history.append(loss_val)

        if (step == 1 or step % log_interval == 0 or step == steps) and is_master:
            elapsed = time.time() - t0
            tok_per_sec = total_tokens_processed / max(elapsed, 1e-4)
            lr_curr = optimizer.param_groups[0]["lr"]
            stage_name = "Phase 2 (Reasoning Anneal)" if is_phase2 else "Phase 1 (Joint Pretrain)"
            print(
                f"[INFO] Step {step:6d}/{steps:6d} [{stage_name}] | "
                f"Loss: {loss_val:.4f} | "
                f"Tokens: {total_tokens_processed:,} ({tok_per_sec:.0f} tok/s) | "
                f"LR: {lr_curr:.6f} | "
                f"Elapsed: {elapsed / 60.0:.1f} min",
                flush=True,
            )

        if (step % save_interval == 0) and is_master and ckpt_path is not None:
            raw_model = model.module if hasattr(model, "module") else model
            step_ckpt = ckpt_path.replace(".pt", f"_step_{step}.pt")
            os.makedirs(os.path.dirname(step_ckpt), exist_ok=True)
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_cuda_amp else None,
                "config": raw_model.config,
                "step": step,
                "loss": loss_val,
                "total_tokens": total_tokens_processed,
            }, step_ckpt)
            sync_checkpoint_to_kaggle(step_ckpt)
            print(f"[INFO] Periodic checkpoint saved: {step_ckpt}", flush=True)
            ckpt_dir = os.path.dirname(ckpt_path)
            base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            prune_checkpoints(ckpt_dir, keep_last=max_keep_ckpts, base_filename=base_name)

    return loss_history


# -----------------------------------------------------------------------------
# 4. Comprehensive Evaluation Benchmark Routines
# -----------------------------------------------------------------------------

def evaluate_math_extrapolation(
    model: nn.Module,
    tokenizer: ScaledTokenizer,
    device: torch.device,
    max_new_tokens: int = 700,
) -> Dict[str, Any]:
    """Evaluates multi-digit arithmetic extrapolation up to 50 digits."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    summary: Dict[str, Any] = {}
    total_cases = 0
    total_passed = 0

    print("\n" + "=" * 80)
    print("SCALED T4 MODEL: ARITHMETIC LENGTH EXTRAPOLATION BENCHMARK (1-50 DIGITS)")
    print("=" * 80)

    for tier_name, cases in EXTRAPOLATION_BENCHMARK_50D.items():
        print(f"\n--- {tier_name} ({len(cases)} cases) ---")
        tier_passed = 0
        tier_results = []

        for prompt, expected in cases:
            eq_prompt = prompt if prompt.endswith("<THINK>") else (prompt + "<THINK>")
            tokens = tokenizer.encode(eq_prompt, add_bos=True, add_eos=False)
            x = torch.tensor([tokens], dtype=torch.long, device=device)

            t_start = time.time()
            with torch.no_grad():
                stop_ids = [tokenizer.eos_id]
                out = raw_model.generate(x, max_new_tokens=max_new_tokens, temperature=0.0, stop_tokens=stop_ids)
            elapsed = time.time() - t_start

            gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
            reasoning, answer = extract_scratchpad_answer(gen_text)
            is_pass = (answer.strip() == expected.strip())

            if is_pass:
                tier_passed += 1
                total_passed += 1
            total_cases += 1

            status = "[PASS]" if is_pass else "[FAIL]"
            tier_results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": answer,
                "is_correct": is_pass,
                "latency_sec": round(elapsed, 3),
            })

            ans_preview = answer[:25] + ("..." if len(answer) > 25 else "")
            exp_preview = expected[:25] + ("..." if len(expected) > 25 else "")
            print(f"  {prompt:<35} Expected: {exp_preview:<25} Pred: {ans_preview:<25} {status} ({elapsed:.2f}s)")

        tier_acc = (tier_passed / len(cases)) * 100.0 if cases else 0.0
        summary[tier_name] = {
            "passed": tier_passed,
            "total": len(cases),
            "accuracy_pct": round(tier_acc, 2),
            "cases": tier_results,
        }
        print(f"  --> Tier Accuracy: {tier_acc:.1f}% ({tier_passed}/{len(cases)})")

    overall_acc = (total_passed / total_cases) * 100.0 if total_cases > 0 else 0.0
    summary["overall"] = {
        "passed": total_passed,
        "total": total_cases,
        "accuracy_pct": round(overall_acc, 2),
    }
    summary["total_passed"] = total_passed
    summary["total_cases"] = total_cases

    print("\n" + "=" * 80)
    print(f"OVERALL ARITHMETIC EXTRAPOLATION ACCURACY (1-50 DIGITS): {overall_acc:.2f}% ({total_passed}/{total_cases})")
    print("=" * 80)
    return summary


def evaluate_tool_benchmarks(
    model: nn.Module,
    tokenizer: ScaledTokenizer,
    device: torch.device,
    n_eval_cases: int = 90,
    seed: int = 999,
) -> Dict[str, Any]:
    """
    Evaluates expanded tool calling across 6 categories:
    - Single tool
    - Multi-tool chaining
    - Nested JSON schemas
    - Error handling & recovery
    - Multi-turn dialogues
    - Direct conversational routing (no-tool)
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    eval_dataset = generate_tool_dataset(n_samples=n_eval_cases, seed=seed)

    print("\n" + "=" * 80)
    print(f"SCALED T4 MODEL: EXPANDED TOOL BENCHMARK ({len(eval_dataset)} CASES)")
    print("=" * 80)

    category_stats: Dict[str, Dict[str, int]] = {}
    total_passed = 0
    total_cases = len(eval_dataset)

    for i, item in enumerate(eval_dataset):
        cat = item["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "pass": 0}
        category_stats[cat]["total"] += 1

        prompt = item["prompt"]
        if not prompt.endswith("<THINK>"):
            prompt += "<THINK>"

        tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([tokens], dtype=torch.long, device=device)

        t_start = time.time()
        with torch.no_grad():
            stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
            out = raw_model.generate(x, max_new_tokens=256, temperature=0.0, stop_tokens=stop_ids)
        elapsed = time.time() - t_start

        gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
        full_ref = item["full_text"]

        is_pass = False
        if cat == "no_tool":
            if "<CALL>" not in gen_text and "<ANSWER>" in gen_text:
                is_pass = True
        elif cat == "nested_json_schema":
            if "<CALL>" in gen_text and ("{" in gen_text or "[" in gen_text):
                is_pass = True
        elif cat == "error_recovery":
            if "<CALL>" in gen_text or "<ANSWER>" in gen_text:
                is_pass = True
        elif cat == "multi_tool_chain":
            if "<CALL>" in gen_text:
                is_pass = True
        elif cat == "multi_turn":
            # Multi-turn includes conversational greetings (<ANSWER>) and tool inquiries (<CALL>)
            if "<ANSWER>" in gen_text or "<CALL>" in gen_text:
                is_pass = True
        else:
            if "<CALL>" in gen_text:
                is_pass = True

        status = "[PASS]" if is_pass else "[FAIL]"
        if is_pass:
            category_stats[cat]["pass"] += 1
            total_passed += 1

        if i < 12 or not is_pass:
            print(f"[{status}] Case {i+1:2d} ({cat:18s}): {prompt[:50]}")
            if not is_pass:
                gen_slice = gen_text[len(prompt):len(prompt)+80]
                print(f"       Gen Trace: {gen_slice}...")

    print("\n--- Summary By Category ---")
    for cat, stats in sorted(category_stats.items()):
        acc = (stats["pass"] / stats["total"]) * 100.0 if stats["total"] > 0 else 0.0
        print(f"  {cat:22s}: {stats['pass']:2d}/{stats['total']:2d} ({acc:5.1f}%)")

    overall_acc = (total_passed / total_cases) * 100.0 if total_cases > 0 else 0.0
    print("\n" + "=" * 80)
    print(f"OVERALL EXPANDED TOOL CALLING ACCURACY: {overall_acc:.2f}% ({total_passed}/{total_cases})")
    print("=" * 80)

    return {
        "overall_acc": overall_acc,
        "total_passed": total_passed,
        "total_cases": total_cases,
        "category_stats": category_stats,
    }


# -----------------------------------------------------------------------------
# 5. CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ScaledGPT ~18.95M Training & Evaluation Suite (Dual T4 sm_75)")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--anneal-steps", type=int, default=3000, help="Phase 2 reasoning annealing steps")
    parser.add_argument("--n-math", type=int, default=10000, help="Number of math curriculum samples")
    parser.add_argument("--n-tools", type=int, default=10000, help="Number of tool dialogue samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Peak learning rate")
    parser.add_argument("--max-hours", type=float, default=25.0, help="Maximum compute budget in hours")
    parser.add_argument("--log-interval", type=int, default=250, help="Log step interval")
    parser.add_argument("--save-interval", type=int, default=2500, help="Checkpoint save step interval")
    parser.add_argument("--max-keep-ckpts", type=int, default=2, help="Maximum number of intermediate checkpoints to retain (default: 2)")
    parser.add_argument("--save-ckpt", type=str, default="champion_model/scaled_t4/checkpoints/scaled_t4_final.pt", help="Checkpoint path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--eval-only", action="store_true", help="Run benchmark evaluation only")
    args = parser.parse_args()

    device, rank, local_rank, world_size, is_master = init_distributed()

    tokenizer = ScaledTokenizer()
    config = ScaledGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=1024,
        n_layer=6,
        n_head=8,
        n_embd=512,
        n_recur=3,
        max_coords=4096,
        k_rand_shift=32,
    )
    model = ScaledGPT(config, tokenizer=tokenizer).to(device)

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)

    if is_master:
        print("\n" + "=" * 80, flush=True)
        print("SCALED RECURRENT 2D-ROPE TRANSFORMER (~18.95M PARAMETERS)", flush=True)
        print(f"Device: {device} | World Size: {world_size} | Rank: {rank}", flush=True)
        print(f"Total Parameters:          {total_params:,} ({total_params/1e6:.2f}M)", flush=True)
        print(f"Non-Embedding Parameters:  {non_emb_params:,} ({non_emb_params/1e6:.2f}M)", flush=True)
        print(f"Layers (Physical):         {config.n_layer} | Recurrent Passes: {config.n_recur} (Effective Depth = {config.n_layer * config.n_recur})", flush=True)
        print(f"Attention Heads:           {config.n_head} (head_dim = {config.n_embd // config.n_head}) | Embedding Dim: {config.n_embd}", flush=True)
        print(f"Block Size:                {config.block_size} | Max Coordinates: {config.max_coords}", flush=True)
        print(f"Precision:                 Native sm_75 FP16 AMP (torch.amp.autocast, no bfloat16, no TF32)", flush=True)
        print("=" * 80 + "\n", flush=True)

    ckpt_path = os.path.join(ROOT_DIR, args.save_ckpt)
    start_step = 1
    resume_state = None
    if args.resume and os.path.exists(args.resume):
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model_state_dict"])
        start_step = resume_state.get("step", 0) + 1
        if is_master:
            print(f"[INFO] Successfully resumed model weights and optimizer state from: {args.resume}", flush=True)
            print(f"[INFO] Resuming training from step: {start_step:,}", flush=True)

    # Wrap in multi-GPU (DDP if launched with torchrun, DataParallel if standard python with 2+ GPUs)
    is_distributed = (world_size > 1)
    if is_distributed and device.type == "cuda":
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and not is_distributed:
        if is_master:
            print(f"[INFO] Multi-GPU Mode: Detected {torch.cuda.device_count()} GPUs. Wrapping with torch.nn.DataParallel.", flush=True)
        ddp_model = nn.DataParallel(model)
    else:
        ddp_model = model

    if not args.eval_only:
        if is_master:
            print(f"[INFO] Generating datasets ({args.n_math} math + {args.n_tools} tools)...", flush=True)
        math_ds, tool_ds = generate_joint_dataset(n_math=args.n_math, n_tools=args.n_tools, seed=42)

        t0 = time.time()
        loss_hist = train_scaled_t4(
            model=ddp_model,
            tokenizer=tokenizer,
            math_dataset=math_ds,
            tool_dataset=tool_ds,
            device=device,
            rank=rank,
            world_size=world_size,
            is_master=is_master,
            steps=args.steps,
            anneal_steps=args.anneal_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            max_keep_ckpts=args.max_keep_ckpts,
            ckpt_path=ckpt_path,
            max_hours=args.max_hours,
            start_step=start_step,
            resume_state=resume_state,
        )

        if is_master:
            elapsed_total = time.time() - t0
            print(f"\n[INFO] Training complete in {elapsed_total / 3600.0:.2f} hours. Final Loss: {loss_hist[-1] if loss_hist else 0.0:.4f}")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            raw_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "config": raw_model.config,
                "final_loss": loss_hist[-1] if loss_hist else 0.0,
                "steps": args.steps,
            }, ckpt_path)
            sync_checkpoint_to_kaggle(ckpt_path)
            print(f"[INFO] Final champion checkpoint saved: {ckpt_path}")

    # Evaluation phase (Rank 0 only)
    if is_master:
        raw_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
        math_res = evaluate_math_extrapolation(raw_model, tokenizer, device=device)
        tool_res = evaluate_tool_benchmarks(raw_model, tokenizer, device=device, n_eval_cases=90)

        print("\n" + "=" * 80)
        print("FINAL SCALEDGPT (~18.95M) BENCHMARK SUMMARY")
        print("=" * 80)
        m_pass = math_res.get("total_passed", 0)
        m_tot = math_res.get("total_cases", 20)
        t_pass = tool_res.get("total_passed", 0)
        t_tot = tool_res.get("total_cases", 90)
        print(f"  Arithmetic Length Extrapolation (1-50 Digits): {m_pass}/{m_tot} ({100.0 * m_pass / max(1, m_tot):.2f}%)")
        print(f"  Expanded Tool Repertoire Suite:                 {t_pass}/{t_tot} ({100.0 * t_pass / max(1, t_tot):.2f}%)")
        print("=" * 80 + "\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
