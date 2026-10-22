import os
import sys
import time
import math
import random
import queue
import threading
import argparse
import contextlib
import shutil
from typing import Tuple, List, Optional, Any, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.tokenizer import MathTokenizer
from src.bpe_tokenizer import BPEMathTokenizer
from src.model import MathGPT, GPTConfig, LoopedMathGPT, LoopedGPTConfig, HierarchicalReasoningGPT, HierarchicalGPTConfig, create_model
from src.models.hybrid import HybridConfig, HybridMathGPT
from src.models.alsm import ALSMConfig, AbacusLoopedMathGPT
from src.models.hnrm import HNRMConfig, HarmonicNRM
from src.models.micromath import MicroMathConfig, MicroMath2M
from src.models.hrm_rope import HRMRoPEConfig, HRMRoPE
from src.models.hrm_text import HRMTextConfig, HRMTextModel
from src.arithmetic import format_addition_char_scratchpad, format_addition_bpe_word_cot, format_addition_place_value_thinking
from src.dataset import get_or_generate_shared_math_pairs, load_philosophy_token_stream, compute_abacus_place_ids
from src.distributed import (
    init_distributed_mode,
    cleanup_distributed,
    wrap_model_ddp,
    get_amp_components,
    distributed_barrier,
)
from benchmarks.extrapolation import evaluate_extrapolation_char, evaluate_extrapolation_bpe

DEFAULT_TARGET_TOKENS = 400_000_000


def sync_checkpoint_to_kaggle(ckpt_path: str, model_type: str):
    """If running inside a Kaggle notebook, mirror checkpoint directly to /kaggle/working/checkpoints/."""
    if os.path.exists("/kaggle/working"):
        try:
            target_dir = os.path.join("/kaggle/working", "checkpoints", model_type)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, os.path.basename(ckpt_path))
            if os.path.abspath(ckpt_path) != os.path.abspath(target_path):
                shutil.copy2(ckpt_path, target_path)
                print(f"[KAGGLE SYNC] Mirrored checkpoint to {target_path}")
        except Exception as e:
            print(f"[WARN] Failed to mirror checkpoint to /kaggle/working: {e}")


CONDITION_METADATA: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Model 1: Character Math Tokenizer (100% Math with Scratchpad)",
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "model_1"),
        "ckpt_prefix": "model_1",
        "tokenizer_cls": MathTokenizer,
        "is_bpe": False,
        "eval_fn": evaluate_extrapolation_char,
    },
    2: {
        "name": "Model 2: Subword BPE Tokenizer (100% Math with Word-CoT)",
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "model_2"),
        "ckpt_prefix": "model_2",
        "tokenizer_cls": BPEMathTokenizer,
        "is_bpe": True,
        "eval_fn": evaluate_extrapolation_bpe,
    },
    3: {
        "name": "Model 3: Subword BPE (50% Math Word-CoT + 50% Philosophy Text)",
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "model_3"),
        "ckpt_prefix": "model_3",
        "tokenizer_cls": BPEMathTokenizer,
        "is_bpe": True,
        "eval_fn": evaluate_extrapolation_bpe,
    },
    4: {
        "name": "Model 4: Place-Value Zero-Reduction Thinking",
        "ckpt_dir": os.path.join(ROOT_DIR, "checkpoints", "model_4"),
        "ckpt_prefix": "model_4",
        "tokenizer_cls": MathTokenizer,
        "is_bpe": False,
        "eval_fn": evaluate_extrapolation_char,
    },
}


def encode_texts_to_batch(
    texts: List[str],
    tokenizer: Any,
    block_size: int,
    device: Optional[torch.device] = None,
    mask_prompt: bool = True,
    return_place_ids: bool = False,
) -> Any:
    B = len(texts)
    pad_id = tokenizer.pad_id
    eq_tokens = tokenizer.encode("=", add_bos=False, add_eos=False)
    eq_token_id = eq_tokens[0] if eq_tokens else None

    x_tensor = torch.full((B, block_size), pad_id, dtype=torch.long)
    y_tensor = torch.full((B, block_size), -100, dtype=torch.long)
    p_tensor = torch.zeros((B, block_size), dtype=torch.long) if return_place_ids else None

    for i, txt in enumerate(texts):
        tokens = tokenizer.encode(txt, add_bos=True, add_eos=True)
        if len(tokens) > block_size + 1:
            tokens = tokens[: block_size + 1]

        n = len(tokens) - 1
        t_tok = torch.tensor(tokens, dtype=torch.long)
        x_tensor[i, :n] = t_tok[:-1]
        y_tensor[i, :n] = t_tok[1:]

        if mask_prompt and eq_token_id is not None:
            eq_matches = (t_tok[:-1] == eq_token_id).nonzero(as_tuple=True)[0]
            if len(eq_matches) > 0:
                eq_idx = eq_matches[0].item()
                y_tensor[i, :eq_idx + 1] = -100

        if return_place_ids and p_tensor is not None:
            p = compute_abacus_place_ids(tokens[:-1], tokenizer)
            p_len = min(len(p), block_size)
            p_tensor[i, :p_len] = torch.tensor(p[:p_len], dtype=torch.long)

    if device is not None:
        x_tensor = x_tensor.to(device, non_blocking=True)
        y_tensor = y_tensor.to(device, non_blocking=True)
        if return_place_ids and p_tensor is not None:
            p_tensor = p_tensor.to(device, non_blocking=True)

    if return_place_ids:
        return x_tensor, y_tensor, p_tensor
    return x_tensor, y_tensor


class AsyncBatchPrefetcher:
    """
    Asynchronous background prefetcher that constructs batches in pinned host memory
    concurrently with GPU forward/backward execution, fully saturating GPU utilization
    and eliminating CPU-bound training pipeline starvation.
    """

    def __init__(self, fetch_fn: Any, total_sub_steps: int, queue_size: int = 4):
        self.fetch_fn = fetch_fn
        self.total_sub_steps = total_sub_steps
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        for sub_step in range(self.total_sub_steps):
            if self.stop_event.is_set():
                break
            batch = self.fetch_fn(sub_step)
            # Pin host memory tensors for high-speed non-blocking CUDA DMA transfers
            pinned = []
            for t in batch:
                if isinstance(t, torch.Tensor):
                    pinned.append(t.pin_memory() if torch.cuda.is_available() else t)
                else:
                    pinned.append(t)
            self.queue.put(pinned)
        self.queue.put(None)

    def next_batch(self) -> Optional[Any]:
        return self.queue.get()

    def stop(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break


def build_batch(
    condition: int,
    a_batch: torch.Tensor,
    b_batch: torch.Tensor,
    tokenizer: Any,
    block_size: int = 512,
    device: Optional[torch.device] = None,
    philo_tokens: Optional[List[int]] = None,
    philo_rng: Optional[random.Random] = None,
    return_place_ids: bool = False,
) -> Any:
    if condition == 1:
        texts = [format_addition_char_scratchpad(a, b) for a, b in zip(a_batch.tolist(), b_batch.tolist())]
        return encode_texts_to_batch(texts, tokenizer, block_size, device, return_place_ids=return_place_ids)
    elif condition == 2:
        texts = [format_addition_bpe_word_cot(a, b) for a, b in zip(a_batch.tolist(), b_batch.tolist())]
        return encode_texts_to_batch(texts, tokenizer, block_size, device, return_place_ids=return_place_ids)
    elif condition == 3:
        total = len(a_batch)
        half = total // 2
        a_sub = a_batch[:half].tolist()
        b_sub = b_batch[:half].tolist()
        math_texts = [format_addition_bpe_word_cot(a, b) for a, b in zip(a_sub, b_sub)]
        x_math, y_math = encode_texts_to_batch(math_texts, tokenizer, block_size, device=None, return_place_ids=False)

        n_philo = total - half
        pad_id = tokenizer.pad_id
        max_idx = len(philo_tokens) - (block_size + 2) if philo_tokens else 0
        x_p_list, y_p_list = [], []
        for _ in range(n_philo):
            idx = philo_rng.randint(0, max(0, max_idx)) if philo_rng else 0
            chunk = philo_tokens[idx : idx + block_size + 1] if philo_tokens else [pad_id] * (block_size + 1)
            if len(chunk) < block_size + 1:
                chunk = chunk + [pad_id] * (block_size + 1 - len(chunk))
            x_p_list.append(torch.tensor(chunk[:block_size], dtype=torch.long))
            y_p_list.append(torch.tensor(chunk[1 : block_size + 1], dtype=torch.long))
        x_philo = torch.stack(x_p_list)
        y_philo = torch.stack(y_p_list)

        x_tensor = torch.cat([x_math, x_philo], dim=0)
        y_tensor = torch.cat([y_math, y_philo], dim=0)
        if device is not None:
            x_tensor = x_tensor.to(device, non_blocking=True)
            y_tensor = y_tensor.to(device, non_blocking=True)
        return x_tensor, y_tensor
    elif condition == 4:
        texts = [format_addition_place_value_thinking(a, b) for a, b in zip(a_batch.tolist(), b_batch.tolist())]
        return encode_texts_to_batch(texts, tokenizer, block_size, device, return_place_ids=return_place_ids)
    raise ValueError(f"Unknown condition: {condition}")


def get_training_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target_tokens", "--target-tokens", dest="target_tokens", type=int, default=DEFAULT_TARGET_TOKENS, help="Target token budget (default: 400M tokens)")
    parser.add_argument("--hours", type=float, default=None, help="Optional maximum hours timeout")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer layers")
    parser.add_argument("--n_head", type=int, default=8, help="Attention heads")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable FP16 AMP")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable AMP")
    parser.add_argument("--log_interval", "--log-interval", dest="log_interval", type=int, default=50, help="Log interval in steps")
    parser.add_argument("--checkpoint_interval_tokens", "--checkpoint-interval-tokens", dest="checkpoint_interval_tokens", type=int, default=50_000_000, help="Intermediate checkpoint token interval")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--model_type", type=str, default="standard", choices=["standard", "looped", "hierarchical", "hybrid", "alsm", "hnrm", "micromath", "hrm_rope"], help="Model architecture type")
    parser.add_argument("--n_core_layers", type=int, default=2, help="Unique core layers for Looped/ALSM Transformer")
    parser.add_argument("--train_loops", type=int, default=6, help="Unrolled loops during training for Looped/ALSM Transformer")
    parser.add_argument("--eval_loops", type=int, default=12, help="Dynamic loops during evaluation for Looped/ALSM Transformer")
    parser.add_argument("--n_fast_steps", type=int, default=4, help="Inner fast loop steps for Hierarchical Reasoning Model")
    parser.add_argument("--n_slow_cycles", type=int, default=3, help="Outer slow loop cycles for Hierarchical Reasoning Model")
    parser.add_argument("--pattern", type=str, default="interleaved", choices=["interleaved", "pure_ssm", "pure_attn"], help="Layer pattern for Hybrid model")
    parser.add_argument("--max_places", type=int, default=128, help="Max place capacity for Abacus embeddings")
    parser.add_argument("--k_rand_shift", type=int, default=32, help="Train-time offset jitter for Abacus embeddings")
    parser.add_argument("--base_freq", type=float, default=1.0, help="Base frequency for HPF oscillators in H-NRM")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps per update")
    return parser


def train_condition(condition: int, args: Optional[argparse.Namespace] = None):
    meta = CONDITION_METADATA[condition]
    if args is None:
        args = get_training_arg_parser(meta["name"]).parse_args([])

    device, rank, local_rank, world_size, is_master = init_distributed_mode(args)
    model_type = getattr(args, "model_type", "standard")
    if model_type != "standard":
        checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints", model_type)
        ckpt_prefix = f"{model_type}_c{condition}"
    else:
        checkpoint_dir = meta["ckpt_dir"]
        ckpt_prefix = meta["ckpt_prefix"]

    if is_master:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("=" * 75)
        print(f"TRAINING {meta["name"].upper()} [Model Type: {model_type}]")
        print(f"World Size: {world_size} | Device: {device} | Precision: FP16 AMP")
        print("=" * 75)

    tokenizer = meta["tokenizer_cls"]()
    model_type = getattr(args, "model_type", "standard")

    if model_type == "looped":
        config = LoopedGPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_core_layers=getattr(args, "n_core_layers", 2),
            train_loops=getattr(args, "train_loops", 6),
            eval_loops=getattr(args, "eval_loops", 12),
            dropout=args.dropout,
            bias=False,
            rope_theta=10000.0,
            deep_supervision=True,
            model_type="looped",
        )
    elif model_type == "hierarchical":
        config = HierarchicalGPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_fast_steps=getattr(args, "n_fast_steps", 4),
            n_slow_cycles=getattr(args, "n_slow_cycles", 3),
            dropout=args.dropout,
            bias=False,
            rope_theta=10000.0,
            model_type="hierarchical",
        )
    elif model_type == "hybrid":
        config = HybridConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
            d_ffn=args.n_embd * 4,
            dropout=args.dropout,
            bias=False,
            rope_theta=10000.0,
            pattern=getattr(args, "pattern", "interleaved"),
            model_type="hybrid",
        )
    elif model_type in ["alsm", "abacus_looped"]:
        config = ALSMConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_core_layers=getattr(args, "n_core_layers", 4),
            train_loops=getattr(args, "train_loops", 6),
            eval_loops=getattr(args, "eval_loops", 12),
            max_places=getattr(args, "max_places", 128),
            k_rand_shift=getattr(args, "k_rand_shift", 32),
            dropout=args.dropout,
            bias=False,
            rope_theta=10000.0,
            deep_supervision=True,
            model_type="alsm",
        )
    elif model_type == "hnrm":
        config = HNRMConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
            d_ffn=args.n_embd * 4,
            dropout=args.dropout,
            bias=False,
            base_freq=getattr(args, "base_freq", 1.0),
            model_type="hnrm",
        )
    elif model_type in ["micromath", "micromath2m"]:
        config = MicroMathConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            n_embd=getattr(args, "n_embd", 192),
            n_head=getattr(args, "n_head", 4),
            n_layer=getattr(args, "n_layer", 3),
            d_ffn=getattr(args, "d_ffn", getattr(args, "n_embd", 192) * 4),
            dropout=args.dropout,
            bias=False,
            max_places=getattr(args, "max_places", 128),
            k_rand_shift=getattr(args, "k_rand_shift", 32),
            model_type="micromath",
        )
    elif model_type == "hrm_rope":
        config = HRMRoPEConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            n_embd=getattr(args, "n_embd", 256),
            n_head=getattr(args, "n_head", 4),
            d_ffn=getattr(args, "d_ffn", 768),
            n_fast_steps=getattr(args, "n_fast_steps", 2),
            n_slow_cycles=getattr(args, "n_slow_cycles", 2),
            dropout=args.dropout,
            bias=False,
            rope_theta=getattr(args, "rope_theta", 10000.0),
            max_coords=getattr(args, "max_coords", 2048),
            k_rand_shift=getattr(args, "k_rand_shift", 32),
            model_type="hrm_rope",
        )
    elif model_type in ["hrm_text", "hrmtext"]:
        config = HRMTextConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            hidden_size=getattr(args, "hidden_size", getattr(args, "n_embd", 256)),
            num_heads=getattr(args, "num_heads", getattr(args, "n_head", 4)),
            expansion=getattr(args, "expansion", 3.0),
            h_cycles=getattr(args, "h_cycles", getattr(args, "n_slow_cycles", 2)),
            l_cycles=getattr(args, "l_cycles", getattr(args, "n_fast_steps", 3)),
            n_layers=getattr(args, "n_layers", getattr(args, "n_layer", 2)),
            half_layers=True,
            dropout=args.dropout,
            bias=False,
            rope_theta=getattr(args, "rope_theta", 10000.0),
            norm_eps=getattr(args, "norm_eps", 1e-6),
            rope_mode=getattr(args, "rope_mode", "1d"),
            model_type="hrm_text",
        )
    else:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=512,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            pos_emb_type="rope",
        )

    raw_model = create_model(config).to(device)
    n_params = raw_model.get_num_params(non_embedding=True)
    target_tokens = args.target_tokens or DEFAULT_TARGET_TOKENS
    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum_steps

    # Auto-tune micro-batch size on CUDA Dual T4:
    # 1. Non-recurrent models (standard, hnrm) on condition 1/4 can scale to batch_size=128
    #    which fully saturates Turing T4 Tensor Cores and utilizes ~11 GiB of VRAM (up from 7 GiB).
    # 2. Large vocabulary BPE models (condition 2/3) use micro-batch size 32 with 2 accum steps.
    # 3. Recurrent / unrolled models (hybrid, alsm, looped, hierarchical) have large activation footprint;
    #    use micro-batch size 32 with 2 accum steps to maintain effective batch size 64 while preventing CUDA OOM.
    if torch.cuda.is_available() and batch_size == 64 and grad_accum_steps == 1 and model_type in ["standard", "hnrm"] and condition in [1, 4]:
        batch_size = 128
        if is_master:
            print("[INFO] High-throughput Tensor Core auto-tune: Scaled batch_size to 128 for Turing T4 (~11 GiB VRAM).")
    elif (condition in [2, 3] or model_type in ["hybrid", "alsm", "looped", "hierarchical"]) and batch_size == 64 and grad_accum_steps == 1:
        batch_size = 32
        grad_accum_steps = 2
        if is_master:
            if model_type in ["hybrid", "alsm", "looped", "hierarchical"]:
                print(f"[INFO] Recurrent/Unrolled model ({model_type}) detected: Auto-tuning to micro-batch_size=32 with grad_accum_steps=2 to prevent Turing T4 CUDA OOM.")
            else:
                print("[INFO] Large vocabulary BPE model detected: Auto-tuning to micro-batch_size=32 with grad_accum_steps=2 to prevent Turing T4 CUDA OOM.")

    tokens_per_step = batch_size * grad_accum_steps * config.block_size * world_size
    total_target_steps = math.ceil(target_tokens / tokens_per_step)

    if is_master:
        print(f"Non-Embedding Parameters: {n_params:,}")
        print(f"Target Token Budget:      {target_tokens / 1e6:.1f}M tokens")
        print(f"Micro-Batch Size:         {batch_size} (Grad Accum: {grad_accum_steps})")
        print(f"Total Target Steps:       {total_target_steps:,}")

    is_alsm = (model_type in ["alsm", "abacus_looped", "micromath", "micromath2m"])
    model = wrap_model_ddp(raw_model, device, local_rank)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_target_steps), eta_min=args.lr * 0.1)
    AutocastCtx, scaler = get_amp_components(device, enabled=args.use_amp)

    if is_master:
        get_or_generate_shared_math_pairs()
    distributed_barrier()

    a_all, b_all = get_or_generate_shared_math_pairs()
    total_equations = len(a_all)

    philo_tokens = None
    philo_rng = None
    if condition == 3:
        philo_tokens = load_philosophy_token_stream(tokenizer)
        if is_master:
            print(f"Philosophy Corpus Loaded: {len(philo_tokens):,} tokens")
        philo_rng = random.Random(42 + rank)

    step_divisor = 2 if condition == 3 else 1
    indices = list(range(rank, total_equations, world_size))
    total_local = len(indices)
    total_sub_steps = total_target_steps * grad_accum_steps

    def fetch_sub_step(sub_step_idx: int) -> Any:
        step_batch = batch_size // step_divisor
        idx_start = (sub_step_idx * step_batch) % max(1, total_local - step_batch)
        batch_indices = indices[idx_start : idx_start + step_batch]
        a_batch = a_all[batch_indices]
        b_batch = b_all[batch_indices]
        return build_batch(
            condition=condition,
            a_batch=a_batch,
            b_batch=b_batch,
            tokenizer=tokenizer,
            block_size=config.block_size,
            device=None,
            philo_tokens=philo_tokens,
            philo_rng=philo_rng,
            return_place_ids=is_alsm,
        )

    prefetcher = AsyncBatchPrefetcher(fetch_sub_step, total_sub_steps=total_sub_steps, queue_size=4)

    model.train()
    start_time = time.time()
    step = 0
    total_tokens_seen = 0
    running_loss = 0.0
    last_checkpoint_tokens = 0

    while total_tokens_seen < target_tokens:
        if args.hours and args.hours > 0:
            if (time.time() - start_time) >= args.hours * 3600.0 and step > 0:
                if is_master:
                    print(f"Reached time limit of {args.hours:.2f} hours.")
                break

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for accum_idx in range(grad_accum_steps):
            batch = prefetcher.next_batch()
            if batch is None:
                break

            if is_alsm:
                x_batch, y_batch, p_batch = batch
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                p_batch = p_batch.to(device, non_blocking=True)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

            # In DDP, avoid gradient communication until the last accumulation step
            sync_context = model.no_sync() if (accum_idx < grad_accum_steps - 1 and hasattr(model, "no_sync")) else contextlib.nullcontext()
            with sync_context:
                with AutocastCtx():
                    if is_alsm:
                        _, loss = model(x_batch, place_ids=p_batch, targets=y_batch)
                    else:
                        _, loss = model(x_batch, targets=y_batch)
                    if loss is None:
                        raise RuntimeError(
                            f"Model forward pass returned None for loss! Model type: {model_type}"
                        )
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            step_loss += loss.item() * grad_accum_steps

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        if scale_after >= scale_before:
            scheduler.step()

        tokens_this_step = batch_size * grad_accum_steps * config.block_size * world_size
        total_tokens_seen += tokens_this_step
        running_loss += step_loss
        step += 1

        if is_master and step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - start_time
            tps = total_tokens_seen / max(elapsed, 1e-4)
            current_lr = scheduler.get_last_lr()[0]
            pct = (total_tokens_seen / target_tokens) * 100.0
            print(
                f"[Step {step:6d}/{total_target_steps:6d} | {pct:5.1f}%] "
                f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                f"Tokens: {total_tokens_seen / 1e6:6.1f}M/{target_tokens / 1e6:.0f}M | "
                f"Speed: {tps:7.0f} tok/s | Elapsed: {elapsed / 60:.1f}m"
            )
            running_loss = 0.0

        if is_master and (total_tokens_seen - last_checkpoint_tokens) >= args.checkpoint_interval_tokens:
            last_checkpoint_tokens = total_tokens_seen
            ckpt_path = os.path.join(checkpoint_dir, f"{ckpt_prefix}_{total_tokens_seen // 1_000_000}m.pt")
            torch.save(
                {
                    "step": step,
                    "tokens_seen": total_tokens_seen,
                    "model_state": raw_model.state_dict(),
                    "model_state_dict": raw_model.state_dict(),
                    "config": config,
                    "model_type": getattr(config, "model_type", "standard"),
                    "condition": condition,
                },
                ckpt_path,
            )
            print(f"[CHECKPOINT] Saved: {ckpt_path}")
            sync_checkpoint_to_kaggle(ckpt_path, model_type)

    prefetcher.stop()
    distributed_barrier()
    total_time = time.time() - start_time

    if is_master:
        final_ckpt = os.path.join(checkpoint_dir, f"{ckpt_prefix}_final.pt")
        ckpt_data = {
            "step": step,
            "tokens_seen": total_tokens_seen,
            "model_state": raw_model.state_dict(),
            "model_state_dict": raw_model.state_dict(),
            "config": config,
            "model_type": getattr(config, "model_type", "standard"),
            "condition": condition,
            "total_time_sec": total_time,
            "target_tokens": target_tokens,
        }
        torch.save(ckpt_data, final_ckpt)
        print(f"[CHECKPOINT] Saved Final Checkpoint: {final_ckpt}")
        sync_checkpoint_to_kaggle(final_ckpt, model_type)

        print(f"\nRunning Extrapolation Benchmark on {meta['name']} Final Checkpoint...")
        meta["eval_fn"](
            raw_model,
            tokenizer,
            device,
            title=f"{meta["name"]} Final ({total_tokens_seen / 1e6:.0f}M tokens)",
            save_results=True,
        )

    cleanup_distributed()
