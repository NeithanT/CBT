import os
import sys
import glob
import json
import random
import time
import argparse
from typing import Callable, Dict, List, Any, Optional, Tuple
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.model import MathGPT, GPTConfig, create_model, apply_rope
from src.tokenizer import MathTokenizer
from src.bpe_tokenizer import BPEMathTokenizer
from src.constants import get_device
from benchmarks.extrapolation import run_extrapolation_benchmark
from src.arithmetic import (
    extract_scratchpad_answer,
    format_addition_char_scratchpad,
    format_addition_bpe_word_cot,
    format_addition_place_value_thinking,
)
from src.dataset import compute_abacus_place_ids
from experiments.run_equalized_15d_1024_and_large_sweep import wilson_ci, generate_random_problems_for_length

SWEEP_LENGTHS = [4, 8, 12, 14, 16, 18, 20, 25, 30, 35]
SWEEP_TRAINED_MAX_DIGITS = 12
SWEEP_TRACE_SLACK = 16
SKIP_DIRS = ("snapshots", "_smoke_tests")


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = create_model(config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state_dict is None:
        raise KeyError(f"No state dict found in checkpoint {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()

    is_char = (config.vocab_size <= 64)
    tokenizer = MathTokenizer() if is_char else BPEMathTokenizer()
    model_type = getattr(config, "model_type", "standard").capitalize()
    if is_char:
        model_desc = f"{model_type} (Char)"
    else:
        model_desc = f"{model_type} (BPE Subword)"
    return model, tokenizer, config, model_desc



def benchmark_single_checkpoint(
    checkpoint_path: str, device: torch.device, label: Optional[str] = None
) -> Dict[str, Any]:
    model, tokenizer, config, model_type = load_model_from_checkpoint(checkpoint_path, device)
    model_label = label or os.path.basename(checkpoint_path).replace(".pt", "")
    n_params = model.get_num_params(non_embedding=True)

    print(f"\nEvaluating: {checkpoint_path}")
    print(f"Type: {model_type} | Params: {n_params:,} | Block size: {config.block_size}")

    return run_extrapolation_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_label,
        is_scratchpad=True,
        is_bpe=(not is_char),
        save_results=True,
    )


def find_all_checkpoints(base_dir: str = ROOT_DIR) -> List[str]:
    patterns = [
        os.path.join(base_dir, "checkpoints", "**", "*.pt"),
        os.path.join(base_dir, "checkpoints_*", "*.pt"),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    found = [p for p in found if not any(f"{os.sep}{d}{os.sep}" in p for d in SKIP_DIRS)]
    return sorted(list(set(found)))


def _trace_formatter(condition: int, is_char: bool) -> Callable[[int, int], str]:
    """Gold trace format the checkpoint was trained on (trainer.py conditions 1-4)."""
    if not is_char:
        return format_addition_bpe_word_cot
    if condition == 4:
        return format_addition_place_value_thinking
    return format_addition_char_scratchpad


def _forward_kwargs(model: torch.nn.Module, idx: torch.Tensor, tokenizer: Any) -> Dict[str, Any]:
    """Coordinates some architectures expect from the caller rather than computing in forward."""
    model_type = getattr(model.config, "model_type", "standard")
    if model_type in ("hrm_rope", "micromath"):
        return {"tokenizer": tokenizer}
    if model_type in ("alsm", "abacus_looped"):
        rows = [compute_abacus_place_ids(r, tokenizer) for r in idx.tolist()]
        return {"place_ids": torch.tensor(rows, dtype=torch.long, device=idx.device)}
    return {}


def _supports_kv_cache(model: torch.nn.Module) -> bool:
    return isinstance(model, MathGPT) and model.config.pos_emb_type == "rope" and getattr(model.config, "model_type", "standard") == "standard"


class _KVStepper:
    """Incremental forward for plain RoPE MathGPT: prefill once, then one token per call."""

    def __init__(self, model: MathGPT):
        self.model = model
        self.cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(model.transformer.h)

    def __call__(self, tokens: torch.Tensor, start: int) -> torch.Tensor:
        m = self.model
        x = m.transformer.wte(tokens)
        bsz, t = tokens.shape
        for i, block in enumerate(m.transformer.h):
            attn = block.attn
            q, k, v = attn.c_attn(block.ln_1(x)).chunk(3, dim=2)
            q, k, v = (z.view(bsz, t, attn.n_head, attn.head_dim).transpose(1, 2) for z in (q, k, v))
            q = apply_rope(q, attn.rope_cos[start:start + t], attn.rope_sin[start:start + t])
            k = apply_rope(k, attn.rope_cos[start:start + t], attn.rope_sin[start:start + t])
            if self.cache[i] is not None:
                k = torch.cat([self.cache[i][0], k], dim=2)
                v = torch.cat([self.cache[i][1], v], dim=2)
            self.cache[i] = (k, v)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=t > 1)
            x = x + attn.c_proj(y.transpose(1, 2).contiguous().view(bsz, t, attn.n_embd))
            x = x + block.mlp(block.ln_2(x))
        return m.lm_head(m.transformer.ln_f(x[:, -1:, :]))[:, -1, :]


@torch.no_grad()
def batched_greedy(
    model: torch.nn.Module, tokenizer: Any, prompts: List[str], max_new_tokens: int, device: torch.device,
    use_cache: bool = True,
) -> Tuple[List[str], List[bool]]:
    """Greedy decoding of equal-length prompts as one batch; returns texts and context-truncation flags."""
    idx = torch.tensor([tokenizer.encode(p, add_bos=True) for p in prompts], dtype=torch.long, device=device)
    block_size = model.config.block_size
    prompt_len = idx.size(1)
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    stepper = _KVStepper(model) if use_cache and _supports_kv_cache(model) else None
    for _ in range(max_new_tokens):
        if idx.size(1) >= block_size:
            break
        if stepper is not None:
            n = idx.size(1)
            last = stepper(idx if n == prompt_len else idx[:, -1:], 0 if n == prompt_len else n - 1)
        else:
            last = model(idx, **_forward_kwargs(model, idx, tokenizer))[0][:, -1, :]
        nxt = torch.argmax(last, dim=-1)
        nxt = torch.where(done, torch.full_like(nxt, tokenizer.eos_id), nxt)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        done |= nxt == tokenizer.eos_id
        if bool(done.all()):
            break
    texts, truncated = [], []
    for row, fin in zip(idx.tolist(), done.tolist()):
        if tokenizer.eos_id in row:
            row = row[: row.index(tokenizer.eos_id) + 1]
        texts.append(tokenizer.decode(row, skip_special=False))
        truncated.append((not fin) and len(row) >= block_size)
    return texts, truncated


def random_sweep_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    n_per_length: int = 30,
    lengths: Optional[List[int]] = None,
    seed: int = 20261023,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    done: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stratified random multi-carry sweep (seen 1-12d vs unseen 14-35d), exact final-answer match."""
    model, tokenizer, config, model_desc = load_model_from_checkpoint(checkpoint_path, device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    condition = int(ckpt.get("condition", 1) or 1)
    is_char = config.vocab_size <= 64
    fmt = _trace_formatter(condition, is_char)
    rel = os.path.relpath(checkpoint_path, ROOT_DIR)
    print(f"\n[INFO] Sweep: {rel} | {model_desc} | condition {condition} | block {config.block_size}", flush=True)

    per_length: Dict[str, Any] = dict(done or {})
    record: Dict[str, Any] = {
        "checkpoint": rel,
        "model": model_desc,
        "model_type": getattr(config, "model_type", "standard"),
        "condition": condition,
        "non_embedding_params": model.get_num_params(non_embedding=True),
        "block_size": config.block_size,
        "seed": seed,
        "per_length": per_length,
    }
    for d in lengths or SWEEP_LENGTHS:
        if str(d) in per_length:
            continue
        rng = random.Random(seed + d)
        problems = generate_random_problems_for_length(d, n_per_length, rng)
        prompts, golds, budgets = [], [], []
        for prompt, expected in problems:
            a, b = (int(x) for x in prompt.rstrip("=").split("+"))
            gold = fmt(a, b)
            prompts.append(prompt + "<THINK>")
            golds.append(expected)
            budgets.append(len(tokenizer.encode(gold)) - len(tokenizer.encode(prompt + "<THINK>")) + 1)
        # Equal prompt lengths batch without padding; BPE logits (V=50k) cap the batch for memory
        max_batch = 32 if is_char else 4
        by_len: Dict[int, List[int]] = {}
        for i, p in enumerate(prompts):
            by_len.setdefault(len(tokenizer.encode(p, add_bos=True)), []).append(i)
        groups = {
            (L, j): m[j:j + max_batch] for L, m in by_len.items() for j in range(0, len(m), max_batch)
        }
        texts: List[str] = [""] * len(prompts)
        trunc: List[bool] = [False] * len(prompts)
        t0 = time.time()
        for members in groups.values():
            budget = max(budgets[i] for i in members) + SWEEP_TRACE_SLACK
            out, tr = batched_greedy(model, tokenizer, [prompts[i] for i in members], budget, device)
            for i, o, t in zip(members, out, tr):
                texts[i], trunc[i] = o, t
        answers = [extract_scratchpad_answer(t)[1] for t in texts]
        correct = [a == g for a, g in zip(answers, golds)]
        k, n = sum(correct), len(correct)
        lo, hi = wilson_ci(k, n)
        failures = [
            {"prompt": prompts[i], "expected": golds[i], "answer": answers[i][:80], "tail": texts[i][-160:]}
            for i in range(n) if not correct[i]
        ][:2]
        per_length[str(d)] = {
            "seen": d <= SWEEP_TRAINED_MAX_DIGITS,
            "correct": k,
            "n": n,
            "acc_pct": round(100.0 * k / n, 1),
            "wilson_ci_pct": [round(lo, 1), round(hi, 1)],
            "context_truncated": sum(trunc),
            "elapsed_sec": round(time.time() - t0, 1),
            "example_failures": failures,
        }
        print(f"  {d:3d}d {'seen  ' if d <= SWEEP_TRAINED_MAX_DIGITS else 'unseen'} {k:3d}/{n} "
              f"({100.0 * k / n:5.1f}%) ctx_trunc={sum(trunc)} [{time.time() - t0:.0f}s]", flush=True)
        if on_progress is not None:
            on_progress(record)
    return record


def main():
    parser = argparse.ArgumentParser(description="Extrapolation Benchmark Runner")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--all", action="store_true", help="Benchmark all discovered checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda:0)")
    parser.add_argument("--sweep", nargs="+", default=None, help="Random multi-carry sweep over these checkpoints")
    parser.add_argument("--n-per-length", type=int, default=30, help="Sweep problems per digit length")
    parser.add_argument("--out-json", type=str, default=None, help="Sweep output JSON")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads")
    parser.add_argument("--resume", action="store_true", help="Skip lengths already in --out-json (per-length seeds are fixed)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else get_device()
    print(f"Benchmark Device: {device}")

    if args.sweep:
        results: List[Dict[str, Any]] = []
        prior: Dict[str, Dict[str, Any]] = {}
        if args.resume and args.out_json and os.path.exists(args.out_json):
            with open(args.out_json, encoding="utf-8") as f:
                prior_records = json.load(f)["results"]
            prior = {r["checkpoint"]: r["per_length"] for r in prior_records}
            requested = {os.path.relpath(os.path.abspath(ck), ROOT_DIR) for ck in args.sweep}
            results.extend(r for r in prior_records if r["checkpoint"] not in requested)

        def dump(partial: Dict[str, Any]) -> None:
            if not args.out_json:
                return
            os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump({"lengths": SWEEP_LENGTHS, "results": results + [partial]}, f, indent=2)

        for ck in args.sweep:
            done = prior.get(os.path.relpath(os.path.abspath(ck), ROOT_DIR))
            results.append(random_sweep_checkpoint(ck, device, n_per_length=args.n_per_length, on_progress=dump, done=done))
        print(f"[OK] Sweep complete: {args.out_json}")
    elif args.checkpoint:
        benchmark_single_checkpoint(args.checkpoint, device)
    elif args.all:
        all_pts = find_all_checkpoints(ROOT_DIR)
        if not all_pts:
            print("No checkpoints found.")
            return
        print(f"Discovered {len(all_pts)} checkpoints:")
        for pt in all_pts:
            print(f"  {os.path.relpath(pt, ROOT_DIR)}")
        for pt in all_pts:
            try:
                benchmark_single_checkpoint(pt, device)
            except Exception as e:
                print(f"[ERROR] Failed {pt}: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
