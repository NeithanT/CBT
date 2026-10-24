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
import re
import sys
import gzip
import time
import math
import random
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        generate_search_v5_example,
        generate_v5_tool_example,
        to_tool_item,
        encode_joint_batch,
        extract_scratchpad_answer,
    )
    from champion_model.scaled_t4.train_and_eval import sync_checkpoint_to_kaggle, prune_checkpoints
except ImportError:
    from model import ScaledGPT, ScaledGPTConfig
    from tokenizer import ScaledTokenizer
    from dataset import (
        generate_arithmetic_equations,
        generate_tool_dataset,
        generate_search_delegation_dataset,
        generate_search_v5_example,
        generate_v5_tool_example,
        to_tool_item,
        encode_joint_batch,
        extract_scratchpad_answer,
    )
    from train_and_eval import sync_checkpoint_to_kaggle, prune_checkpoints


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def build_benchmark_holdout() -> Tuple[Set[str], List[str]]:
    """
    Returns (normalized benchmark prompts, free-text expected values). Values come from the
    benchmark's own `case_free_values`, so the filter matches `audit_contamination` exactly.
    """
    from benchmarks.evaluate_champion import PHRASING_26_CASES, OOD_18_CASES, case_free_values
    from benchmarks.evaluate_unseen_ood import UNSEEN_OOD_30_CASES

    prompts: Set[str] = set()
    values: Set[str] = set()
    for case in PHRASING_26_CASES + OOD_18_CASES + UNSEEN_OOD_30_CASES:
        prompts.add(_norm(case["prompt"].split("<USER>")[1].split("</USER>")[0]))
    for case in PHRASING_26_CASES + OOD_18_CASES + [{"expected_arg": c["expected_arg"]} for c in UNSEEN_OOD_30_CASES]:
        values.update(v for v in case_free_values(case) if v)
    return prompts, sorted(values)


def is_contaminated(item: Dict[str, str], prompts: Set[str], values: List[str]) -> bool:
    """
    Rejects a trace if its opening user turn equals a benchmark prompt (benchmarks score first turns only)
    or any free value is a raw substring anywhere in the trace (the audit_contamination rule).
    """
    text = item["full_text"]
    first_user = _norm(text.split("<USER>")[1].split("</USER>")[0])
    return first_user in prompts or any(v in text for v in values)


class CurriculumV5Stream:
    """Infinite decontaminated stream of V5 search and tool traces (fresh spans every batch)."""

    def __init__(self, seed: int, prompts: Set[str], spans: List[str], corpus_path: Optional[str] = None):
        self.rng = random.Random(seed)
        self.prompts, self.spans = prompts, spans
        self.rejected = 0
        self.corpus = gzip.open(corpus_path, "wt", encoding="utf-8") if corpus_path else None

    def _draw(self, gen) -> Dict[str, str]:
        while True:
            item = to_tool_item(gen(self.rng))
            if not is_contaminated(item, self.prompts, self.spans):
                if self.corpus is not None:
                    self.corpus.write(item["full_text"] + "\n")
                return item
            self.rejected += 1

    def close(self) -> None:
        if self.corpus is not None:
            self.corpus.close()

    def search(self, n: int) -> List[Dict[str, str]]:
        return [self._draw(generate_search_v5_example) for _ in range(n)]

    def tools(self, n: int) -> List[Dict[str, str]]:
        return [self._draw(generate_v5_tool_example) for _ in range(n)]


def call_weighted_loss(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, call_s_id: int, call_e_id: int, call_weight: float) -> torch.Tensor:
    """Cross-entropy with `call_weight` applied to targets inside <CALL>...</CALL> (argument copy positions)."""
    opens = (x == call_s_id).long().cumsum(dim=1)
    closes = (x == call_e_id).long().cumsum(dim=1)
    weights = torch.where(opens > closes, call_weight, 1.0).to(logits.dtype)
    tok_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none")
    valid = (y.reshape(-1) != -100).float()
    w = weights.reshape(-1).float() * valid
    return (tok_loss * w).sum() / w.sum().clamp_min(1.0)


def cosine_lr(progress: float, step: int, peak: float, floor: float, warmup: int) -> float:
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = min(1.0, max(0.0, progress))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def run_heldout_eval(model: nn.Module, tokenizer: ScaledTokenizer, device: torch.device, items: List[Dict[str, str]]) -> float:
    """Exact-call accuracy on generated held-out traces (first <CALL> or direct <ANSWER> routing)."""
    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()
    stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
    passed = 0
    for item in items:
        text = item["full_text"]
        toks = tokenizer.encode(item["prompt"] + "<THINK>", add_bos=True, add_eos=False)
        out = raw_model.generate(torch.tensor([toks], device=device), max_new_tokens=140, temperature=0.0, stop_tokens=stop_ids)
        resp = tokenizer.decode(out[0][len(toks):].tolist(), skip_special=False)
        first = text.split("</THINK>", 1)[1]
        if first.startswith("<CALL>"):
            target = first.split("</CALL>")[0] + "</CALL>"
            ok = target in resp
        else:
            ok = "<ANSWER>" in resp and "<CALL>" not in resp
        passed += int(ok)
    if was_training:
        raw_model.train()
    return passed / max(1, len(items))


def make_math_heldout(seed: int) -> List[Tuple[int, int]]:
    """Seen (25d) and unseen (30-45d) random, dense-carry, and cascade problems; seed disjoint from the benchmark."""
    rng = random.Random(seed)
    rand = lambda d: rng.randint(10 ** (d - 1), 10 ** d - 1)
    dense = lambda d: int("".join(str(rng.randint(5, 9)) for _ in range(d)))
    return [(rand(25), rand(25)), (dense(30), dense(30)), (rand(40), rand(40)), (int("9" * 45), rng.randint(1, 9))]


@torch.no_grad()
def run_math_heldout(model: nn.Module, tokenizer: ScaledTokenizer, device: torch.device, problems: List[Tuple[int, int]]) -> float:
    """Exact final-answer accuracy on held-out long additions (guards extrapolation against SFT drift)."""
    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()
    passed = 0
    for a, b in problems:
        toks = tokenizer.encode(f"{a}+{b}=<THINK>", add_bos=True, add_eos=False)
        digits = max(len(str(a)), len(str(b)))
        out = raw_model.generate(torch.tensor([toks], device=device), max_new_tokens=(digits + 2) * 15 + 40, temperature=0.0, stop_tokens=[tokenizer.eos_id])
        _, ans = extract_scratchpad_answer(tokenizer.decode(out[0].tolist(), skip_special=False))
        passed += int(ans == str(a + b))
    if was_training:
        raw_model.train()
    return passed / max(1, len(problems))


def run_sft_probe(model: nn.Module, tokenizer: ScaledTokenizer, device: torch.device, step: int):
    """Runs a non-blocking fast probe across all tools, search copy induction, and arithmetic carry."""
    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()

    probe_prompts = [
        ("Weather (on Rome)", "<USER>What is the weather on Rome</USER><THINK>", 'get_weather(city="Rome")'),
        ("Weather (Buenos Aires)", "<USER>What is the weather in Buenos Aires?</USER><THINK>", 'get_weather(city="Buenos Aires")'),
        ("Weather (for London)", "<USER>Check the weather for London</USER><THINK>", 'get_weather(city="London")'),
        ("Time (in GMT)", "<USER>What time is it in GMT</USER><THINK>", 'get_time(timezone="GMT")'),
        ("Calculator (*)", "<USER>Compute 84 * 19</USER><THINK>", 'calculator(expr='),
        ("Unit Convert", "<USER>How many km is 42 miles</USER><THINK>", 'unit_convert(value=42, from_unit="miles", to_unit="km")'),
        ("Search (brutalist)", "<USER>Search for brutalist architecture</USER><THINK>", 'search(query="brutalist architecture")'),
        ("Search (crispr cas9)", "<USER>Explain crispr cas9</USER><THINK>", 'search(query="crispr cas9")'),
        ("Search (existentialism)", "<USER>What is existentialism?</USER><THINK>", 'search(query="existentialism")'),
        ("Search (Marcus Aurelius)", "<USER>Who was Marcus Aurelius?</USER><THINK>", 'search(query="Marcus Aurelius")'),
        ("Search (Multi-Word Nonce)", "<USER>Tell me about Zondar Krell</USER><THINK>", 'search(query="Zondar Krell")'),
        ("Nested JSON", "<USER>Query the top 5 active records in the engineering department.</USER><THINK>", "database_query("),
        ("Math 9-Digit", "123456789+987654321=<THINK>", "1111111110"),
        ("Math 25-Digit Cascade", "9999999999999999999999999+1=<THINK>", "10000000000000000000000000"),
    ]

    print(f"\n--- [PROBE Step {step}] Universal All-Tools Alignment & Carry Check ---")
    for label, p, expected in probe_prompts:
        toks = tokenizer.encode(p, add_bos=True, add_eos=False)
        x = torch.tensor([toks], dtype=torch.long, device=device)
        with torch.no_grad():
            stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
            budget = 140 if p.startswith("<USER>") else 600
            out = raw_model.generate(x, max_new_tokens=budget, temperature=0.0, stop_tokens=stop_ids)
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

        print(f"  {status} {label:<24} -> {disp}", flush=True)
    print("-" * 75 + "\n", flush=True)

    if was_training:
        raw_model.train()


def train_v5(args: argparse.Namespace, model: nn.Module, tokenizer: ScaledTokenizer, device: torch.device) -> None:
    """
    Curriculum V5 SFT: decontaminated online stream, CALL-weighted loss, wall-clock-aware cosine LR,
    and best-checkpoint selection on a generated held-out set (never on benchmark suites).
    """
    save_path = os.path.join(ROOT_DIR, args.save_ckpt)
    ckpt_dir = os.path.dirname(save_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    corpus_path = save_path.replace(".pt", "_corpus.txt.gz")

    prompts, spans = build_benchmark_holdout()
    stream = CurriculumV5Stream(seed=args.seed, prompts=prompts, spans=spans, corpus_path=corpus_path)
    heldout_stream = CurriculumV5Stream(seed=args.seed + 7919, prompts=prompts, spans=spans)
    heldout = heldout_stream.search(args.heldout_size // 2) + heldout_stream.tools(args.heldout_size - args.heldout_size // 2)
    math_heldout = make_math_heldout(seed=args.seed + 104729)
    math_ds = [m for m in generate_arithmetic_equations(n_samples=20000, seed=4242, max_train_digits=25)
               if not any(v in m["full_text"] for v in spans)]
    for m in math_ds:
        stream.corpus.write(m["full_text"] + "\n")
    rng_m = random.Random(args.seed)
    m_idx = 0

    n_m = max(1, round(args.batch_size * args.math_frac))
    n_s = max(1, round(args.batch_size * args.search_frac))
    n_t = args.batch_size - n_m - n_s
    budget_s = args.max_hours * 3600.0

    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_cuda_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    base_name = os.path.splitext(os.path.basename(save_path))[0]
    best_path = save_path.replace(".pt", "_best.pt")

    def save(path: str, step: int, extra: Dict[str, Any]) -> None:
        raw_model = model.module if hasattr(model, "module") else model
        torch.save({"model_state_dict": raw_model.state_dict(), "config": raw_model.config, "step": step, **extra}, path)
        sync_checkpoint_to_kaggle(path)

    print("=" * 80)
    print(f"STARTING CURRICULUM V5 SFT (<= {args.steps:,} steps, <= {args.max_hours:.2f} h wall clock)")
    print(f"Device: {device} | Peak LR: {args.lr:.2e} -> {args.min_lr:.1e} | Warmup: {args.warmup_steps} | Shift k={args.k_rand_shift}")
    print(f"Batch {args.batch_size}: math {n_m} + search {n_s} + tools {n_t} | CALL loss weight: {args.call_loss_weight}")
    print(f"Held-out: {len(heldout)} traces + {len(math_heldout)} long additions | Benchmark holdout: {len(prompts)} prompts, {len(spans)} free values")
    print(f"Training corpus (for audit_contamination): {corpus_path}")
    print("=" * 80 + "\n", flush=True)

    t0 = time.time()
    best_key: Tuple[float, float] = (-1.0, -1.0)
    step = 0
    while step < args.steps:
        elapsed = time.time() - t0
        progress = max(step / args.steps, elapsed / budget_s)
        if progress >= 1.0:
            print(f"[INFO] Schedule complete at step {step} ({elapsed / 3600.0:.2f} h).", flush=True)
            break
        step += 1
        lr = cosine_lr(progress, step, args.lr, args.min_lr, args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if m_idx + n_m > len(math_ds):
            rng_m.shuffle(math_ds)
            m_idx = 0
        batch_m = math_ds[m_idx : m_idx + n_m]
        m_idx += n_m
        x_b, y_b = encode_joint_batch(batch_m, stream.tools(n_t) + stream.search(n_s), tokenizer, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_amp):
            logits, _ = model(x_b)
        loss = call_weighted_loss(logits, x_b, y_b, tokenizer.call_s_id, tokenizer.call_e_id, args.call_loss_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_val = float(loss.item())

        if step == 1 or step % args.log_interval == 0:
            print(
                f"[SFT V5] Step {step:6d} | Loss: {loss_val:.4f} | LR: {lr:.2e} | "
                f"Progress: {100.0 * progress:5.1f}% | Elapsed: {elapsed / 60.0:.1f} min | Rejected: {stream.rejected}",
                flush=True,
            )
        if step == 1 or step % args.probe_interval == 0:
            run_sft_probe(model, tokenizer, device, step)
        if step % args.save_interval == 0:
            t_eval = time.time()
            acc = run_heldout_eval(model, tokenizer, device, heldout)
            math_acc = run_math_heldout(model, tokenizer, device, math_heldout)
            print(
                f"[HELDOUT] Step {step} | Tools exact: {100.0 * acc:.1f}% | Math exact: {100.0 * math_acc:.0f}% | "
                f"best tools {100.0 * max(best_key[1], 0.0):.1f}% | eval {time.time() - t_eval:.0f}s",
                flush=True,
            )
            if math_acc < 1.0:
                print(f"[WARN] Arithmetic regression on held-out long additions at step {step}.", flush=True)
            meta = {"loss": loss_val, "heldout_acc": acc, "math_heldout_acc": math_acc, "curriculum": "v5"}
            save(save_path.replace(".pt", f"_step_{step}.pt"), step, meta)
            prune_checkpoints(ckpt_dir, keep_last=args.max_keep_ckpts, base_filename=base_name)
            # Best = highest tool accuracy among checkpoints with the highest held-out arithmetic
            # (identical to "math == 100%, then max tools" whenever math reaches 100%)
            if (math_acc, acc) >= best_key:
                best_key = (math_acc, acc)
                save(best_path, step, meta)
                print(f"[OK] New best held-out checkpoint: {best_path}", flush=True)

    stream.close()
    acc = run_heldout_eval(model, tokenizer, device, heldout)
    math_acc = run_math_heldout(model, tokenizer, device, math_heldout)
    print(f"[HELDOUT] Final step {step} | Tools exact: {100.0 * acc:.1f}% | Math exact: {100.0 * math_acc:.0f}%", flush=True)
    meta = {"heldout_acc": acc, "math_heldout_acc": math_acc, "curriculum": "v5"}
    save(save_path, step, meta)
    if (math_acc, acc) >= best_key:
        best_key = (math_acc, acc)
        save(best_path, step, meta)
    if best_key[0] < 1.0:
        print(f"[WARN] No checkpoint reached exact held-out arithmetic (best {100.0 * best_key[0]:.0f}%); _best.pt holds the highest.", flush=True)
    run_sft_probe(model, tokenizer, device, step)
    print(f"\n[OK] Curriculum V5 SFT complete. Final: {save_path} | Best: {best_path}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="ScaledGPT Universal Robust SFT Engine (Curriculum V4 / V5)")
    parser.add_argument("--steps", type=int, default=12000, help="SFT steps (default 12000)")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="Peak SFT learning rate for annealing")
    parser.add_argument("--k-rand-shift", type=int, default=45, help="Random shift coordinate horizon (default 45)")
    parser.add_argument("--max-hours", type=float, default=10.0, help="Max training hours (default 10.0)")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging step interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint save step interval")
    parser.add_argument("--probe-interval", type=int, default=500, help="Probe test step interval")
    parser.add_argument("--max-keep-ckpts", type=int, default=2, help="Checkpoints to retain")
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default="checkpoints/cbt/cbt_18m_2d_rope_R3_k32_base.pt",
        help="Base checkpoint path",
    )
    parser.add_argument(
        "--save-ckpt",
        type=str,
        default="checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sftv4.pt",
        help="Target checkpoint path",
    )
    parser.add_argument("--curriculum", type=str, default="v4", choices=["v4", "v5"], help="v4 produces the champion; v5 is the decontaminated broad-copy curriculum")
    parser.add_argument("--min-lr", type=float, default=2.0e-6, help="V5 cosine floor")
    parser.add_argument("--warmup-steps", type=int, default=200, help="V5 linear warmup steps")
    parser.add_argument("--call-loss-weight", type=float, default=2.0, help="V5 loss weight on <CALL> argument targets")
    parser.add_argument("--math-frac", type=float, default=0.20, help="V5 math replay fraction per batch")
    parser.add_argument("--search-frac", type=float, default=0.35, help="V5 search fraction per batch")
    parser.add_argument("--heldout-size", type=int, default=64, help="V5 generated held-out traces for checkpoint selection")
    parser.add_argument("--seed", type=int, default=2026, help="V5 stream seed")
    parser.add_argument("--pos-emb", type=str, default=None, choices=["rope_2d", "rope_1d"], help="Expected position encoding; aborts if the resolved base checkpoint differs")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = ScaledTokenizer()

    # Resolve base checkpoint (pre-SFT base first, then Stage 1 pretraining)
    resolved_base = None
    candidates = [
        args.base_ckpt,
        os.path.join(ROOT_DIR, args.base_ckpt),
        os.path.join(ROOT_DIR, "checkpoints/cbt/cbt_18m_2d_rope_R3_k32_base.pt"),
        "/kaggle/working/checkpoints/scaled_t4/champion_base.pt",
        os.path.join(ROOT_DIR, "checkpoints/cbt/cbt_18m_2d_rope_R3_k32_pretrain.pt"),
    ]
    for c in candidates:
        if os.path.exists(c) and not os.path.isdir(c):
            resolved_base = c
            break

    ckpt_data = torch.load(resolved_base, map_location=device, weights_only=False) if resolved_base else {}
    # Position encoding is inherited from the base checkpoint (rope_2d or the rope_1d ablation)
    pos_emb_type = getattr(ckpt_data.get("config"), "pos_emb_type", "rope_2d")
    if args.pos_emb is not None and (resolved_base is None or pos_emb_type != args.pos_emb):
        raise SystemExit(f"[FAIL] --pos-emb {args.pos_emb} but resolved base is {resolved_base} ({pos_emb_type})")
    config = ScaledGPTConfig(vocab_size=tokenizer.vocab_size, k_rand_shift=args.k_rand_shift, pos_emb_type=pos_emb_type)
    model = ScaledGPT(config, tokenizer=tokenizer).to(device)
    print(f"[INFO] Position encoding: {pos_emb_type}")

    if resolved_base is not None:
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.config.k_rand_shift = args.k_rand_shift
        step_tag = f" (from step {ckpt_data.get('step')})" if "step" in ckpt_data else ""
        print(f"[INFO] Successfully loaded base weights{step_tag} from: {resolved_base}")
    else:
        print(f"[WARN] Base checkpoint not found. Starting from scratch.")

    # Multi-GPU DataParallel wrapper if 2+ GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[INFO] Detected {torch.cuda.device_count()} GPUs. Wrapping with DataParallel.")
        model = nn.DataParallel(model)

    if args.curriculum == "v5":
        train_v5(args, model, tokenizer, device)
        return

    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.steps), eta_min=2e-6)
    use_cuda_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    print(f"\n[INFO] Loading SFT datasets (Curriculum V4: Multi-Tool 45% + Search Copy 35% + Math Replay 20%)...")
    data_cache = os.path.join(ROOT_DIR, "champion_model/scaled_t4/data/sft_dataset_v4.pt")
    if os.path.exists(data_cache):
        cached = torch.load(data_cache, map_location="cpu", weights_only=False)
        if cached.get("version") == "v4_universal_robust":
            print(f"[INFO] Found V4 Universal Robust pre-packaged dataset: {data_cache}")
            search_ds = cached["search"]
            math_ds = cached["math"]
            tool_ds = cached["tools"]
        else:
            print(f"[INFO] Upgrading dataset cache to V4 Universal Robust Curriculum...")
            search_ds = generate_search_delegation_dataset(n_samples=15000, seed=42)
            math_ds = generate_arithmetic_equations(n_samples=6000, seed=777, max_train_digits=25)
            tool_ds = generate_tool_dataset(n_samples=25000, seed=888)
            os.makedirs(os.path.dirname(data_cache), exist_ok=True)
            torch.save({
                "version": "v4_universal_robust",
                "search": search_ds,
                "math": math_ds,
                "tools": tool_ds,
            }, data_cache)
        print(f"       Loaded {len(tool_ds):,} multi-tool, {len(search_ds):,} search, and {len(math_ds):,} math samples.")
    else:
        print(f"[INFO] Generating V4 Universal Robust SFT dataset on-the-fly...")
        search_ds = generate_search_delegation_dataset(n_samples=15000, seed=42)
        math_ds = generate_arithmetic_equations(n_samples=6000, seed=777, max_train_digits=25)
        tool_ds = generate_tool_dataset(n_samples=25000, seed=888)
        os.makedirs(os.path.dirname(data_cache), exist_ok=True)
        torch.save({
            "version": "v4_universal_robust",
            "search": search_ds,
            "math": math_ds,
            "tools": tool_ds,
        }, data_cache)
        print(f"       Generated and cached {len(tool_ds):,} multi-tool, {len(search_ds):,} search, and {len(math_ds):,} math samples.")

    save_path = os.path.join(ROOT_DIR, args.save_ckpt)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    prune_checkpoints(ckpt_dir, keep_last=args.max_keep_ckpts, base_filename=base_name)

    print("=" * 80)
    print(f"STARTING UNIVERSAL ROBUST ANNEALING SFT ({args.steps:,} steps)")
    print(f"Device: {device} | Peak LR: {args.lr:.2e} -> 2.0e-06 | Batch Size: {args.batch_size} | Shift k={args.k_rand_shift}")
    print(f"Curriculum V4: 45% Multi-Tool + 35% Search Copy + 20% Math Replay (1-25 Digits, k={args.k_rand_shift})")
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

        # Batch Composition (32): Math 6 (18.75%), Search 11 (34.38%), Multi-Tools 15 (46.88%)
        n_m = max(1, int(args.batch_size * 0.20))
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

    # Save final SFT model
    raw_model = model.module if hasattr(model, "module") else model
    final_payload = {
        "model_state_dict": raw_model.state_dict(),
        "config": raw_model.config,
        "steps": args.steps,
    }
    torch.save(final_payload, save_path)
    sync_checkpoint_to_kaggle(save_path)
    print(f"\n[OK] Final All-Tools Annealed SFT model saved: {save_path}\n", flush=True)


if __name__ == "__main__":
    main()
