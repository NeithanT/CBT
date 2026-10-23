import os
import sys
import argparse
from typing import List, Optional, Any
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.tokenizer import MathTokenizer
from src.bpe_tokenizer import BPEMathTokenizer
from src.model import MathGPT, GPTConfig, create_model
from src.constants import get_device


def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")

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
    return model, tokenizer, config


def format_token(token: str) -> str:
    if token == " ":
        return "[SPC]"
    if token == "\n":
        return "[\\n]"
    return token


def render_ascii_heatmap(
    tokens: List[str],
    attn_matrix: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
):
    seq_len = len(tokens)
    print(f"\n--- Attention Weights (Layer {layer_idx + 1}, Head {head_idx + 1}) ---")
    header = "       " + " ".join(f"{format_token(t):>4}" for t in tokens)
    print(header)
    print("       " + "----" * seq_len)

    shade_chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
    for i in range(seq_len):
        row_str = f"{format_token(tokens[i]):>5} |"
        for j in range(seq_len):
            val = attn_matrix[i, j].item()
            idx = min(int(val * len(shade_chars)), len(shade_chars) - 1)
            char = shade_chars[idx]
            row_str += f"   {char}"
        print(row_str)
    print("--------------------------------------------------\n")


def generate_completion(
    model: MathGPT,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    show_attention: bool = False,
    attn_layer: int = 0,
    attn_head: int = 0,
) -> str:
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    eos_token_id = tokenizer.eos_id
    out_tokens = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_id=eos_token_id,
        save_attention=show_attention,
    )

    generated_ids = out_tokens[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special=False)

    if show_attention:
        attn_maps = model.get_attention_maps()
        if attn_maps and len(attn_maps) > attn_layer:
            layer_attn = attn_maps[attn_layer]
            token_strs = [
                format_token(tokenizer.decode([tid], skip_special=False))
                for tid in generated_ids[: layer_attn.size(-1)]
            ]
            head_weights = layer_attn[0, attn_head]
            render_ascii_heatmap(
                token_strs, head_weights, layer_idx=attn_layer, head_idx=attn_head
            )

    return generated_text


def interactive_session(model: MathGPT, tokenizer: Any, device: torch.device):
    print("MathGPT Interactive Console (Type 'exit' to quit)")
    print("-" * 50)
    while True:
        try:
            prompt = input("Prompt >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.lower() in ("exit", "quit", "q"):
            break
        if not prompt:
            continue

        result = generate_completion(model, tokenizer, prompt, device, max_new_tokens=64)
        print(f"Response: {result}\n")


def main():
    parser = argparse.ArgumentParser(description="ScaledGPT & MathGPT Interactive Inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt execution")
    parser.add_argument("--show_attention", action="store_true", help="Display ASCII attention map (MathGPT only)")
    parser.add_argument("--layer", type=int, default=0, help="Attention layer to visualize")
    parser.add_argument("--head", type=int, default=0, help="Attention head to visualize")
    args = parser.parse_args()

    champion_ckpt = os.path.join(ROOT_DIR, "champion_model", "scaled_t4", "checkpoints", "champion_v2_final.pt")

    # If no specific checkpoint is passed, try champion model or pull from Hugging Face
    if args.checkpoint is None:
        if not os.path.exists(champion_ckpt):
            try:
                from huggingface_hub import hf_hub_download
                print("[INFO] Champion weights not found locally. Pulling from Hugging Face (Neithanvarvas/CBT)...")
                champion_ckpt = hf_hub_download(
                    repo_id="Neithanvarvas/CBT",
                    filename="champion_v2_final.pt",
                    local_dir=os.path.dirname(champion_ckpt)
                )
            except Exception as e:
                print(f"[WARN] Automatic Hugging Face pull unsuccessful: {e}")

        if os.path.exists(champion_ckpt):
            from champion_model.scaled_t4.interactive import (
                ScaledTokenizer,
                ScaledGPTConfig,
                ScaledGPT,
                run_math_prompt,
                run_dialogue_prompt,
            )
            import re

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            tokenizer = ScaledTokenizer()
            config = ScaledGPTConfig(vocab_size=tokenizer.vocab_size)
            model = ScaledGPT(config, tokenizer=tokenizer).to(device)

            ckpt = torch.load(champion_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            print(f"[INFO] Loaded ScaledGPT Champion Model from {champion_ckpt}")

            if args.prompt:
                p = args.prompt.strip()
                is_math = bool(re.match(r"^[\d\s\+\-\=]+$", p) and ("+" in p or "-" in p))
                if is_math:
                    run_math_prompt(model, tokenizer, p, device)
                else:
                    run_dialogue_prompt(model, tokenizer, p, device)
                return

            print("\n" + "=" * 80)
            print("CHARACTER BASED TRANSFORMER (CBT, ~18.95M) INTERACTIVE TERMINAL")
            print("Type an arithmetic equation (e.g. '12345+67890=') or a natural question.")
            print("Type 'exit' or 'quit' to terminate.")
            print("=" * 80 + "\n")

            while True:
                try:
                    user_input = input("User >> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[INFO] Exiting interactive CLI.")
                    break
                if not user_input or user_input.lower() in ["exit", "quit"]:
                    break
                is_math = bool(re.match(r"^[\d\s\+\-\=]+$", user_input) and ("+" in user_input or "-" in user_input))
                if is_math:
                    run_math_prompt(model, tokenizer, user_input, device)
                else:
                    run_dialogue_prompt(model, tokenizer, user_input, device)
            return

    # Fallback to legacy MathGPT checkpoint loader
    ckpt = args.checkpoint or os.path.join(ROOT_DIR, "checkpoints", "baseline", "math_gpt.pt")
    if not os.path.exists(ckpt):
        for candidate in [
            os.path.join(ROOT_DIR, "checkpoints", "baseline", "math_gpt.pt"),
            os.path.join(ROOT_DIR, "checkpoints", "model_1", "model_1_final_8h.pt"),
            os.path.join(ROOT_DIR, "math_gpt.pt"),
        ]:
            if os.path.exists(candidate):
                ckpt = candidate
                break

    device = get_device()
    try:
        model, tokenizer, config = load_model(ckpt, device)
    except FileNotFoundError:
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        return

    if args.prompt:
        output = generate_completion(
            model,
            tokenizer,
            args.prompt,
            device,
            show_attention=args.show_attention,
            attn_layer=args.layer,
            attn_head=args.head,
        )
        print(f"Prompt:   {args.prompt}")
        print(f"Response: {output}")
    else:
        interactive_session(model, tokenizer, device)


if __name__ == "__main__":
    main()
