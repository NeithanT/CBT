#!/usr/bin/env python3
"""
Interactive CLI Runner for the Unified ~200k Champion Model.
Supports both multi-digit column arithmetic and natural language tool routing with live execution.
"""

import os
import sys
import re
import json
import time
import argparse
from typing import Tuple, Optional, Dict, Any

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.unified_200k.model import UnifiedGPT, UnifiedGPTConfig
from champion_model.unified_200k.tokenizer import UnifiedTokenizer
from champion_model.math_model.arithmetic import extract_scratchpad_answer


def mock_execute_tool(call_text: str) -> str:
    """Executes or mocks tool calls parsed from <CALL>...</CALL>."""
    # 1. calculator(expr="...")
    calc_m = re.match(r'calculator\(expr="([^"]+)"\)', call_text)
    if calc_m:
        expr = calc_m.group(1)
        try:
            # Safe evaluation for basic arithmetic
            allowed = re.match(r"^[\d\+\-\*\/\(\)\s\.]+$", expr)
            if allowed:
                val = eval(expr)
                return json.dumps({"result": val})
        except Exception:
            pass
        return json.dumps({"result": "error"})

    # 2. get_weather(city="...")
    w_m = re.match(r'get_weather\(city="([^"]+)"\)', call_text)
    if w_m:
        city = w_m.group(1)
        return json.dumps({"temp": 21, "condition": "sunny"})

    # 3. get_time(timezone="...")
    t_m = re.match(r'get_time\(timezone="([^"]+)"\)', call_text)
    if t_m:
        tz = t_m.group(1)
        return json.dumps({"time": "14:30"})

    # 4. unit_convert(value=..., from_unit="...", to_unit="...")
    u_m = re.match(r'unit_convert\(value=([\d\.]+),\s*from_unit="([^"]+)",\s*to_unit="([^"]+)"\)', call_text)
    if u_m:
        val = float(u_m.group(1))
        f_u, t_u = u_m.group(2), u_m.group(3)
        factors = {
            ("km", "miles"): 0.62,
            ("miles", "km"): 1.61,
            ("kg", "lbs"): 2.20,
            ("lbs", "kg"): 0.45,
            ("meters", "feet"): 3.28,
            ("feet", "meters"): 0.30,
            ("celsius", "fahrenheit"): 1.8,
        }
        factor = factors.get((f_u, t_u), 1.0)
        return json.dumps({"value": round(val * factor, 2), "unit": t_u})

    # 5. search(query="...")
    s_m = re.match(r'search\(query="([^"]+)"\)', call_text)
    if s_m:
        query = s_m.group(1)
        return json.dumps({"snippet": f"Verified factual data regarding {query}"})

    return json.dumps({"status": "executed"})


def run_math_query(model: UnifiedGPT, tokenizer: UnifiedTokenizer, query: str, device: torch.device) -> None:
    """Runs a multi-digit arithmetic query with column scratchpad breakdown."""
    prompt = query.strip()
    if not prompt.endswith("="):
        prompt += "="
    prompt_with_think = prompt + "<THINK>"

    tokens = tokenizer.encode(prompt_with_think, add_bos=True, add_eos=False)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=320, temperature=0.0, stop_tokens=[tokenizer.eos_id])
    elapsed = time.time() - t0

    full_gen = tokenizer.decode(out[0].tolist(), skip_special=False)
    reasoning, answer = extract_scratchpad_answer(full_gen)

    print("\n--- Model Output (Arithmetic) ---")
    print(f"Problem:             {prompt}")
    print(f"Internal Scratchpad: {full_gen[len('<BOS>' + prompt):]}")
    print(f"Final Answer:        {answer}")
    print(f"Latency:             {elapsed:.2f}s\n")


def run_tool_query(model: UnifiedGPT, tokenizer: UnifiedTokenizer, user_input: str, device: torch.device) -> None:
    """Runs a natural language query with live tool selection, call, and answering."""
    prompt = f"<USER>{user_input.strip()}</USER><THINK>"
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    # Step 1: Generate thought + potential tool call
    t0 = time.time()
    with torch.no_grad():
        stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
        out1 = model.generate(x, max_new_tokens=120, temperature=0.0, stop_tokens=stop_ids)

    gen1 = tokenizer.decode(out1[0].tolist(), skip_special=False)

    # Check if a tool call was emitted
    if "<CALL>" in gen1:
        call_part = gen1.split("<CALL>")[1].split("</CALL>")[0]
        result_json = mock_execute_tool(call_part)

        # Feed back <RESULT>...</RESULT> and generate final answer
        prompt_with_result = gen1.rstrip("<EOS>")
        if not prompt_with_result.endswith("</CALL>"):
            prompt_with_result += "</CALL>"
        prompt_with_result += f"<RESULT>{result_json}</RESULT>"

        tokens2 = tokenizer.encode(prompt_with_result, add_bos=False, add_eos=False)
        x2 = torch.tensor([tokens2], dtype=torch.long, device=device)
        with torch.no_grad():
            out2 = model.generate(x2, max_new_tokens=100, temperature=0.0, stop_tokens=[tokenizer.answer_e_id, tokenizer.eos_id])
        full_dialogue = tokenizer.decode(out2[0].tolist(), skip_special=False)
    else:
        full_dialogue = gen1

    elapsed = time.time() - t0

    # Extract clean answer
    answer_text = ""
    if "<ANSWER>" in full_dialogue:
        answer_text = full_dialogue.split("<ANSWER>")[1].split("</ANSWER>")[0]
    elif "</THINK>" in full_dialogue:
        answer_text = full_dialogue.split("</THINK>")[1]
    else:
        answer_text = full_dialogue[len(prompt):]

    print("\n--- Model Output (Tool Calling & Dialogue) ---")
    print(f"User Input:     {user_input}")
    print(f"Full Trace:     {full_dialogue[len('<BOS>'):]}")
    print(f"Assistant:      {answer_text.strip()}")
    print(f"Latency:        {elapsed:.2f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for Unified ~200k Model")
    parser.add_argument("--ckpt", type=str, default="champion_model/unified_200k/checkpoints/unified_200k.pt")
    parser.add_argument("--config", type=str, default="champion_model/unified_200k/config_200k.json")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)

    print("\n" + "=" * 70)
    print("UNIFIED ~200K CHAMPION MODEL - INTERACTIVE CLI")
    print("=" * 70)

    cfg_path = os.path.join(ROOT_DIR, args.config)
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    valid_keys = {k: v for k, v in cfg_dict.items() if k in UnifiedGPTConfig.__dataclass_fields__}
    config = UnifiedGPTConfig(**valid_keys)

    tokenizer = UnifiedTokenizer()
    model = UnifiedGPT(config, tokenizer=tokenizer)

    ckpt_path = os.path.join(ROOT_DIR, args.ckpt)
    if not os.path.exists(ckpt_path):
        print(f"[FAIL] Checkpoint not found at: {ckpt_path}")
        return

    print(f"[INFO] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("[INFO] Model successfully loaded! (204,864 parameters)")
    print("[INFO] Type any question, tool request, or arithmetic equation.")
    print("[INFO] Examples:")
    print("       - Arithmetic: '15+25=' or '12345678901234+87654321098765='")
    print("       - Tool Calling: 'Can you check the forecast for Berlin?'")
    print("       - Tool Calling: 'What time is it in UTC?'")
    print("       - Tool Calling: 'Convert 40 lbs to kg.'")
    print("       - Dialogue: 'Hello! How are you?'")
    print("       - Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # Check if pure arithmetic problem (e.g. 123+456= or 123+456)
        is_math = bool(re.match(r"^\d+\+\d+=?$", user_input.replace(" ", "")))

        if is_math:
            run_math_query(model, tokenizer, user_input, device=device)
        else:
            run_tool_query(model, tokenizer, user_input, device=device)


if __name__ == "__main__":
    main()
