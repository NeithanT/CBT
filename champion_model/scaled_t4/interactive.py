#!/usr/bin/env python3
"""
Interactive CLI Runner for ScaledGPT (~18.95M Parameter Recurrent 2D-RoPE Transformer).
Supports live interactive inference across:
1. High-digit column scratchpad arithmetic (up to 50 digits).
2. Expanded tool execution (Single-tool, Chaining, Nested JSON schemas, Error handling).
3. Multi-turn interactive dialogues.
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
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    from champion_model.scaled_t4.model import ScaledGPT, ScaledGPTConfig
    from champion_model.scaled_t4.tokenizer import ScaledTokenizer
    from champion_model.scaled_t4.dataset import extract_scratchpad_answer
except ImportError:
    from model import ScaledGPT, ScaledGPTConfig
    from tokenizer import ScaledTokenizer
    from dataset import extract_scratchpad_answer


def mock_execute_tool(call_text: str) -> str:
    """Executes or mocks tool calls parsed from <CALL>...</CALL>."""
    # 1. calculator(expr="...")
    calc_m = re.match(r'calculator\(expr="([^"]+)"\)', call_text)
    if calc_m:
        expr = calc_m.group(1)
        try:
            allowed = re.match(r"^[\d\+\-\*\/\(\)\s\.]+$", expr)
            if allowed:
                val = eval(expr)
                return json.dumps({"result": val})
        except ZeroDivisionError:
            return json.dumps({"error": "ZeroDivisionError: division by zero"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # 2. get_weather(city="...")
    w_m = re.match(r'get_weather\(city="([^"]+)"\)', call_text)
    if w_m:
        city = w_m.group(1)
        if city.lower() in ["atlantis", "narnia"]:
            return json.dumps({"status": "error", "code": 404, "message": f"City '{city}' not found"})
        return json.dumps({"temp": 22, "condition": "Sunny", "city": city})

    # 3. get_time(timezone="...")
    t_m = re.match(r'get_time\(timezone="([^"]+)"\)', call_text)
    if t_m:
        tz = t_m.group(1)
        return json.dumps({"time": "14:30", "timezone": tz})

    # 4. unit_convert
    u_m = re.match(r'unit_convert\(value=([\d\.]+),\s*from_unit="([^"]+)",\s*to_unit="([^"]+)"\)', call_text)
    if u_m:
        val = float(u_m.group(1))
        f_u, t_u = u_m.group(2), u_m.group(3)
        factors = {
            ("km", "miles"): 0.621371,
            ("miles", "km"): 1.60934,
            ("kg", "lbs"): 2.20462,
            ("lbs", "kg"): 0.453592,
        }
        factor = factors.get((f_u, t_u), 1.0)
        return json.dumps({"value": round(val * factor, 2), "unit": t_u})

    # 5. database_query
    if call_text.startswith("database_query("):
        return json.dumps({"status": "success", "count": 3, "records": [{"id": 101, "status": "active"}]})

    # 6. create_event
    if call_text.startswith("create_event("):
        return json.dumps({"event_id": "evt_98214", "status": "confirmed"})

    # 7. file_manager
    if call_text.startswith("file_manager("):
        return json.dumps({"status": "ok", "bytes_written": 64})

    # 8. fetch_api
    if call_text.startswith("fetch_api("):
        return json.dumps({"status": 200, "data": {"status": "healthy"}})

    # 9. search(query="...")
    s_m = re.match(r'search\(query="([^"]+)"\)', call_text)
    if s_m:
        query = s_m.group(1).strip()
        q_lower = query.lower()

        # Offline dictionary lookup for instant, deterministic testing
        kb = {
            "tomato": "the edible berry of the plant Solanum lycopersicum, commonly prepared as a culinary vegetable.",
            "tomatoes": "the edible berry of the plant Solanum lycopersicum, commonly prepared as a culinary vegetable.",
            "apple": "a round, edible fruit produced by an apple tree (Malus domestica), cultivated worldwide.",
            "apples": "a round, edible fruit produced by an apple tree (Malus domestica), cultivated worldwide.",
            "potato": "a starchy, tuberous root crop of the nightshade family Solanaceae.",
            "potatoes": "a starchy, tuberous root crop of the nightshade family Solanaceae.",
            "coffee": "a brewed drink prepared from roasted coffee beans of flowering Coffea species.",
            "banana": "an elongated, edible fruit produced by several kinds of large herbaceous plants in the genus Musa.",
            "bananas": "an elongated, edible fruit produced by several kinds of large herbaceous plants in the genus Musa.",
            "gravity": "a fundamental interaction causing mutual attraction between all things with mass or energy.",
            "photosynthesis": "a process used by plants to convert light energy into chemical energy stored in glucose.",
            "atom": "the basic unit of a chemical element, consisting of a dense central nucleus surrounded by electrons.",
            "python creator": "Guido van Rossum, a Dutch programmer.",
            "speed of light": "approximately 299,792 kilometers per second in a vacuum.",
            "mount everest": "Earth's highest mountain above sea level, located in the Himalayas.",
        }

        for k, v in kb.items():
            if k in q_lower or q_lower in k:
                return json.dumps({"snippet": v})

        # Try live Wikipedia REST API if network is available
        try:
            import urllib.request
            import urllib.parse
            clean_q = urllib.parse.quote(query.replace(" ", "_"))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_q}"
            req = urllib.request.Request(url, headers={"User-Agent": "ScaledGPT-Search/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "extract" in data and data["extract"]:
                    return json.dumps({"snippet": data["extract"][:220]})
        except Exception:
            pass

        return json.dumps({"snippet": f"{query.capitalize()} is an entity and topic in verified knowledge sources."})

    # Fallback
    return json.dumps({"status": "executed", "call": call_text})


def print_human_readable_output(text: str, is_math: bool = False):
    """Formats internal model output into clean, human-readable terminal blocks."""
    print("\n" + "-" * 80)
    if is_math:
        m = re.search(r"(?:<BOS>)?(.*?)=(?:<THINK>)?(.*?)(?:</THINK>=)(.*?)(?:<EOS>|$)", text)
        if m:
            problem = m.group(1).strip()
            scratch = m.group(2).strip()
            ans = m.group(3).strip()
            print(f"Problem:    {problem} = ?")
            print(f"Scratchpad: {scratch}")
            print(f"Answer:     {ans}")
            try:
                if "+" in problem:
                    parts = problem.split("+")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        expected = int(parts[0]) + int(parts[1])
                        status = "[PASS]" if str(expected) == ans else "[FAIL]"
                        print(f"Validation: {status} (Expected: {expected})")
            except Exception:
                pass
        else:
            print(f"Result:     {text}")
    else:
        think_m = re.search(r"<THINK>(.*?)</THINK>", text)
        call_m = re.search(r"<CALL>(.*?)</CALL>", text)
        result_m = re.search(r"<RESULT>(.*?)</RESULT>", text)
        answer_m = re.search(r"<ANSWER>(.*?)</ANSWER>", text)

        if think_m:
            print(f"Reasoning:   {think_m.group(1).strip()}")
        if call_m:
            print(f"Tool Call:   {call_m.group(1).strip()}")
        if result_m:
            print(f"Tool Result: {result_m.group(1).strip()}")
        if answer_m:
            print(f"Response:    {answer_m.group(1).strip()}")
        if not any([think_m, call_m, result_m, answer_m]):
            print(f"Response:    {text}")

    print("-" * 80)
    print(f"[RAW TOKENS] {text}\n", flush=True)


def run_math_prompt(model: ScaledGPT, tokenizer: ScaledTokenizer, prompt: str, device: torch.device) -> str:
    """Processes a multi-digit arithmetic expression."""
    clean_prompt = prompt.strip()
    if not clean_prompt.endswith("="):
        clean_prompt += "="
    full_prompt = clean_prompt + "<THINK>"

    tokens = tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(x, max_new_tokens=700, temperature=0.0, stop_tokens=[tokenizer.eos_id])

    gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)
    print_human_readable_output(gen_text, is_math=True)
    return gen_text


def run_dialogue_prompt(model: ScaledGPT, tokenizer: ScaledTokenizer, user_input: str, device: torch.device) -> str:
    """Processes a user dialogue turn with live tool execution."""
    prompt = f"<USER>{user_input}</USER><THINK>"
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        stop_ids = [tokenizer.call_e_id, tokenizer.answer_e_id, tokenizer.eos_id]
        out = model.generate(x, max_new_tokens=256, temperature=0.0, stop_tokens=stop_ids)

    gen_text = tokenizer.decode(out[0].tolist(), skip_special=False)

    if "<CALL>" in gen_text and "</CALL>" in gen_text:
        call_part = gen_text.split("<CALL>")[1].split("</CALL>")[0]
        result_str = mock_execute_tool(call_part)

        # Continue generation with the tool result (append tokens directly to avoid double BOS)
        res_suffix = f"<RESULT>{result_str}</RESULT>"
        res_tokens = tokenizer.encode(res_suffix, add_bos=False, add_eos=False)
        c_x = torch.cat([out, torch.tensor([res_tokens], dtype=torch.long, device=device)], dim=1)

        with torch.no_grad():
            out2 = model.generate(c_x, max_new_tokens=256, temperature=0.0, stop_tokens=[tokenizer.answer_e_id, tokenizer.eos_id])

        full_dialogue = tokenizer.decode(out2[0].tolist(), skip_special=False)
    else:
        full_dialogue = gen_text

    print_human_readable_output(full_dialogue, is_math=False)
    return full_dialogue


def main():
    parser = argparse.ArgumentParser(description="ScaledGPT Interactive CLI")
    parser.add_argument("--ckpt", type=str, default="champion_model/scaled_t4/checkpoints/champion_final.pt", help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Execute a single prompt and exit")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads for inference")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    tokenizer = ScaledTokenizer()
    config = ScaledGPTConfig(vocab_size=tokenizer.vocab_size)
    model = ScaledGPT(config, tokenizer=tokenizer).to(device)

    # Search candidate paths for the trained checkpoint
    candidates = [
        args.ckpt,
        os.path.join(ROOT_DIR, args.ckpt),
        os.path.join(ROOT_DIR, "champion_model/scaled_t4/checkpoints/champion_final.pt"),
        os.path.join(ROOT_DIR, "champion_model/checkpoints/champion_final.pt"),
        os.path.join(ROOT_DIR, "champion_model/scaled_t4/checkpoints/scaled_t4_sft_all_tools.pt"),
        os.path.join(ROOT_DIR, "champion_model/scaled_t4/checkpoints/scaled_t4_sft_search.pt"),
        os.path.join(ROOT_DIR, "champion_model/scaled_t4/checkpoints/scaled_t4_final.pt"),
        os.path.join(ROOT_DIR, "champion_model/scaled_t4/checkpoints/scaled_t4_final_timeout_step_45059.pt"),
        os.path.join(ROOT_DIR, "champion_model/checkpoints/scaled_t4_sft_search_step_7000.zip"),
        os.path.join(ROOT_DIR, "champion_model/checkpoints/scaled_t4_final.zip"),
        os.path.join(ROOT_DIR, "champion_model/checkpoints/scaled_t4_final_timeout_step_45059.zip"),
    ]
    resolved_path = None
    for p in candidates:
        if os.path.exists(p) and not os.path.isdir(p):
            resolved_path = p
            break

    if resolved_path is not None:
        ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        step_info = f" (Step {ckpt.get('step')})" if 'step' in ckpt else ""
        print(f"[INFO] Successfully loaded trained checkpoint{step_info} from: {resolved_path}")
    else:
        print(f"[WARN] Checkpoint not found in candidate paths. Running with initialized weights.")

    model.eval()

    if args.prompt:
        user_input = args.prompt.strip()
        is_math = bool(re.match(r"^[\d\s\+\-\=]+$", user_input) and ("+" in user_input or "-" in user_input))
        if is_math:
            run_math_prompt(model, tokenizer, user_input, device)
        else:
            run_dialogue_prompt(model, tokenizer, user_input, device)
        return

    print("\n" + "=" * 80)
    print("SCALEDGPT (~18.95M) INTERACTIVE TERMINAL")
    print("Type an arithmetic equation (e.g. '123456789+987654321=') or ask a question.")
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

        # Check if input is arithmetic (contains only digits and + or -)
        is_math = bool(re.match(r"^[\d\s\+\-\=]+$", user_input) and ("+" in user_input or "-" in user_input))
        if is_math:
            run_math_prompt(model, tokenizer, user_input, device)
        else:
            run_dialogue_prompt(model, tokenizer, user_input, device)


if __name__ == "__main__":
    main()
