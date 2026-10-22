from typing import Tuple, List
import random


def format_addition_char_scratchpad(a: int, b: int) -> str:
    """Formats addition into column-by-column carry scratchpad for character tokenizer."""
    s_a, s_b = str(a), str(b)
    len_a, len_b = len(s_a), len(s_b)
    max_len = max(len_a, len_b)

    carry = 0
    steps, digits = [], []

    for col in range(max_len):
        da = int(s_a[len_a - 1 - col]) if col < len_a else 0
        db = int(s_b[len_b - 1 - col]) if col < len_b else 0
        col_sum = carry + da + db
        next_carry = col_sum // 10
        res_digit = col_sum % 10
        steps.append(f"({carry}+{da}+{db})={next_carry}{res_digit}")
        digits.append(str(res_digit))
        carry = next_carry

    if carry > 0:
        steps.append(f"({carry}+0+0)=0{carry}")
        digits.append(str(carry))

    ans = "".join(reversed(digits))
    return f"{a}+{b}=<THINK>" + "".join(steps) + f"</THINK>={ans}"


def format_addition_bpe_word_cot(a: int, b: int) -> str:
    """Formats addition into natural language Word-CoT for BPE tokenizer."""
    s_a, s_b = str(a), str(b)
    len_a, len_b = len(s_a), len(s_b)
    max_len = max(len_a, len_b)

    carry = 0
    steps, digits = [], []

    for col in range(max_len):
        da = int(s_a[len_a - 1 - col]) if col < len_a else 0
        db = int(s_b[len_b - 1 - col]) if col < len_b else 0
        col_sum = carry + da + db
        next_carry = col_sum // 10
        res_digit = col_sum % 10

        carry_str = f" + carry {carry}" if carry > 0 else ""
        steps.append(f"Column {col}: {da}+{db}{carry_str}={col_sum}, digit {res_digit}, carry {next_carry}.")
        digits.append(str(res_digit))
        carry = next_carry

    if carry > 0:
        steps.append(f"Column {max_len}: carry {carry} placed.")
        digits.append(str(carry))

    ans = "".join(reversed(digits))
    return f"{a}+{b}=<THINK>{' '.join(steps)}</THINK>={ans}"


def format_addition_place_value_thinking(a: int, b: int) -> str:
    """Formats addition into place-value zero-reduction thinking trace."""
    s_a, s_b = str(a), str(b)
    max_len = max(len(s_a), len(s_b))

    lines = []
    for k in range(max_len - 1, 0, -1):
        place = 10 ** k
        da = (a // place) % 10
        db = (b // place) % 10
        if da > 0:
            lines.append(str(da * place))
        if db > 0:
            lines.append(str(db * place))

    ua = a % 10
    ub = b % 10
    u_sum = ua + ub
    if u_sum >= 10:
        lines.append(str((u_sum // 10) * 10))
        lines.append(str(u_sum % 10))
    else:
        lines.append(str(u_sum))

    steps_str = "\n".join(lines)
    ans = a + b
    return f"{a}+{b}=<THINK>\n{steps_str}\n</THINK>={ans}"


def format_subtraction_char_scratchpad(a: int, b: int) -> str:
    """Formats subtraction (a - b with a >= b) into column borrow scratchpad."""
    if a < b:
        a, b = b, a
    s_a, s_b = str(a), str(b)
    len_a, len_b = len(s_a), len(s_b)

    borrow = 0
    steps, digits = [], []

    for col in range(len_a):
        da = int(s_a[len_a - 1 - col])
        db = int(s_b[len_b - 1 - col]) if col < len_b else 0
        cur_a = da - borrow
        if cur_a < db:
            cur_a += 10
            next_borrow = 1
        else:
            next_borrow = 0
        res_digit = cur_a - db
        steps.append(f"({borrow}:{da}-{db})={res_digit}")
        digits.append(str(res_digit))
        borrow = next_borrow

    ans = str(int("".join(reversed(digits))))
    return f"{a}-{b}=<THINK>" + "".join(steps) + f"</THINK>={ans}"


def format_multiplication_char_scratchpad(a: int, b: int) -> str:
    """Formats multiplication (a * b) with digit-wise partial products."""
    s_a, s_b = str(a), str(b)
    len_b = len(s_b)
    steps = []

    for idx, d_char in enumerate(reversed(s_b)):
        d = int(d_char)
        partial = a * d
        shift = "0" * idx
        steps.append(f"({a}*{d})={partial}{shift}")

    ans = str(a * b)
    return f"{a}*{b}=<THINK>" + "+".join(steps) + f"</THINK>={ans}"


def extract_scratchpad_answer(generated_text: str) -> Tuple[str, str]:
    """Extracts reasoning chain and final answer from generated text."""
    cleaned = generated_text.replace("<EOS>", "").replace("<PAD>", "").strip()
    if "</THINK>=" in cleaned:
        parts = cleaned.split("</THINK>=")
        reasoning = parts[0].replace("<THINK>", "").strip()
        final_answer = parts[1].strip()
    elif "</THINK>" in cleaned:
        parts = cleaned.split("</THINK>")
        reasoning = parts[0].replace("<THINK>", "").strip()
        final_answer = parts[1].lstrip("=").strip()
    elif "=" in cleaned:
        parts = cleaned.split("=")
        final_answer = parts[-1].strip()
        reasoning = "=".join(parts[:-1]).replace("<THINK>", "").strip()
    else:
        reasoning = cleaned
        final_answer = ""
    return reasoning, final_answer

