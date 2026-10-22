"""
Arithmetic scratchpad formatting and answer extraction utilities.
"""

from typing import Tuple


def format_addition_char_scratchpad(a: int, b: int) -> str:
    """
    Formats multi-digit addition with canonical column-by-column scratchpad.
    Example: 15 + 25 -> '15+25=<THINK>(0+5+5)=10(1+1+2)=04</THINK>=40'
    """
    s_a = str(a)
    s_b = str(b)
    rev_a = s_a[::-1]
    rev_b = s_b[::-1]

    max_len = max(len(rev_a), len(rev_b))
    carry = 0
    steps = []

    for i in range(max_len):
        da = int(rev_a[i]) if i < len(rev_a) else 0
        db = int(rev_b[i]) if i < len(rev_b) else 0
        col_sum = carry + da + db
        next_carry = col_sum // 10
        out_digit = col_sum % 10
        steps.append(f"({carry}+{da}+{db})={next_carry}{out_digit}")
        carry = next_carry

    if carry > 0:
        steps.append(f"({carry}+0+0)=0{carry}")

    scratchpad = "".join(steps)
    expected_ans = str(a + b)
    return f"{a}+{b}=<THINK>{scratchpad}</THINK>={expected_ans}"


def extract_scratchpad_answer(text: str) -> Tuple[str, str]:
    """
    Extracts the intermediate reasoning and final arithmetic answer from generation.
    Returns: (reasoning_steps, clean_answer_string)
    """
    clean_text = text.replace("<BOS>", "").replace("<EOS>", "").strip()
    reasoning = ""
    answer = ""

    if "<THINK>" in clean_text and "</THINK>" in clean_text:
        parts = clean_text.split("</THINK>")
        reasoning = parts[0].split("<THINK>")[-1]
        post_think = parts[1]
        if "=" in post_think:
            answer = post_think.split("=")[-1].strip()
        else:
            answer = post_think.strip()
    elif "=" in clean_text:
        parts = clean_text.split("=")
        answer = parts[-1].strip()

    # Clean out trailing delimiter leakages if any
    for bad in ["<", "(", ")", "+", "-", "*", "/", " "]:
        if bad in answer:
            answer = answer.split(bad)[0].strip()

    return reasoning, answer
