import os
import re
import random
import urllib.request
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .tokenizer import MathTokenizer
    from .bpe_tokenizer import BPEMathTokenizer
    from .arithmetic import format_addition_bpe_word_cot, format_addition_char_scratchpad
except ImportError:
    from tokenizer import MathTokenizer
    from bpe_tokenizer import BPEMathTokenizer
    from arithmetic import format_addition_bpe_word_cot, format_addition_char_scratchpad

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DATA_DIR = os.path.join(ROOT_DIR, "data")
PHILOSOPHY_DATA_DIR = os.path.join(SHARED_DATA_DIR, "philosophy")


class MathEquationGenerator:
    """Generates synthetic arithmetic equations formatted as text."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_addition(self, min_val: int = 0, max_val: int = 999) -> str:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        return f"{a}+{b}={a + b}"

    def generate_subtraction(self, min_val: int = 0, max_val: int = 999) -> str:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        return f"{a}-{b}={a - b}"

    def generate_multiplication(self, min_val: int = 0, max_val: int = 99) -> str:
        a = random.randint(min_val, max_val)
        b = random.randint(0, 50)
        return f"{a}*{b}={a * b}"

    def generate_division(self, min_val: int = 1, max_val: int = 50) -> str:
        b = random.randint(1, 25)
        quotient = random.randint(0, max_val)
        a = b * quotient
        return f"{a}/{b}={quotient}"

    def generate_exponent(self) -> str:
        base = random.randint(1, 12)
        exp = random.randint(0, 4)
        return f"{base}^{exp}={base ** exp}"

    def generate_parentheses(self) -> str:
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        c = random.randint(1, 10)
        op = random.choice(["+", "-"])
        if op == "+":
            return f"({a}+{b})*{c}={(a + b) * c}"
        return f"({a}-{b})*{c}={(a - b) * c}"

    def generate_sample(self) -> str:
        weights = [0.35, 0.25, 0.20, 0.10, 0.05, 0.05]
        choice = random.choices(
            [
                self.generate_addition,
                self.generate_subtraction,
                self.generate_multiplication,
                self.generate_division,
                self.generate_exponent,
                self.generate_parentheses,
            ],
            weights=weights,
            k=1,
        )[0]
        return choice()

    def generate_dataset(self, num_samples: int) -> List[str]:
        equations = set()
        max_attempts = num_samples * 3
        attempts = 0
        while len(equations) < num_samples and attempts < max_attempts:
            equations.add(self.generate_sample())
            attempts += 1
        return list(equations)


def generate_commutativity_dataset(
    min_val: int = 0,
    max_val: int = 50,
    train_ratio: float = 0.8,
    reversed_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    rng = random.Random(seed)
    pairs = []
    for a in range(min_val, max_val + 1):
        for b in range(a, max_val + 1):
            pairs.append((a, b))

    rng.shuffle(pairs)
    n_train = int(len(pairs) * train_ratio)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    train_canonical = [f"{a}+{b}={a + b}" for a, b in train_pairs]
    test_canonical_unseen = [f"{a}+{b}={a + b}" for a, b in test_pairs]

    all_commuted_pairs = [(b, a) for a, b in pairs if a != b]
    rng.shuffle(all_commuted_pairs)

    if reversed_ratio > 0.0:
        n_rev_train = max(1, int(len(all_commuted_pairs) * reversed_ratio))
        rev_train_pairs = all_commuted_pairs[:n_rev_train]
        rev_test_pairs = all_commuted_pairs[n_rev_train:]

        train_equations = train_canonical + [f"{b}+{a}={a + b}" for b, a in rev_train_pairs]
        test_commuted_seen = [f"{b}+{a}={a + b}" for b, a in rev_train_pairs]
        test_commuted_unseen = [f"{b}+{a}={a + b}" for b, a in rev_test_pairs]
    else:
        train_equations = train_canonical
        test_commuted_seen = [f"{b}+{a}={a + b}" for a, b in train_pairs if a != b]
        test_commuted_unseen = [f"{b}+{a}={a + b}" for a, b in test_pairs if a != b]

    return train_equations, test_canonical_unseen, test_commuted_seen, test_commuted_unseen


def compute_abacus_place_ids(
    token_ids: List[int],
    tokenizer: Any,
    max_places: int = 128,
) -> List[int]:
    """
    Computes decimal place value indices for tokens.
    For standard operands, indices count right-to-left relative to '+' and '='.
    Inside <THINK> scratchpads, column blocks (c+da+db)=c's receive column place indices.
    For final answers, indices count down from the highest column to LSD (0).
    """
    id_to_token = getattr(tokenizer, "id_to_token", {})
    chars = [id_to_token.get(t, "") for t in token_ids]
    n = len(token_ids)
    place_ids = [0] * n

    plus_idx = -1
    eq_idx = -1
    think_start = -1
    think_end = -1

    for i, c in enumerate(chars):
        if c == "+" and plus_idx == -1:
            plus_idx = i
        elif c == "=" and plus_idx != -1 and eq_idx == -1:
            eq_idx = i
        elif c == "<THINK>":
            think_start = i
        elif c == "</THINK>":
            think_end = i

    # Operand 1 (before '+')
    if plus_idx != -1:
        for i in range(plus_idx):
            if chars[i].isdigit():
                place_ids[i] = min(plus_idx - 1 - i, max_places - 1)

    # Operand 2 (between '+' and '=')
    if eq_idx != -1 and plus_idx != -1:
        for i in range(plus_idx + 1, eq_idx):
            if chars[i].isdigit():
                place_ids[i] = min(eq_idx - 1 - i, max_places - 1)

    # Scratchpad inside <THINK>
    col_count = 0
    if think_start != -1:
        end_search = think_end if think_end != -1 else n
        cur_col = -1
        for i in range(think_start + 1, end_search):
            if chars[i] == "(":
                cur_col += 1
            if cur_col >= 0:
                place_ids[i] = min(cur_col, max_places - 1)
        col_count = cur_col + 1

    # After <THINK> (or plain equation after '=')
    if think_end != -1:
        ans_start = -1
        for i in range(think_end + 1, n):
            if chars[i] == "=":
                ans_start = i + 1
                break
        if ans_start != -1:
            ans_digit_idx = 0
            for i in range(ans_start, n):
                if chars[i].isdigit():
                    place_ids[i] = min(max(0, col_count - 1 - ans_digit_idx), max_places - 1)
                    ans_digit_idx += 1
    elif eq_idx != -1 and think_start == -1:
        # Plain equation without scratchpad: e.g. 1234+56=1290
        ans_digits = [i for i in range(eq_idx + 1, n) if chars[i].isdigit()]
        num_ans = len(ans_digits)
        for idx, pos in enumerate(ans_digits):
            place_ids[pos] = min(num_ans - 1 - idx, max_places - 1)

    return place_ids


class MathDataset(Dataset):
    """PyTorch Dataset for causal language modeling on character math equations."""

    def __init__(
        self,
        equations: List[str],
        tokenizer: MathTokenizer,
        block_size: int = 32,
        mask_prompt: bool = False,
    ):
        self.equations = equations
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mask_prompt = mask_prompt
        self.encoded_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prepare_data()

    def _prepare_data(self):
        eq_token_id = self.tokenizer.token_to_id["="]
        pad_token_id = self.tokenizer.pad_id

        for eq in self.equations:
            token_ids = self.tokenizer.encode(eq, add_bos=True, add_eos=True)
            if len(token_ids) > self.block_size + 1:
                token_ids = token_ids[: self.block_size + 1]

            x = token_ids[:-1]
            y = token_ids[1:]

            pad_len = self.block_size - len(x)
            if pad_len > 0:
                x_padded = x + [pad_token_id] * pad_len
                y_padded = y + [-100] * pad_len
            else:
                x_padded = x[: self.block_size]
                y_padded = y[: self.block_size]

            if self.mask_prompt and eq_token_id in x_padded:
                eq_idx = x_padded.index(eq_token_id)
                for i in range(eq_idx):
                    y_padded[i] = -100

            x_tensor = torch.tensor(x_padded, dtype=torch.long)
            y_tensor = torch.tensor(y_padded, dtype=torch.long)
            self.encoded_samples.append((x_tensor, y_tensor))

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_samples[idx]


def create_dataloaders(
    num_train: int = 10000,
    num_val: int = 1000,
    batch_size: int = 64,
    block_size: int = 32,
    mask_prompt: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, MathTokenizer]:
    tokenizer = MathTokenizer()
    generator = MathEquationGenerator(seed=seed)

    all_samples = generator.generate_dataset(num_train + num_val)
    train_samples = all_samples[:num_train]
    val_samples = all_samples[num_train:]

    train_ds = MathDataset(train_samples, tokenizer, block_size=block_size, mask_prompt=mask_prompt)
    val_ds = MathDataset(val_samples, tokenizer, block_size=block_size, mask_prompt=mask_prompt)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer


def get_or_generate_shared_math_pairs(
    num_samples: int = 2500000,
    seed: int = 42,
    filepath: Optional[str] = None,
    force_regenerate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns deterministic integer pairs (1-12 digits) for controlled comparative study."""
    if filepath is None:
        candidates = [
            os.path.join(SHARED_DATA_DIR, "shared_math_pairs.pt"),
            os.path.join(SHARED_DATA_DIR, "shared_math_pairs_12digits.pt"),
        ]
        filepath = candidates[0]
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath) and not force_regenerate:
        try:
            try:
                data = torch.load(filepath, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(filepath, map_location="cpu")
            a_tensor = data["a"]
            b_tensor = data["b"]
            if len(a_tensor) >= num_samples:
                return a_tensor[:num_samples], b_tensor[:num_samples]
        except Exception:
            pass

    rng = random.Random(seed)
    a_list, b_list = [], []
    for _ in range(num_samples):
        len_a = rng.randint(1, 12)
        len_b = rng.randint(1, 12)
        min_a = 0 if len_a == 1 else 10 ** (len_a - 1)
        max_a = 10 ** len_a - 1
        min_b = 0 if len_b == 1 else 10 ** (len_b - 1)
        max_b = 10 ** len_b - 1
        a_list.append(rng.randint(min_a, max_a))
        b_list.append(rng.randint(min_b, max_b))

    a_tensor = torch.tensor(a_list, dtype=torch.int64)
    b_tensor = torch.tensor(b_list, dtype=torch.int64)
    torch.save(
        {"a": a_tensor, "b": b_tensor, "seed": seed, "num_samples": num_samples, "max_digits": 12},
        filepath,
    )
    return a_tensor, b_tensor


def clean_gutenberg_text(raw_text: str) -> str:
    start_match = re.search(r"\*\*\* START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw_text, re.IGNORECASE)
    text = raw_text[start_match.end() :] if start_match else raw_text
    end_match = re.search(r"\*\*\* END OF TH(E|IS) PROJECT GUTENBERG EBOOK", text, re.IGNORECASE)
    if end_match:
        text = text[: end_match.start()]
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_and_prepare_philosophy_corpus(data_dir: str = PHILOSOPHY_DATA_DIR) -> str:
    os.makedirs(data_dir, exist_ok=True)
    merged_path = os.path.join(data_dir, "philosophy_corpus.txt")
    if os.path.exists(merged_path) and os.path.getsize(merged_path) > 100000:
        return merged_path

    sources = {
        "kant_critique_of_pure_reason": "https://www.gutenberg.org/cache/epub/4280/pg4280.txt",
        "descartes_discourse_on_method": "https://www.gutenberg.org/cache/epub/59/pg59.txt",
        "russell_problems_of_philosophy": "https://www.gutenberg.org/cache/epub/2529/pg2529.txt",
        "spinoza_ethics": "https://www.gutenberg.org/cache/epub/3800/pg3800.txt",
    }
    collected = []
    for name, url in sources.items():
        dst = os.path.join(data_dir, f"{name}.txt")
        if not os.path.exists(dst):
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception:
                continue
        if os.path.exists(dst):
            with open(dst, "r", encoding="utf-8", errors="ignore") as f:
                cleaned = clean_gutenberg_text(f.read())
                collected.append(cleaned)

    merged_text = "\n\n" + ("=" * 40) + "\n\n".join(collected)
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(merged_text)
    return merged_path


def load_philosophy_token_stream(tokenizer: BPEMathTokenizer, corpus_path: Optional[str] = None) -> List[int]:
    if corpus_path is None or not os.path.exists(corpus_path):
        corpus_path = download_and_prepare_philosophy_corpus()
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return tokenizer.encode(text)


class ComparativeDatasetBuilder:
    """Pre-encodes batches for Model 1 (Char), Model 2 (BPE Word-CoT), and Model 3 (Philo-Math)."""

    def __init__(self, block_size: int = 192):
        self.block_size = block_size
        self.char_tokenizer = MathTokenizer()
        self.bpe_tokenizer = BPEMathTokenizer()
        self._philosophy_tokens: Optional[List[int]] = None

    def get_philosophy_tokens(self) -> List[int]:
        if self._philosophy_tokens is None:
            self._philosophy_tokens = load_philosophy_token_stream(self.bpe_tokenizer)
        return self._philosophy_tokens

    def build_model_a_batch(self, pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._encode_samples(
            [format_addition_char_scratchpad(a, b) for a, b in pairs],
            self.char_tokenizer,
        )

    def build_model_b_batch(self, pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._encode_samples(
            [format_addition_bpe_word_cot(a, b) for a, b in pairs],
            self.bpe_tokenizer,
        )

    def build_model_c_batch(
        self, pairs: List[Tuple[int, int]], philo_rng: random.Random
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        half = len(pairs) // 2
        math_pairs = pairs[:half]
        math_texts = [format_addition_bpe_word_cot(a, b) for a, b in math_pairs]
        x_m, y_m = self._encode_samples(math_texts, self.bpe_tokenizer)

        philo_tokens = self.get_philosophy_tokens()
        max_idx = len(philo_tokens) - (self.block_size + 2)
        n_philo = len(pairs) - half
        x_p_list, y_p_list = [], []

        for _ in range(n_philo):
            idx = philo_rng.randint(0, max(0, max_idx))
            chunk = philo_tokens[idx : idx + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.bpe_tokenizer.pad_id] * (self.block_size + 1 - len(chunk))
            x_p_list.append(torch.tensor(chunk[: self.block_size], dtype=torch.long))
            y_p_list.append(torch.tensor(chunk[1 : self.block_size + 1], dtype=torch.long))

        x_p = torch.stack(x_p_list)
        y_p = torch.stack(y_p_list)
        return torch.cat([x_m, x_p], dim=0), torch.cat([y_m, y_p], dim=0)

    def _encode_samples(
        self, texts: List[str], tokenizer: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_list, y_list = [], []
        pad_id = tokenizer.pad_id
        for txt in texts:
            tokens = tokenizer.encode(txt, add_bos=True, add_eos=True)
            if len(tokens) > self.block_size + 1:
                tokens = tokens[: self.block_size + 1]
            x = tokens[:-1]
            y = tokens[1:]
            pad_len = self.block_size - len(x)
            if pad_len > 0:
                x = x + [pad_id] * pad_len
                y = y + [-100] * pad_len
            x_list.append(torch.tensor(x[: self.block_size], dtype=torch.long))
            y_list.append(torch.tensor(y[: self.block_size], dtype=torch.long))
        return torch.stack(x_list), torch.stack(y_list)
