"""
Character-level Math Tokenizer for HRM-RoPE.
Restricted to arithmetic digits, operators, scratchpad tags, and delimiter tokens.
"""

import re
from typing import List, Dict, Optional


class MathTokenizer:
    """Character-level tokenizer for multi-digit arithmetic and scratchpad reasoning."""

    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        digits: Optional[List[str]] = None,
        operators: Optional[List[str]] = None,
        separators: Optional[List[str]] = None,
    ):
        self.special_tokens = special_tokens or [
            "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<THINK>", "</THINK>"
        ]
        self.digits = digits or [str(d) for d in range(10)]
        self.operators = operators or ["+", "-", "*", "/", "^", "%"]
        self.separators = separators or ["(", ")", "=", " ", "\n"]

        self.vocab: List[str] = (
            self.special_tokens + self.digits + self.operators + self.separators
        )
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for idx, tok in enumerate(self.vocab)}

        self.pad_id: int = self.token_to_id["<PAD>"]
        self.bos_id: int = self.token_to_id["<BOS>"]
        self.eos_id: int = self.token_to_id["<EOS>"]
        self.unk_id: int = self.token_to_id["<UNK>"]

        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(tok) for tok in sorted_specials) + "|."
        self._pattern = re.compile(pattern)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)

        for match in self._pattern.finditer(text):
            tok_str = match.group()
            token_id = self.token_to_id.get(tok_str, self.unk_id)
            tokens.append(token_id)

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        out = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "<UNK>")
            if skip_special and tok in self.special_tokens:
                continue
            out.append(tok)
        return "".join(out)
