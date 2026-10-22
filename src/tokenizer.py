import re
from typing import List, Dict, Union, Optional, Any

try:
    from .constants import TOKENIZER_CONSTANTS, TokenizerConstants
except ImportError:
    from constants import TOKENIZER_CONSTANTS, TokenizerConstants


class MathTokenizer:
    """Character-level tokenizer restricted to arithmetic symbols and special tokens."""

    def __init__(
        self,
        constants: Optional[TokenizerConstants] = None,
        special_tokens: Optional[List[str]] = None,
        digits: Optional[List[str]] = None,
        operators: Optional[List[str]] = None,
        separators: Optional[List[str]] = None,
    ):
        cfg = constants or TOKENIZER_CONSTANTS
        self.special_tokens = special_tokens or list(cfg.special_tokens)
        self.digits = digits or list(cfg.digits)
        self.operators = operators or list(cfg.operators)
        self.separators = separators or list(cfg.separators)

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

    def decode(
        self,
        token_ids: Union[List[int], Any],
        skip_special: bool = False,
    ) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        chars: List[str] = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            chars.append(self.id_to_token.get(tid, "<UNK>"))
        return "".join(chars)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"MathTokenizer(vocab_size={self.vocab_size}, special_tokens={self.special_tokens})"
