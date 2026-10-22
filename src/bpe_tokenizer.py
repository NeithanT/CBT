from typing import List, Union, Optional, Any
import tiktoken


class BPEMathTokenizer:
    """Subword BPE tokenizer wrapping tiktoken GPT-2 with arithmetic and reasoning special tokens."""

    def __init__(self):
        base = tiktoken.get_encoding("gpt2")
        self.base_n_vocab = base.n_vocab

        self.special_tokens = {
            "<PAD>": self.base_n_vocab,
            "<BOS>": self.base_n_vocab + 1,
            "<EOS>": self.base_n_vocab + 2,
            "<THINK>": self.base_n_vocab + 3,
            "</THINK>": self.base_n_vocab + 4,
        }

        self.pad_id = self.special_tokens["<PAD>"]
        self.bos_id = self.special_tokens["<BOS>"]
        self.eos_id = self.special_tokens["<EOS>"]
        self.think_id = self.special_tokens["<THINK>"]
        self.end_think_id = self.special_tokens["</THINK>"]

        self.enc = tiktoken.Encoding(
            name="gpt2_math_cot",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        return self.base_n_vocab + len(self.special_tokens)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = self.enc.encode(text, allowed_special=set(self.special_tokens.keys()))
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids: Union[List[int], Any], skip_special: bool = False) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        if skip_special:
            filtered = [tid for tid in token_ids if tid not in (self.pad_id, self.bos_id, self.eos_id)]
            return self.enc.decode(filtered)
        return self.enc.decode(token_ids)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"BPEMathTokenizer(vocab_size={self.vocab_size}, special_tokens={list(self.special_tokens.keys())})"
