"""
Scaled Unified Tokenizer for ScaledGPT (Dual T4 Optimization).
Supports:
1. Multi-digit column arithmetic with scratchpad tokens (1-50+ digits).
2. Natural language tool-calling dialogues with structured discourse tags.
3. Multi-turn conversation markers and nested JSON syntax.
"""

import string
from typing import List, Dict, Any, Optional
import torch


class ScaledTokenizer:
    """Character-level tokenizer with structured discourse tokens and arithmetic symbols."""

    def __init__(self):
        self.special_tokens: List[str] = [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<UNK>",
            "<USER>",
            "</USER>",
            "<THINK>",
            "</THINK>",
            "<CALL>",
            "</CALL>",
            "<RESULT>",
            "</RESULT>",
            "<ANSWER>",
            "</ANSWER>",
        ]

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        for i, tok in enumerate(self.special_tokens):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

        ascii_chars = sorted(list(set(string.ascii_letters + string.digits + string.punctuation + " \n\t")))
        cur_id = len(self.special_tokens)
        for ch in ascii_chars:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = cur_id
                self.id_to_token[cur_id] = ch
                cur_id += 1

        self.pad_id: int = self.token_to_id["<PAD>"]
        self.bos_id: int = self.token_to_id["<BOS>"]
        self.eos_id: int = self.token_to_id["<EOS>"]
        self.unk_id: int = self.token_to_id["<UNK>"]

        # Discourse markers
        self.user_s_id: int = self.token_to_id["<USER>"]
        self.user_e_id: int = self.token_to_id["</USER>"]
        self.think_s_id: int = self.token_to_id["<THINK>"]
        self.think_e_id: int = self.token_to_id["</THINK>"]
        self.call_s_id: int = self.token_to_id["<CALL>"]
        self.call_e_id: int = self.token_to_id["</CALL>"]
        self.result_s_id: int = self.token_to_id["<RESULT>"]
        self.result_e_id: int = self.token_to_id["</RESULT>"]
        self.answer_s_id: int = self.token_to_id["<ANSWER>"]
        self.answer_e_id: int = self.token_to_id["</ANSWER>"]

        # Arithmetic delimiters
        self.plus_id: int = self.token_to_id["+"]
        self.eq_id: int = self.token_to_id["="]
        self.lparen_id: int = self.token_to_id["("]
        self.digit_ids: List[int] = [self.token_to_id[str(d)] for d in range(10)]

        self.vocab_size: int = len(self.token_to_id)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)

        i = 0
        n = len(text)
        while i < n:
            matched_special = False
            for sp in self.special_tokens:
                if text.startswith(sp, i):
                    tokens.append(self.token_to_id[sp])
                    i += len(sp)
                    matched_special = True
                    break
            if not matched_special:
                ch = text[i]
                tokens.append(self.token_to_id.get(ch, self.unk_id))
                i += 1

        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        chars: List[str] = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok_str = self.id_to_token[tid]
                if skip_special and tok_str in self.special_tokens:
                    continue
                chars.append(tok_str)
            else:
                chars.append("<UNK>")
        return "".join(chars)


# Aliases for Character Based Transformer (CBT)
CBTTokenizer = ScaledTokenizer

