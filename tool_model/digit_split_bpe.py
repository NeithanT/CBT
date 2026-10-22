from typing import List, Union, Dict, Any, Optional
import tiktoken
import torch


class DigitSplitBPETokenizer:
    """
    Unified 4k Subword BPE Tokenizer with Single-Digit Splitting.
    Solves the BPE chunking catastrophe by strictly preserving 1:1 digit tokens (0-9)
    while merging natural language subwords.
    """

    def __init__(self, vocab_size: int = 4000):
        base = tiktoken.get_encoding("gpt2")
        
        # Keep top ranks, but prohibit any multi-character merges containing digits
        filtered_ranks = {
            k: v for k, v in base._mergeable_ranks.items()
            if v < vocab_size and not (len(k) > 1 and any(c in b"0123456789" for c in k))
        }
        self.num_base_tokens = len(filtered_ranks)

        self.special_tokens = {
            "<PAD>": vocab_size + 100,
            "<BOS>": vocab_size + 101,
            "<EOS>": vocab_size + 102,
            "<UNK>": vocab_size + 103,
            "<USER>": vocab_size + 104,
            "</USER>": vocab_size + 105,
            "<THINK>": vocab_size + 106,
            "</THINK>": vocab_size + 107,
            "<CALL>": vocab_size + 108,
            "</CALL>": vocab_size + 109,
            "<RESULT>": vocab_size + 110,
            "</RESULT>": vocab_size + 111,
            "<ANSWER>": vocab_size + 112,
            "</ANSWER>": vocab_size + 113,
        }

        self.pad_id = self.special_tokens["<PAD>"]
        self.bos_id = self.special_tokens["<BOS>"]
        self.eos_id = self.special_tokens["<EOS>"]
        self.unk_id = self.special_tokens["<UNK>"]

        self.user_s_id = self.special_tokens["<USER>"]
        self.user_e_id = self.special_tokens["</USER>"]
        self.think_s_id = self.special_tokens["<THINK>"]
        self.think_e_id = self.special_tokens["</THINK>"]
        self.call_s_id = self.special_tokens["<CALL>"]
        self.call_e_id = self.special_tokens["</CALL>"]
        self.result_s_id = self.special_tokens["<RESULT>"]
        self.result_e_id = self.special_tokens["</RESULT>"]
        self.answer_s_id = self.special_tokens["<ANSWER>"]
        self.answer_e_id = self.special_tokens["</ANSWER>"]

        # Pre-tokenize regex: single digit `\d` comes first, preventing digit merges!
        pat_str = r"""\d|""" + base._pat_str

        self.enc = tiktoken.Encoding(
            name="digit_split_bpe_4k",
            pat_str=pat_str,
            mergeable_ranks=filtered_ranks,
            special_tokens=self.special_tokens,
        )

        # Cache single-digit token IDs for 2D coordinate calculation
        self.digit_ids = set()
        for d in range(10):
            d_toks = self.enc.encode(str(d))
            if len(d_toks) == 1:
                self.digit_ids.add(d_toks[0])

        self.plus_id = self.enc.encode("+")[0]
        self.eq_id = self.enc.encode("=")[0]
        self.lparen_id = self.enc.encode("(")[0]

    @property
    def vocab_size(self) -> int:
        return max(self.special_tokens.values()) + 1

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = self.enc.encode(text, allowed_special=set(self.special_tokens.keys()))
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special: bool = False) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special:
            filtered = [
                tid for tid in token_ids
                if tid not in (self.pad_id, self.bos_id, self.eos_id)
            ]
            return self.enc.decode(filtered)
        return self.enc.decode(token_ids)

    def __len__(self) -> int:
        return self.vocab_size
