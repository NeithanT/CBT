from typing import List, Union, Dict, Any, Optional
import tiktoken
import torch


class ToolBPETokenizer:
    """4k Vocabulary Subword BPE Tokenizer for ToolGPT."""

    def __init__(self, vocab_size: int = 4000):
        base = tiktoken.get_encoding("gpt2")
        self.target_vocab_size = vocab_size

        # Top mergeable ranks up to target vocab size
        top_ranks = {k: v for k, v in base._mergeable_ranks.items() if v < vocab_size}
        self.num_base_tokens = len(top_ranks)

        self.special_tokens = {
            "<PAD>": vocab_size,
            "<BOS>": vocab_size + 1,
            "<EOS>": vocab_size + 2,
            "<UNK>": vocab_size + 3,
            "<USER>": vocab_size + 4,
            "</USER>": vocab_size + 5,
            "<THINK>": vocab_size + 6,
            "</THINK>": vocab_size + 7,
            "<CALL>": vocab_size + 8,
            "</CALL>": vocab_size + 9,
            "<RESULT>": vocab_size + 10,
            "</RESULT>": vocab_size + 11,
            "<ANSWER>": vocab_size + 12,
            "</ANSWER>": vocab_size + 13,
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

        self.enc = tiktoken.Encoding(
            name="tool_bpe_4k",
            pat_str=base._pat_str,
            mergeable_ranks=top_ranks,
            special_tokens=self.special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        return self.num_base_tokens + len(self.special_tokens)

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
