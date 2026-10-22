import string
from typing import List, Union, Dict, Any, Optional
import torch


class ToolTokenizer:
    """Character-level tokenizer with structured discourse tokens for tool calling."""

    def __init__(self):
        # Base special tokens
        self.special_tokens = [
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

        # Printable ASCII characters (letters, digits, punctuation, whitespace)
        ascii_chars = sorted(list(set(string.ascii_letters + string.digits + string.punctuation + " \n\t")))
        cur_id = len(self.special_tokens)
        for ch in ascii_chars:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = cur_id
                self.id_to_token[cur_id] = ch
                cur_id += 1

        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

        self.user_s_id = self.token_to_id["<USER>"]
        self.user_e_id = self.token_to_id["</USER>"]
        self.think_s_id = self.token_to_id["<THINK>"]
        self.think_e_id = self.token_to_id["</THINK>"]
        self.call_s_id = self.token_to_id["<CALL>"]
        self.call_e_id = self.token_to_id["</CALL>"]
        self.result_s_id = self.token_to_id["<RESULT>"]
        self.result_e_id = self.token_to_id["</RESULT>"]
        self.answer_s_id = self.token_to_id["<ANSWER>"]
        self.answer_e_id = self.token_to_id["</ANSWER>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)

        i = 0
        n = len(text)
        # Match special tokens first
        while i < n:
            matched = False
            for sp in self.special_tokens:
                if text.startswith(sp, i):
                    tokens.append(self.token_to_id[sp])
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                tokens.append(self.token_to_id.get(ch, self.unk_id))
                i += 1

        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special: bool = False) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        res = []
        for tid in token_ids:
            if skip_special and tid in (self.pad_id, self.bos_id, self.eos_id):
                continue
            tok = self.id_to_token.get(tid, "")
            res.append(tok)
        return "".join(res)

    def __len__(self) -> int:
        return self.vocab_size
