"""
Unit tests for MicroMath-2M architecture.
Verifies parameter count strictly < 2M, weight tying, NoPE + Abacus place indexing,
forward/backward pass, and autoregressive generation.
"""

import unittest
import torch

from src.tokenizer import MathTokenizer
from src.dataset import compute_abacus_place_ids
from src.arithmetic import format_addition_char_scratchpad
from src.models.micromath import MicroMathConfig, MicroMath2M
from src.models.factory import create_model


class TestMicroMath2M(unittest.TestCase):
    """Test suite for MicroMath-2M architecture."""

    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = MathTokenizer()

    def test_parameter_count_under_2m(self):
        """Verifies MicroMath-2M parameters are strictly < 2M non-embedding."""
        config = MicroMathConfig(
            vocab_size=32,
            n_embd=192,
            n_layer=3,
            n_head=4,
            d_ffn=768,
        )
        model = MicroMath2M(config)
        non_emb = model.get_num_params(non_embedding=True)
        total_params = model.get_num_params(non_embedding=False)

        self.assertLess(non_emb, 2_000_000, f"Non-embedding params {non_emb} exceeds 2M ceiling")
        self.assertLess(total_params, 2_000_000, f"Total params {total_params} exceeds 2M ceiling")
        self.assertEqual(non_emb, 1_359_360)

    def test_weight_tying(self):
        """Verifies tied weights between token embedding and LM head."""
        config = MicroMathConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = MicroMath2M(config)
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_factory_instantiation(self):
        """Verifies model factory instantiates MicroMath2M."""
        config = MicroMathConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = create_model(config)
        self.assertIsInstance(model, MicroMath2M)

    def test_tensor_place_ids_alignment(self):
        """Verifies compute_tensor_place_ids matches dataset place IDs."""
        config = MicroMathConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = MicroMath2M(config)

        for a, b in [(15, 25), (1234, 5678), (99, 1)]:
            eq = format_addition_char_scratchpad(a, b)
            tokens = self.tokenizer.encode(eq, add_bos=False, add_eos=False)
            expected_places = compute_abacus_place_ids(tokens, self.tokenizer)

            idx = torch.tensor([tokens], dtype=torch.long)
            computed_places = model.compute_tensor_place_ids(
                idx,
                plus_token_id=self.tokenizer.token_to_id["+"],
                eq_token_id=self.tokenizer.token_to_id["="],
                lparen_token_id=self.tokenizer.token_to_id["("],
                think_start_id=self.tokenizer.token_to_id["<THINK>"],
                think_end_id=self.tokenizer.token_to_id["</THINK>"],
                digit_ids=[self.tokenizer.token_to_id[str(d)] for d in range(10)],
            )
            self.assertEqual(computed_places[0].tolist(), expected_places)

    def test_forward_and_backward(self):
        """Verifies forward pass, cross-entropy loss, and gradient backpropagation."""
        config = MicroMathConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = MicroMath2M(config)
        model.train()

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T))
        place_ids = torch.randint(0, 10, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        logits, loss = model(idx, place_ids=place_ids, targets=targets)
        self.assertEqual(logits.shape, (B, T, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_generation(self):
        """Verifies autoregressive token generation with dynamic place tracking."""
        config = MicroMathConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = MicroMath2M(config)
        model.eval()

        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=4, tokenizer=self.tokenizer)
        self.assertEqual(out.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
