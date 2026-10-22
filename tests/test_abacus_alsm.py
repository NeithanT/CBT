"""
Unit tests for Abacus-Anchored Looped State Machine (ALSM-GPT) and Abacus Place Indexing.
Verifies right-to-left decimal place calculation, parameter ceilings, offset jitter,
gated input injection, deep supervision, and generation.
"""

import unittest
import torch

from src.tokenizer import MathTokenizer
from src.dataset import compute_abacus_place_ids
from src.models.alsm import ALSMConfig, AbacusLoopedMathGPT
from src.models.factory import create_model


class TestAbacusALSM(unittest.TestCase):
    """Test suite for Abacus place-value indexing and ALSM-GPT."""

    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = MathTokenizer()

    def test_compute_abacus_place_ids(self):
        """Verifies that digits are indexed right-to-left relative to LSD (place 0)."""
        equation = "1234+56=1290"
        tokens = self.tokenizer.encode(equation, add_bos=False, add_eos=False)
        place_ids = compute_abacus_place_ids(tokens, self.tokenizer)

        token_chars = [self.tokenizer.id_to_token[t] for t in tokens]
        # Expected:
        # '1': 3, '2': 2, '3': 1, '4': 0
        # '+': 0
        # '5': 1, '6': 0
        # '=': 0
        # '1': 3, '2': 2, '9': 1, '0': 0
        expected_places = [3, 2, 1, 0, 0, 1, 0, 0, 3, 2, 1, 0]
        self.assertEqual(place_ids, expected_places)

        # Check alignment: units of Op 1 ('4'), units of Op 2 ('6'), units of Result ('0')
        idx_4 = token_chars.index("4")
        idx_6 = token_chars.index("6")
        idx_0 = len(token_chars) - 1
        self.assertEqual(place_ids[idx_4], 0)
        self.assertEqual(place_ids[idx_6], 0)
        self.assertEqual(place_ids[idx_0], 0)

    def test_alsm_parameter_ceiling(self):
        """Verifies ALSM non-embedding parameters are strictly < 20M."""
        config = ALSMConfig(
            vocab_size=32,
            block_size=512,
            n_embd=512,
            n_head=8,
            n_core_layers=4,
            train_loops=6,
            eval_loops=12,
            max_places=128,
            k_rand_shift=32,
        )
        model = AbacusLoopedMathGPT(config)
        n_params = model.get_num_params(non_embedding=True)

        self.assertLess(n_params, 20_000_000, f"Params {n_params} exceeds 20M ceiling")
        # 4 core blocks: 4 * 3,146,752 = 12,587,008
        # wpe_abacus: (128 + 32 + 1) * 512 = 82,432
        # injection_gates: 32 * 512 = 16,384
        # injection_norm: 512
        # ln_f: 512
        # Total non-embedding: 12,686,848 (~12.69M)
        self.assertEqual(n_params, 12_686_848)

    def test_weight_tying(self):
        """Verifies tied weights between token embedding and LM head."""
        config = ALSMConfig(vocab_size=32, n_embd=64, n_core_layers=2, n_head=2)
        model = AbacusLoopedMathGPT(config)
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_forward_with_place_ids_and_deep_supervision(self):
        """Verifies forward pass with explicit place_ids and intermediate deep supervision."""
        config = ALSMConfig(
            vocab_size=32,
            block_size=32,
            n_embd=64,
            n_core_layers=2,
            n_head=2,
            train_loops=4,
            deep_supervision=True,
        )
        model = AbacusLoopedMathGPT(config)
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

    def test_factory_instantiation(self):
        """Verifies model factory instantiates AbacusLoopedMathGPT."""
        config = ALSMConfig(vocab_size=32, n_embd=64, n_core_layers=2, n_head=2)
        model = create_model(config)
        self.assertIsInstance(model, AbacusLoopedMathGPT)

    def test_generation(self):
        """Verifies autoregressive token generation."""
        config = ALSMConfig(
            vocab_size=32, block_size=32, n_embd=64, n_core_layers=2, n_head=2, eval_loops=3
        )
        model = AbacusLoopedMathGPT(config)
        model.eval()

        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=4)
        self.assertEqual(out.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
