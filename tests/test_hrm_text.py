"""
Unit tests for compressed HRM-Text architecture.
Verifies parameter ceiling (< 2.0M), weight tying, forward/backward, and generation.
"""

import unittest
import torch
import torch.nn.functional as F

from src.models.hrm_text import HRMTextConfig, HRMTextModel
from src.models.factory import create_model
from src.tokenizer import MathTokenizer


class TestHRMText(unittest.TestCase):
    """Test suite for compressed HRM-Text architecture."""

    def setUp(self):
        self.tokenizer = MathTokenizer()

    def test_parameter_count_under_2m(self):
        """Verifies that non-embedding parameters are strictly below 2,000,000."""
        config = HRMTextConfig(
            vocab_size=32,
            hidden_size=256,
            num_heads=4,
            expansion=3.0,
            n_layers=2,
            half_layers=True,
        )
        model = HRMTextModel(config)
        non_emb_params = model.get_num_params(non_embedding=True)
        total_params = model.get_num_params(non_embedding=False)

        self.assertLess(
            non_emb_params,
            2_000_000,
            f"Non-embedding params {non_emb_params:,} must be strictly < 2,000,000",
        )
        self.assertEqual(non_emb_params, 1_443_840)
        self.assertEqual(total_params, 1_452_032)

    def test_weight_tying(self):
        """Verifies weight tying between embedding and output projection."""
        config = HRMTextConfig(vocab_size=32, hidden_size=64, num_heads=2)
        model = HRMTextModel(config)
        self.assertIs(model.wte.weight, model.lm_head.weight)

    def test_model_factory_instantiation(self):
        """Verifies model factory instantiates HRMTextModel from config."""
        config = HRMTextConfig(vocab_size=32, hidden_size=64, num_heads=2)
        model = create_model(config)
        self.assertIsInstance(model, HRMTextModel)

    def test_forward_and_backward(self):
        """Verifies forward pass and gradient backpropagation through H and L loops."""
        config = HRMTextConfig(
            vocab_size=32,
            block_size=32,
            hidden_size=64,
            num_heads=2,
            expansion=2.0,
            h_cycles=2,
            l_cycles=2,
            n_layers=2,
        )
        model = HRMTextModel(config)
        model.train()

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        # Forward without targets
        logits, loss_none = model(idx)
        self.assertEqual(logits.shape, (B, T, config.vocab_size))
        self.assertIsNone(loss_none)

        # Forward with targets
        logits, loss = model(idx, targets=targets)
        self.assertEqual(logits.shape, (B, T, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        # Backward pass
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Missing gradient for {name}")

    def test_generation(self):
        """Verifies autoregressive token generation."""
        config = HRMTextConfig(
            vocab_size=32,
            block_size=32,
            hidden_size=64,
            num_heads=2,
            h_cycles=1,
            l_cycles=1,
        )
        model = HRMTextModel(config)
        model.eval()

        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=4)
        self.assertEqual(out.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
