"""
Unit tests for Hybrid SSM-Transformer (HybridMathGPT).
Verifies parameter ceiling (<20M non-embedding parameters), weight tying,
forward and backward passes, generation, and layer patterns.
"""

import unittest
import torch

from src.models.hybrid import HybridConfig, HybridMathGPT
from src.models.factory import create_model


class TestHybridMathGPT(unittest.TestCase):
    """Test suite for HybridMathGPT."""

    def setUp(self):
        torch.manual_seed(42)

    def test_parameter_ceiling(self):
        """Verifies non-embedding parameters are strictly < 20M and align with baseline."""
        config = HybridConfig(
            vocab_size=32,
            block_size=512,
            n_layer=6,
            n_embd=512,
            n_head=8,
            d_ffn=2048,
            pattern="interleaved",
        )
        model = HybridMathGPT(config)
        n_params = model.get_num_params(non_embedding=True)

        self.assertLess(n_params, 20_000_000, f"Params {n_params} exceeds 20M ceiling")
        # 3 Attn (3 * 3,146,752) + 3 GLRU (3 * 3,149,312) + 512 = 18,888,704
        self.assertEqual(n_params, 18_888_704)

    def test_weight_tying(self):
        """Verifies tied weights between token embedding and output projection."""
        config = HybridConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = HybridMathGPT(config)
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_forward_pass_and_loss(self):
        """Verifies forward pass with logits shape and cross-entropy loss computation."""
        config = HybridConfig(vocab_size=32, block_size=64, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = HybridMathGPT(config)

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        logits, loss = model(idx, targets=targets)
        self.assertEqual(logits.shape, (B, T, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        # Backward pass
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_factory_instantiation(self):
        """Verifies model factory instantiates HybridMathGPT."""
        config = HybridConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = create_model(config)
        self.assertIsInstance(model, HybridMathGPT)

    def test_layer_patterns(self):
        """Verifies support for pure_ssm, pure_attn, and interleaved patterns."""
        for pattern, expected_glru_count in [("interleaved", 2), ("pure_ssm", 4), ("pure_attn", 0)]:
            config = HybridConfig(
                vocab_size=32, n_embd=64, n_layer=4, n_head=2, d_ffn=128, pattern=pattern
            )
            model = HybridMathGPT(config)
            glru_count = sum(1 for b in model.transformer.h if b.mixer_type == "glru")
            self.assertEqual(glru_count, expected_glru_count)

    def test_generation(self):
        """Verifies autoregressive token generation."""
        config = HybridConfig(vocab_size=32, block_size=64, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = HybridMathGPT(config)
        model.eval()

        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=5)
        self.assertEqual(out.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
