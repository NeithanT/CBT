"""
Unit tests for Looped Transformer architecture (src.models.looped).
Validates configuration, weight tying, parameter ceilings, deep supervision,
RoPE dynamic cache extension, and autoregressive generation.
"""

import math
import unittest
import torch
import torch.nn.functional as F

from src.models.looped import (
    LoopedGPTConfig,
    LoopedCausalAttention,
    LoopedMLP,
    CoreRecurrentBlock,
    LoopedMathGPT,
)
from src.models.factory import create_model


class TestLoopedGPTConfig(unittest.TestCase):
    """Verifies default hyperparameters and compatibility of LoopedGPTConfig."""

    def test_default_config_values(self):
        cfg = LoopedGPTConfig()
        self.assertEqual(cfg.vocab_size, 32)
        self.assertEqual(cfg.block_size, 512)
        self.assertEqual(cfg.n_embd, 512)
        self.assertEqual(cfg.n_head, 8)
        self.assertEqual(cfg.n_core_layers, 2)
        self.assertEqual(cfg.train_loops, 6)
        self.assertEqual(cfg.eval_loops, 12)
        self.assertEqual(cfg.dropout, 0.0)
        self.assertFalse(cfg.bias)
        self.assertEqual(cfg.rope_theta, 10000.0)
        self.assertTrue(cfg.deep_supervision)
        self.assertEqual(cfg.model_type, "looped")
        self.assertEqual(cfg.pos_emb_type, "rope")
        self.assertFalse(cfg.use_pos_emb)


class TestLoopedArchitecture(unittest.TestCase):
    """Tests model components, parameter budget, forward passes, and training dynamics."""

    def setUp(self):
        # Compact config for fast unit testing
        self.config = LoopedGPTConfig(
            vocab_size=32,
            block_size=64,
            n_embd=64,
            n_head=4,
            n_core_layers=2,
            train_loops=3,
            eval_loops=5,
            dropout=0.0,
            bias=False,
            rope_theta=10000.0,
            deep_supervision=True,
        )
        self.model = LoopedMathGPT(self.config)

    def test_weight_tying(self):
        self.assertIs(self.model.transformer.wte.weight, self.model.lm_head.weight)

    def test_parameter_ceiling(self):
        full_cfg = LoopedGPTConfig()
        full_model = LoopedMathGPT(full_cfg)
        non_emb_params = full_model.get_num_params(non_embedding=True)
        self.assertLess(non_emb_params, 20_000_000)
        self.assertGreater(non_emb_params, 1_000_000)

    def test_core_recurrent_block_residual_scaling(self):
        block = CoreRecurrentBlock(self.config)
        x = torch.randn(2, 8, self.config.n_embd)
        out1 = block(x, scale_factor=1.0)
        out2 = block(x, scale_factor=0.5)
        self.assertFalse(torch.allclose(out1, out2))

    def test_forward_pass_training_deep_supervision(self):
        self.model.train()
        idx = torch.randint(0, self.config.vocab_size, (2, 8))
        targets = torch.randint(0, self.config.vocab_size, (2, 8))
        logits, loss = self.model(idx, targets)

        self.assertEqual(logits.shape, (2, 8, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        # Verify gradient flow back through recurrent unrolls
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_forward_pass_eval_mode(self):
        self.model.eval()
        idx = torch.randint(0, self.config.vocab_size, (2, 8))
        targets = torch.randint(0, self.config.vocab_size, (2, 8))
        with torch.no_grad():
            logits, loss = self.model(idx, targets)

        self.assertEqual(logits.shape, (2, 8, self.config.vocab_size))
        self.assertIsNotNone(loss)

    def test_custom_num_loops(self):
        self.model.eval()
        idx = torch.randint(0, self.config.vocab_size, (1, 6))
        with torch.no_grad():
            logits_3, _ = self.model(idx, num_loops=3)
            logits_7, _ = self.model(idx, num_loops=7)
        self.assertEqual(logits_3.shape, (1, 6, self.config.vocab_size))
        self.assertEqual(logits_7.shape, (1, 6, self.config.vocab_size))
        # Representations evolve across additional recurrent unrolls
        self.assertFalse(torch.allclose(logits_3, logits_7, atol=1e-4))

    def test_generate(self):
        self.model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        generated = self.model.generate(prompt, max_new_tokens=5, temperature=0.0)
        self.assertEqual(generated.shape, (1, 8))
        self.assertTrue(torch.equal(generated[:, :3], prompt))

    def test_dynamic_rope_cache_extension(self):
        self.model.eval()
        # Input length exceeding precomputed block_size * 2 (128)
        seq_len = 150
        idx = torch.randint(0, self.config.vocab_size, (1, seq_len))
        with torch.no_grad():
            logits, _ = self.model(idx)
        self.assertEqual(logits.shape, (1, seq_len, self.config.vocab_size))

    def test_factory_instantiation(self):
        model = create_model(self.config)
        self.assertIsInstance(model, LoopedMathGPT)


if __name__ == "__main__":
    unittest.main()
