"""
Unit tests for HRM-RoPE (Hierarchical Reasoning Model with Decoupled 2D-RoPE).
Verifies parameter ceiling strictly < 2M, weight tying, 2D-RoPE place-step rotation,
forward/backward hierarchical unrolling, and autoregressive generation.
"""

import unittest
import torch

from src.tokenizer import MathTokenizer
from src.arithmetic import format_addition_char_scratchpad
from src.models.hrm_rope import (
    HRMRoPEConfig,
    HRMRoPE,
    precompute_2d_rope_freqs,
    apply_2d_rope,
)
from src.models.factory import create_model


class TestHRMRoPE(unittest.TestCase):
    """Test suite for HRM-RoPE architecture."""

    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = MathTokenizer()

    def test_parameter_count_under_2m(self):
        """Verifies HRM-RoPE parameters are strictly < 2M non-embedding."""
        config = HRMRoPEConfig(
            vocab_size=32,
            n_embd=256,
            n_head=4,
            d_ffn=768,
            n_fast_steps=2,
            n_slow_cycles=2,
        )
        model = HRMRoPE(config)
        non_emb = model.get_num_params(non_embedding=True)
        total_params = model.get_num_params(non_embedding=False)

        self.assertLess(non_emb, 2_000_000, f"Non-embedding params {non_emb} exceeds 2M ceiling")
        self.assertLess(total_params, 2_000_000, f"Total params {total_params} exceeds 2M ceiling")
        self.assertEqual(non_emb, 1_574_144)
        self.assertEqual(total_params, 1_582_336)

    def test_weight_tying(self):
        """Verifies tied weights between token embedding and LM head."""
        config = HRMRoPEConfig(vocab_size=32, n_embd=64, n_head=2, d_ffn=128)
        model = HRMRoPE(config)
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_factory_instantiation(self):
        """Verifies model factory instantiates HRMRoPE."""
        config = HRMRoPEConfig(vocab_size=32, n_embd=64, n_head=2, d_ffn=128)
        model = create_model(config)
        self.assertIsInstance(model, HRMRoPE)

    def test_2d_rope_invariance_on_identical_place(self):
        """
        Verifies that when place offset Delta p = 0, the place subspace rotation difference is identity.
        """
        head_dim = 64
        cos, sin = precompute_2d_rope_freqs(head_dim, max_coords=128)

        B, H, T = 1, 1, 2
        q = torch.randn(B, H, T, head_dim)
        k = torch.randn(B, H, T, head_dim)

        # Assign same place coordinate (p=5) and same step coordinate (tau=3)
        place_coords = torch.tensor([[5, 5]], dtype=torch.long)
        step_coords = torch.tensor([[3, 3]], dtype=torch.long)

        q_rot = apply_2d_rope(q, place_coords, step_coords, cos, sin)
        k_rot = apply_2d_rope(k, place_coords, step_coords, cos, sin)

        # Inner product between token 0 and token 1
        dot_orig = torch.sum(q[:, :, 0, :] * k[:, :, 1, :])
        dot_rot = torch.sum(q_rot[:, :, 0, :] * k_rot[:, :, 1, :])

        self.assertTrue(
            torch.allclose(dot_orig, dot_rot, atol=1e-5),
            f"Inner product differed: orig={dot_orig.item()}, rot={dot_rot.item()}",
        )

    def test_compute_tensor_2d_coords(self):
        """Verifies dynamic 2D coordinate calculation aligns operands and scratchpad."""
        config = HRMRoPEConfig(vocab_size=32, n_embd=64, n_head=2, d_ffn=128)
        model = HRMRoPE(config)

        eq = format_addition_char_scratchpad(15, 25)
        tokens = self.tokenizer.encode(eq, add_bos=False, add_eos=False)
        idx = torch.tensor([tokens], dtype=torch.long)

        p_coords, s_coords = model.compute_tensor_2d_coords(
            idx,
            plus_token_id=self.tokenizer.token_to_id["+"],
            eq_token_id=self.tokenizer.token_to_id["="],
            lparen_token_id=self.tokenizer.token_to_id["("],
            think_start_id=self.tokenizer.token_to_id["<THINK>"],
            think_end_id=self.tokenizer.token_to_id["</THINK>"],
        )

        self.assertEqual(p_coords.shape, idx.shape)
        self.assertEqual(s_coords.shape, idx.shape)
        self.assertGreaterEqual(p_coords.max().item(), 0)
        self.assertLessEqual(s_coords.max().item(), 9)

    def test_forward_and_backward(self):
        """Verifies forward pass and gradient backpropagation through hierarchical recurrence."""
        config = HRMRoPEConfig(
            vocab_size=32,
            block_size=32,
            n_embd=64,
            n_head=2,
            d_ffn=128,
            n_fast_steps=2,
            n_slow_cycles=2,
        )
        model = HRMRoPE(config)
        model.train()

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T))
        place_coords = torch.randint(0, 10, (B, T))
        step_coords = torch.randint(0, 9, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        logits, loss = model(
            idx,
            place_coords=place_coords,
            step_coords=step_coords,
            targets=targets,
        )
        self.assertEqual(logits.shape, (B, T, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        # Positional targets calling convention (standard PyTorch / trainer.py)
        logits_pos, loss_pos = model(idx, targets)
        self.assertEqual(logits_pos.shape, (B, T, config.vocab_size))
        self.assertIsNotNone(loss_pos)
        self.assertGreater(loss_pos.item(), 0.0)

        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_generation(self):
        """Verifies autoregressive token generation with dynamic 2D coordinates."""
        config = HRMRoPEConfig(
            vocab_size=32,
            block_size=32,
            n_embd=64,
            n_head=2,
            d_ffn=128,
            n_fast_steps=1,
            n_slow_cycles=1,
        )
        model = HRMRoPE(config)
        model.eval()

        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=4, tokenizer=self.tokenizer)
        self.assertEqual(out.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
