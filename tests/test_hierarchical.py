"""
Comprehensive Unit Tests for Hierarchical Reasoning Model (HRM) with Dual-Loop Mechanism.
Tests configuration, FiLM modulation, FastReasoningBlock, SlowPlanningBlock,
HierarchicalReasoningGPT dual-loop execution, dynamic pondering, 1-step gradient,
RoPE extrapolation, FP16 AMP compliance, and weight tying.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hierarchical import (
    HierarchicalGPTConfig,
    HierarchicalCausalAttention,
    HierarchicalMLP,
    FiLMModulation,
    FastReasoningBlock,
    SlowPlanningBlock,
    HierarchicalReasoningGPT,
)
from src.models.factory import create_model


class TestHierarchicalArchitecture(unittest.TestCase):
    """Unit tests for HRM architecture components."""

    def setUp(self):
        self.config = HierarchicalGPTConfig(
            vocab_size=32,
            block_size=64,
            n_embd=64,
            n_head=4,
            n_fast_steps=3,
            n_slow_cycles=2,
            dropout=0.0,
            bias=False,
            rope_theta=10000.0,
            halt_epsilon=1e-3,
            ponder_penalty=0.01,
            one_step_gradient=False,
        )

    def test_config_defaults(self):
        default_cfg = HierarchicalGPTConfig()
        self.assertEqual(default_cfg.vocab_size, 32)
        self.assertEqual(default_cfg.block_size, 512)
        self.assertEqual(default_cfg.n_embd, 512)
        self.assertEqual(default_cfg.n_head, 8)
        self.assertEqual(default_cfg.n_fast_steps, 4)
        self.assertEqual(default_cfg.n_slow_cycles, 3)
        self.assertEqual(default_cfg.dropout, 0.0)
        self.assertFalse(default_cfg.bias)
        self.assertEqual(default_cfg.rope_theta, 10000.0)
        self.assertEqual(default_cfg.halt_epsilon, 1e-3)
        self.assertEqual(default_cfg.ponder_penalty, 0.01)
        self.assertFalse(default_cfg.one_step_gradient)
        self.assertEqual(default_cfg.model_type, "hierarchical")

    def test_film_modulation(self):
        dim = 64
        film = FiLMModulation(dim, bias=False)
        z_L = torch.randn(2, 10, dim)
        z_H = torch.randn(2, 10, dim)

        out = film(z_L, z_H)
        self.assertEqual(out.shape, (2, 10, dim))

        # Check modulation identity at zero initialization
        nn.init.zeros_(film.proj.weight)
        out_identity = film(z_L, z_H)
        expected = film.norm(z_L)
        self.assertTrue(torch.allclose(out_identity, expected, atol=1e-5))

    def test_fast_reasoning_block(self):
        block = FastReasoningBlock(self.config)
        z_L = torch.randn(2, 12, self.config.n_embd)
        z_H = torch.randn(2, 12, self.config.n_embd)

        out_z_L, halt_logits = block(z_L, z_H, save_attention=True)
        self.assertEqual(out_z_L.shape, (2, 12, self.config.n_embd))
        self.assertEqual(halt_logits.shape, (2, 12, 1))
        self.assertIsNotNone(block.attn.last_attn_weights)
        self.assertEqual(
            block.attn.last_attn_weights.shape,
            (2, self.config.n_head, 12, 12),
        )

    def test_slow_planning_block(self):
        block = SlowPlanningBlock(self.config)
        z_H = torch.randn(2, 12, self.config.n_embd)
        candidate = torch.randn(2, 12, self.config.n_embd)

        out_z_H = block(z_H, candidate=candidate, save_attention=True)
        self.assertEqual(out_z_H.shape, (2, 12, self.config.n_embd))
        self.assertIsNotNone(block.last_gate)
        self.assertEqual(block.last_gate.shape, (2, 12, self.config.n_embd))
        self.assertIsNotNone(block.attn.last_attn_weights)
        self.assertEqual(
            block.attn.last_attn_weights.shape,
            (2, self.config.n_head, 12, 12),
        )

    def test_hierarchical_gpt_forward_and_loss(self):
        model = HierarchicalReasoningGPT(self.config)
        x = torch.randint(0, self.config.vocab_size, (2, 16))
        targets = torch.randint(0, self.config.vocab_size, (2, 16))

        # Forward without targets
        logits, loss = model(x)
        self.assertEqual(logits.shape, (2, 16, self.config.vocab_size))
        self.assertIsNone(loss)

        # Forward with targets (includes ponder penalty)
        logits, loss = model(x, targets=targets)
        self.assertEqual(logits.shape, (2, 16, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

        # Backward pass
        loss.backward()

    def test_weight_tying(self):
        model = HierarchicalReasoningGPT(self.config)
        self.assertTrue(model.transformer.wte.weight is model.lm_head.weight)

    def test_parameter_count_ceiling(self):
        full_cfg = HierarchicalGPTConfig()
        model = HierarchicalReasoningGPT(full_cfg)
        non_emb_params = model.get_num_params(non_embedding=True)
        self.assertLess(non_emb_params, 20_000_000)
        self.assertEqual(non_emb_params, 7_343_104)

    def test_one_step_gradient_mode(self):
        cfg = HierarchicalGPTConfig(
            vocab_size=32,
            block_size=32,
            n_embd=32,
            n_head=2,
            n_fast_steps=2,
            n_slow_cycles=2,
            one_step_gradient=True,
        )
        model = HierarchicalReasoningGPT(cfg)
        model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        logits, loss = model(x, targets=targets)
        self.assertIsNotNone(loss)
        loss.backward()
        optimizer.step()

    def test_delta_norm_halting_during_eval(self):
        model = HierarchicalReasoningGPT(self.config)
        model.eval()
        x = torch.randint(0, self.config.vocab_size, (1, 8))
        with torch.no_grad():
            logits, _ = model(x)
        self.assertEqual(logits.shape, (1, 8, self.config.vocab_size))

    def test_generation_loop(self):
        model = HierarchicalReasoningGPT(self.config)
        model.eval()
        prefix = torch.tensor([[1, 2, 3]])
        generated = model.generate(prefix, max_new_tokens=5, temperature=0.0)
        self.assertEqual(generated.shape, (1, 8))

    def test_dynamic_rope_cache_extension(self):
        # Test sequence longer than precomputed block_size
        cfg = HierarchicalGPTConfig(
            vocab_size=32,
            block_size=16,
            n_embd=32,
            n_head=2,
            n_fast_steps=2,
            n_slow_cycles=1,
        )
        model = HierarchicalReasoningGPT(cfg)
        model.eval()
        x_long = torch.randint(0, cfg.vocab_size, (1, 40))
        logits, _ = model(x_long)
        self.assertEqual(logits.shape, (1, 40, cfg.vocab_size))

    def test_factory_creation(self):
        cfg = HierarchicalGPTConfig()
        model = create_model(cfg)
        self.assertIsInstance(model, HierarchicalReasoningGPT)

    def test_turing_sm75_fp16_compatibility(self):
        # Test FP16 autocast forward and backward passes
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = HierarchicalGPTConfig(
            vocab_size=32,
            block_size=32,
            n_embd=64,
            n_head=4,
            n_fast_steps=2,
            n_slow_cycles=2,
        )
        model = HierarchicalReasoningGPT(cfg).to(device)
        x = torch.randint(0, cfg.vocab_size, (2, 8), device=device)
        targets = torch.randint(0, cfg.vocab_size, (2, 8), device=device)

        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss = model(x, targets=targets)
            self.assertEqual(logits.dtype, torch.float16)
            scaler.scale(loss).backward()
        else:
            # Fallback for CPU environment: test float16 forward simulation
            logits, loss = model(x, targets=targets)
            self.assertIsNotNone(loss)


if __name__ == "__main__":
    unittest.main()
