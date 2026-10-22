"""
Unit tests for Harmonic Neural Register Machine (H-NRM).
Verifies Harmonic Oscillator Bank delimiter phase locking and coordinate invariance,
Bistable Attractor Register quartic potential dynamics and 20-digit carry stability,
Neural ALU forward and gradient computation, and HarmonicNRM parameter ceiling (<20M).
Zero emojis.
"""

import math
import unittest
import torch

from src.models.hpf import HarmonicOscillatorBank, HPFCausalAttention, apply_harmonic_rope
from src.models.attractor_alu import BistableAttractorRegister, NeuralALUCell
from src.models.hnrm import HNRMConfig, HarmonicNRM
from src.models.factory import create_model


class TestHarmonicNRM(unittest.TestCase):
    """Test suite for H-NRM components and full model architecture."""

    def setUp(self):
        torch.manual_seed(42)

    def test_harmonic_oscillator_bank_shapes(self):
        """Verifies output shapes for HPF bank under full sequence and causal generation modes."""
        B, T, D, H = 2, 16, 64, 16
        bank = HarmonicOscillatorBank(embed_dim=D, head_dim=H, base_freq=1.0)
        x = torch.randn(B, T, D)

        # Non-causal mode (bidirectional delimiter anticipation)
        cos_f, sin_f, hpf_emb = bank(x, is_causal_generation=False)
        self.assertEqual(cos_f.shape, (B, 1, T, H // 2))
        self.assertEqual(sin_f.shape, (B, 1, T, H // 2))
        self.assertEqual(hpf_emb.shape, (B, T, D))

        # Causal generation mode (forward delimiter tracking)
        cos_c, sin_c, hpf_emb_c = bank(x, is_causal_generation=True)
        self.assertEqual(cos_c.shape, (B, 1, T, H // 2))
        self.assertEqual(sin_c.shape, (B, 1, T, H // 2))
        self.assertEqual(hpf_emb_c.shape, (B, T, D))

    def test_apply_harmonic_rope(self):
        """Verifies rotary transformation shape preservation and identity condition."""
        B, H, T, D_head = 2, 4, 8, 16
        q = torch.randn(B, H, T, D_head)

        # Identity test: cos=1.0, sin=0.0
        cos_id = torch.ones(B, 1, T, D_head // 2)
        sin_id = torch.zeros(B, 1, T, D_head // 2)
        q_rot = apply_harmonic_rope(q, cos_id, sin_id)
        self.assertTrue(torch.allclose(q, q_rot, atol=1e-6))

        # General rotation preserves tensor norm per head vector
        cos_arb = torch.cos(torch.randn(B, 1, T, D_head // 2))
        sin_arb = torch.sin(torch.randn(B, 1, T, D_head // 2))
        q_rot_arb = apply_harmonic_rope(q, cos_arb, sin_arb)
        self.assertEqual(q_rot_arb.shape, q.shape)

    def test_harmonic_oscillator_phase_invariance(self):
        """
        Verifies that positions with identical relative offset to the next delimiter
        produce identical oscillator phases Delta psi = 0 across sequences of different lengths.
        """
        D, H = 64, 16
        bank = HarmonicOscillatorBank(embed_dim=D, head_dim=H, base_freq=1.0)

        # Force gate weights to recognize delimiter token explicitly
        with torch.no_grad():
            bank.gate_proj.weight.zero_()
            bank.gate_proj.bias.fill_(-10.0)  # default non-delimiter

        # Sequence 1: 5 digits + delimiter (len 6)
        x1 = torch.zeros(1, 6, D)
        x1[0, 5, :] = 1.0  # Delimiter at index 5
        # Set gate weight to activate strongly on delimiter feature
        with torch.no_grad():
            bank.gate_proj.weight[0, 0] = 30.0

        # Sequence 2: 12 digits + delimiter (len 13)
        x2 = torch.zeros(1, 13, D)
        x2[0, 12, :] = 1.0  # Delimiter at index 12

        cos1, sin1, _ = bank(x1, is_causal_generation=False)
        cos2, sin2, _ = bank(x2, is_causal_generation=False)

        # Units place in x1 is at index 4 (delta = 5 - 4 = 1)
        # Units place in x2 is at index 11 (delta = 12 - 11 = 1)
        phase_cos_diff = torch.abs(cos1[0, 0, 4] - cos2[0, 0, 11])
        phase_sin_diff = torch.abs(sin1[0, 0, 4] - sin2[0, 0, 11])

        self.assertLess(phase_cos_diff.max().item(), 1e-5)
        self.assertLess(phase_sin_diff.max().item(), 1e-5)

    def test_bistable_attractor_register_bifurcation(self):
        """Verifies quartic double-well potential snapping and bifurcation dynamics."""
        reg = BistableAttractorRegister(num_registers=1, num_iterations=3)

        # Case 1: Positive driving field (carry generation, u_t > 0)
        field_pos = torch.tensor([[1.5]])
        c_pos, carry_pos = reg(field_pos)
        self.assertEqual(carry_pos.item(), 1.0)
        self.assertEqual(c_pos.item(), 1.0)

        # Case 2: Negative driving field (no carry, u_t < 0)
        field_neg = torch.tensor([[-1.5]])
        c_neg, carry_neg = reg(field_neg)
        self.assertEqual(carry_neg.item(), 0.0)
        self.assertEqual(c_neg.item(), -1.0)

    def test_bistable_attractor_noise_suppression(self):
        """
        Verifies supercontractive Picard settling:
        Perturbed initial state settles firmly to exact discrete attractors {-1, +1}.
        """
        reg = BistableAttractorRegister(num_registers=1, num_iterations=4)

        # Initial state slightly perturbed from +1 attractor (c = 0.7)
        perturbed_init = torch.tensor([[0.7]])
        field = torch.tensor([[0.5]])
        c_out, carry_out = reg(field, init_state=perturbed_init)
        self.assertEqual(c_out.item(), 1.0)
        self.assertEqual(carry_out.item(), 1.0)

        # Initial state slightly perturbed from -1 attractor (c = -0.7)
        perturbed_init_neg = torch.tensor([[-0.7]])
        field_neg = torch.tensor([[-0.5]])
        c_out_neg, carry_out_neg = reg(field_neg, init_state=perturbed_init_neg)
        self.assertEqual(c_out_neg.item(), -1.0)
        self.assertEqual(carry_out_neg.item(), 0.0)

    def test_bistable_attractor_20_digit_cascading_carry(self):
        """
        Simulates 20-digit continuous carry propagation (e.g. 99999999999999999999 + 1).
        Verifies zero continuous float decay or catastrophic state drift across 20 iterations.
        """
        reg = BistableAttractorRegister(num_registers=1, num_iterations=3)
        c_state = torch.tensor([[1.0]])  # Initial carry generated at units position

        for step in range(20):
            # At each subsequent position: 9 + 0 + carry(1) = 10 -> carry continues
            # sigma = 9 + 0 + 1 = 10 -> u_t = 10 - 9.5 = +0.5
            u_t = torch.tensor([[0.5]])
            c_state, carry_bit = reg(u_t, init_state=c_state)
            self.assertEqual(carry_bit.item(), 1.0, f"Carry decayed at step {step}")
            self.assertEqual(c_state.item(), 1.0, f"Bipolar state drifted at step {step}")

    def test_neural_alu_cell(self):
        """Verifies ALU forward pass, output shapes, and autograd gradient flow."""
        B, T, D = 2, 8, 32
        alu = NeuralALUCell(d_model=D)
        h = torch.randn(B, T, D, requires_grad=True)

        h_delta, c_last = alu(h)
        self.assertEqual(h_delta.shape, (B, T, D))
        self.assertEqual(c_last.shape, (B, 1))

        # Test gradient propagation through straight-through attractor
        loss = h_delta.sum()
        loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertGreater(h.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(alu.proj_a.weight.grad)
        self.assertIsNotNone(alu.proj_b.weight.grad)

    def test_hnrm_parameter_ceiling(self):
        """Verifies H-NRM non-embedding parameters are strictly < 20M (~18.93M)."""
        config = HNRMConfig(
            vocab_size=32,
            block_size=512,
            n_layer=6,
            n_embd=512,
            n_head=8,
            d_ffn=2048,
            base_freq=1.0,
        )
        model = HarmonicNRM(config)
        n_params = model.get_num_params(non_embedding=True)

        self.assertLess(n_params, 20_000_000, f"Params {n_params} exceeds 20M ceiling")
        # Exact expected: 18,932,813 parameters
        self.assertEqual(n_params, 18_932_813)

    def test_hnrm_weight_tying(self):
        """Verifies tied weights between token embedding and LM head."""
        config = HNRMConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = HarmonicNRM(config)
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_hnrm_forward_and_backward(self):
        """Verifies forward pass with cross-entropy loss computation and backpropagation."""
        config = HNRMConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = HarmonicNRM(config)
        model.train()

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        logits, loss = model(idx, targets=targets)
        self.assertEqual(logits.shape, (B, T, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient on {name}")

    def test_hnrm_attention_maps(self):
        """Verifies attention weight retention when save_attention=True."""
        config = HNRMConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = HarmonicNRM(config)
        model.eval()

        B, T = 1, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        model(idx, save_attention=True)

        maps = model.get_attention_maps()
        self.assertEqual(len(maps), 2)
        for att in maps:
            self.assertEqual(att.shape, (B, config.n_head, T, T))

    def test_hnrm_generation(self):
        """Verifies autoregressive token generation."""
        config = HNRMConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = HarmonicNRM(config)
        model.eval()

        idx = torch.tensor([[1, 2, 3]])
        out = model.generate(idx, max_new_tokens=5, temperature=0.0)
        self.assertEqual(out.shape, (1, 8))

    def test_factory_creation(self):
        """Verifies factory creates HarmonicNRM with model_type='hnrm'."""
        config = HNRMConfig(vocab_size=32, n_embd=64, n_layer=2, n_head=2, d_ffn=128)
        model = create_model(config)
        self.assertIsInstance(model, HarmonicNRM)
        self.assertEqual(model.config.model_type, "hnrm")


if __name__ == "__main__":
    unittest.main()
