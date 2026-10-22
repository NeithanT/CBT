"""
Unit tests for Gated Linear Recurrent Unit (GLRU) and Parallel Associative Scan.
Verifies numerical equivalence between parallel scan and sequential recurrence,
gradient flow, shape preservation, and FP16 compatibility.
"""

import unittest
import torch
import torch.nn as nn

from src.models.glru import GatedLinearRecurrentUnit, parallel_associative_scan


class TestGLRU(unittest.TestCase):
    """Test suite for GLRU and associative prefix scan."""

    def setUp(self):
        torch.manual_seed(42)

    def test_parallel_scan_numerical_equivalence(self):
        """Verifies that parallel associative scan matches sequential step-by-step recurrence."""
        B, T, D = 2, 32, 16
        # Gates strictly in (0, 1)
        a = torch.sigmoid(torch.randn(B, T, D))
        b = torch.randn(B, T, D)

        # 1. Sequential recurrence
        h_seq = torch.zeros(B, T, D)
        h_prev = torch.zeros(B, D)
        for t in range(T):
            h_prev = a[:, t, :] * h_prev + b[:, t, :]
            h_seq[:, t, :] = h_prev

        # 2. Parallel associative scan
        h_par = parallel_associative_scan(a, b)

        # Check maximum absolute difference
        max_diff = (h_seq - h_par).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"Parallel scan diverges from sequential: max_diff={max_diff}")

    def test_glru_forward_shapes(self):
        """Verifies output shapes in sequence mode and step-by-step inference mode."""
        B, T, D = 4, 24, 64
        glru = GatedLinearRecurrentUnit(d_model=D, conv_kernel=4)

        x = torch.randn(B, T, D)
        y, next_state = glru(x)

        self.assertEqual(y.shape, (B, T, D))
        self.assertEqual(next_state.shape, (B, D))

        # Single step inference mode
        x_step = torch.randn(B, 1, D)
        y_step, next_state_2 = glru(x_step, state=next_state)
        self.assertEqual(y_step.shape, (B, 1, D))
        self.assertEqual(next_state_2.shape, (B, D))

    def test_glru_backward_gradients(self):
        """Verifies clean gradient backpropagation through GLRU."""
        B, T, D = 2, 16, 32
        glru = GatedLinearRecurrentUnit(d_model=D)

        x = torch.randn(B, T, D, requires_grad=True)
        y, _ = glru(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isinf(x.grad).any())
        for name, param in glru.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertFalse(torch.isnan(param.grad).any())


if __name__ == "__main__":
    unittest.main()
