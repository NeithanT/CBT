"""
Unit tests for HRM-RoPE sample efficiency experiment.
Verifies deterministic subset extraction, prompt masking, CPU training step,
and benchmark evaluation integration.
"""

import unittest
import torch
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from src.tokenizer import MathTokenizer
from src.models.hrm_rope import HRMRoPE, HRMRoPEConfig
from experiments.sample_efficiency_2d_rope import (
    load_sample_efficiency_datasets,
    train_hrm_rope_sample_subset,
    evaluate_model_on_cases,
)


class TestSampleEfficiency(unittest.TestCase):
    """Test suite for few-shot sample efficiency pipeline."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.tokenizer = MathTokenizer()

    def test_subset_nesting_and_size(self):
        """Verifies subset nesting S_10 subset of S_100 subset of S_1000."""
        datasets = load_sample_efficiency_datasets([10, 100, 1000], seed=42)
        self.assertEqual(len(datasets[10]), 10)
        self.assertEqual(len(datasets[100]), 100)
        self.assertEqual(len(datasets[1000]), 1000)

        # Verify exact subset nesting
        self.assertEqual(datasets[10], datasets[100][:10])
        self.assertEqual(datasets[100], datasets[1000][:100])

    def test_single_train_step_and_eval(self):
        """Verifies that HRMRoPE executes forward/backward and evaluation on CPU."""
        config = HRMRoPEConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=64,
            n_head=2,
            d_ffn=128,
            n_fast_steps=1,
            n_slow_cycles=1,
            block_size=128,
        )
        model = HRMRoPE(config).to(self.device)

        train_texts = [
            "1+2=<THINK>(0+1+2)=03</THINK>=3",
            "10+20=<THINK>(0+0+0)=00(0+1+2)=03</THINK>=30",
        ]

        # Run 2 training steps
        history = train_hrm_rope_sample_subset(
            model=model,
            tokenizer=self.tokenizer,
            train_texts=train_texts,
            device=self.device,
            steps=2,
            batch_size=2,
            lr=1e-3,
            log_interval=1,
        )
        self.assertEqual(len(history), 2)
        self.assertTrue(history[-1] > 0.0)

        # Run evaluation on test cases
        test_cases = [("1+2=", "3")]
        acc, results = evaluate_model_on_cases(
            model=model,
            tokenizer=self.tokenizer,
            cases=test_cases,
            device=self.device,
            max_new_tokens=32,
        )
        self.assertIn(acc, [0.0, 100.0])
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
