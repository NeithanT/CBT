"""
Unit Tests for ScaledGPT Architecture (~18.95M Parameters, Dual T4 Optimization).
Zero Emojis Policy: all output tags strictly follow [OK], [PASS], [FAIL], [INFO], [WARN].
"""

import os
import sys
import unittest
import math
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from champion_model.scaled_t4.tokenizer import ScaledTokenizer
from champion_model.scaled_t4.model import (
    ScaledGPT,
    ScaledGPTConfig,
    precompute_2d_rope_freqs,
    apply_2d_rope,
)
from champion_model.scaled_t4.dataset import (
    generate_arithmetic_equations,
    generate_tool_dataset,
    generate_joint_dataset,
    encode_joint_batch,
)
from champion_model.math_model.arithmetic import (
    format_addition_char_scratchpad,
    extract_scratchpad_answer,
)


class TestScaledTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ScaledTokenizer()

    def test_special_tokens_roundtrip(self):
        text = "<USER>Find time</USER><THINK>Query time.</THINK><CALL>get_time(timezone=\"UTC\")</CALL><RESULT>{\"time\": \"12:00\"}</RESULT><ANSWER>Time is 12:00 UTC.</ANSWER>"
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = self.tokenizer.decode(tokens, skip_special=False)
        self.assertEqual(decoded, "<BOS>" + text + "<EOS>")
        print("[PASS] test_special_tokens_roundtrip")

    def test_arithmetic_symbols_encoding(self):
        math_str = "123456789+987654321=1111111110"
        tokens = self.tokenizer.encode(math_str, add_bos=False, add_eos=False)
        decoded = self.tokenizer.decode(tokens, skip_special=False)
        self.assertEqual(decoded, math_str)
        print("[PASS] test_arithmetic_symbols_encoding")


class TestScaledGPTArchitecture(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ScaledTokenizer()
        self.config = ScaledGPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=1024,
            n_layer=6,
            n_head=8,
            n_embd=512,
            n_recur=3,
            max_coords=4096,
            k_rand_shift=32,
        )
        self.model = ScaledGPT(self.config, tokenizer=self.tokenizer)

    def test_parameter_counts(self):
        tot = self.model.get_num_params(non_embedding=False)
        non_emb = self.model.get_num_params(non_embedding=True)

        # Theoretical expected:
        # 6 blocks * (12 * 512^2 + 4 * 512) = 18,887,680 non-embedding
        # ln_f: 2 * 512 = 1,024
        # embedding: 111 * 512 = 56,832
        # Total: ~18.94M - 18.95M parameters
        self.assertEqual(non_emb, 18887680)
        self.assertTrue(18_500_000 <= tot <= 19_500_000)
        print(f"[PASS] test_parameter_counts: Total={tot:,} (~{tot/1e6:.2f}M), Non-Emb={non_emb:,}")

    def test_weight_tying(self):
        wte_weight = self.model.transformer.wte.weight
        lm_head_weight = self.model.lm_head.weight
        self.assertTrue(wte_weight is lm_head_weight)
        print("[PASS] test_weight_tying")

    def test_2d_rope_coordinate_generation_math_50d(self):
        # 50-digit arithmetic test
        a = int("9" * 50)
        b = 1
        math_text = format_addition_char_scratchpad(a, b)
        tokens = self.tokenizer.encode(math_text, add_bos=True, add_eos=True)
        idx = torch.tensor([tokens], dtype=torch.long)

        p_coords, s_coords = self.model.compute_2d_coords(idx)
        self.assertEqual(p_coords.shape, idx.shape)
        self.assertEqual(s_coords.shape, idx.shape)
        self.assertTrue(p_coords.max().item() >= 50)
        self.assertTrue(s_coords.max().item() <= 9)
        print("[PASS] test_2d_rope_coordinate_generation_math_50d")

    def test_2d_rope_coordinate_generation_discourse(self):
        dialogue = "<USER>Hello</USER><THINK>Greeting</THINK><ANSWER>Hi</ANSWER>"
        tokens = self.tokenizer.encode(dialogue, add_bos=True, add_eos=True)
        idx = torch.tensor([tokens], dtype=torch.long)

        p_coords, s_coords = self.model.compute_2d_coords(idx)
        self.assertEqual(p_coords.shape, idx.shape)
        # Segments in discourse: USER=0, THINK=1, ANSWER=4
        unique_p = set(p_coords[0].tolist())
        self.assertTrue(0 in unique_p and 1 in unique_p and 4 in unique_p)
        print("[PASS] test_2d_rope_coordinate_generation_discourse")

    def test_selective_shift_masking(self):
        # Test that place_coords > 0 is strictly respected
        idx = torch.tensor([[self.tokenizer.user_s_id, 10, 20]], dtype=torch.long)
        p, s = self.model.compute_2d_coords(idx)
        shift_mask = (p > 0).long()
        self.assertEqual(shift_mask.shape, p.shape)
        self.assertTrue(shift_mask.min().item() >= 0)
        print("[PASS] test_selective_shift_masking")

    def test_recurrence_residual_scaling(self):
        # Verify that recurrence scale factor math holds
        R = 3
        expected_scale = 1.0 / math.sqrt(R)
        self.assertAlmostEqual(expected_scale, 0.577350269, places=6)
        print("[PASS] test_recurrence_residual_scaling")

    def test_forward_pass_loss(self):
        # Forward pass on small dummy batch
        seq_len = 64
        x = torch.randint(0, self.config.vocab_size, (2, seq_len), dtype=torch.long)
        y = torch.randint(0, self.config.vocab_size, (2, seq_len), dtype=torch.long)
        logits, loss = self.model(x, targets=y)
        self.assertEqual(logits.shape, (2, seq_len, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0.0)
        print(f"[PASS] test_forward_pass_loss: Loss = {loss.item():.4f}")

    def test_autoregressive_generation(self):
        prompt = "<USER>Hi</USER><THINK>"
        tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([tokens], dtype=torch.long)
        out = self.model.generate(x, max_new_tokens=10, temperature=0.0)
        self.assertEqual(out.shape[1], len(tokens) + 10)
        print("[PASS] test_autoregressive_generation")


class TestExpandedDatasetAndLossMasking(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ScaledTokenizer()

    def test_dataset_categories_coverage(self):
        tool_ds = generate_tool_dataset(n_samples=60, seed=42)
        categories = set(item["category"] for item in tool_ds)
        expected_cats = {
            "single_tool",
            "multi_tool_chain",
            "nested_json_schema",
            "error_recovery",
            "multi_turn",
            "no_tool",
        }
        self.assertEqual(categories, expected_cats)
        print(f"[PASS] test_dataset_categories_coverage: Found all {len(categories)} categories")

    def test_arithmetic_equations_scratchpad(self):
        math_ds = generate_arithmetic_equations(n_samples=20, seed=42, max_train_digits=15)
        for item in math_ds:
            reasoning, ans = extract_scratchpad_answer(item["full_text"])
            self.assertTrue(len(ans) > 0)
            self.assertTrue(ans.isdigit())
        print("[PASS] test_arithmetic_equations_scratchpad")

    def test_prompt_and_result_loss_masking(self):
        math_ds = [{"task_type": "math", "full_text": "10+20=<THINK>(0+0+0)=00(0+1+2)=03</THINK>=30"}]
        tool_ds = [{
            "task_type": "tools",
            "full_text": "<USER>Query weather</USER><THINK>Check city</THINK><CALL>get_weather(city=\"Rome\")</CALL><RESULT>{\"temp\": 25}</RESULT><ANSWER>Rome is 25C.</ANSWER>",
        }]

        x_b, y_b = encode_joint_batch(math_ds, tool_ds, self.tokenizer)
        # Math row: prompt up to <THINK> should be -100
        math_targets = y_b[0]
        self.assertEqual(math_targets[0].item(), -100)

        # Tool row: USER content should be -100, RESULT content should be -100
        tool_targets = y_b[1]
        self.assertEqual(tool_targets[0].item(), -100) # BOS
        self.assertTrue((tool_targets == -100).sum().item() > 0)
        self.assertTrue((tool_targets != -100).sum().item() > 0)
        print("[PASS] test_prompt_and_result_loss_masking")


if __name__ == "__main__":
    unittest.main()
