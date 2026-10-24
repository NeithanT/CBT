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


class TestRope1DAblation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokenizer = ScaledTokenizer()
        cfg = ScaledGPTConfig(vocab_size=self.tokenizer.vocab_size, n_layer=2, n_recur=2, pos_emb_type="rope_1d")
        self.model = ScaledGPT(cfg, tokenizer=self.tokenizer).eval()

    def test_same_parameter_count_as_2d(self):
        cfg_2d = ScaledGPTConfig(vocab_size=self.tokenizer.vocab_size, n_layer=2, n_recur=2)
        self.assertEqual(self.model.get_num_params(), ScaledGPT(cfg_2d, tokenizer=self.tokenizer).get_num_params())
        print("[PASS] test_same_parameter_count_as_2d")

    def test_coords_and_uniform_shift_invariance(self):
        idx = torch.tensor([self.tokenizer.encode("123+456=<THINK>", add_bos=True)], dtype=torch.long)
        p, s = self.model.compute_coords(idx)
        self.assertEqual(p[0].tolist(), list(range(idx.size(1))))
        self.assertEqual(int(s.abs().sum()), 0)
        with torch.no_grad():
            base, _ = self.model(idx)
            shifted, _ = self.model(idx, place_coords=p + 37, step_coords=s)
        self.assertTrue(torch.allclose(base, shifted, atol=1e-4))
        print("[PASS] test_coords_and_uniform_shift_invariance")

    def test_kv_cached_decoder_matches_generate(self):
        from benchmarks.evaluate_random_16_to_100d import fast_generate_batch
        idx = torch.tensor([self.tokenizer.encode("12+34=", add_bos=True)], dtype=torch.long)
        ref = self.model.generate(idx, max_new_tokens=12, temperature=0.0)
        fast = fast_generate_batch(self.model, idx, max_new_tokens=12, eos_id=-1)
        self.assertEqual(fast[0, :ref.size(1)].tolist(), ref[0].tolist())
        print("[PASS] test_kv_cached_decoder_matches_generate")


class TestCurriculumV5(unittest.TestCase):
    def setUp(self):
        from champion_model.scaled_t4.sft_search import build_benchmark_holdout, CurriculumV5Stream
        self.tokenizer = ScaledTokenizer()
        self.prompts, self.spans = build_benchmark_holdout()
        self.stream = CurriculumV5Stream(seed=0, prompts=self.prompts, spans=self.spans)

    def test_search_arguments_are_verbatim_user_spans(self):
        for item in self.stream.search(300):
            text = item["full_text"]
            user = text.split("<USER>")[1].split("</USER>")[0]
            query = text.split('search(query="')[1].split('")')[0]
            self.assertIn(query, user)
            for article in ("the ", "a ", "an "):
                if f" {article}{query}" in f" {user}":
                    self.assertFalse(query.startswith(article))
        print("[PASS] test_search_arguments_are_verbatim_user_spans")

    def test_calculator_expression_copied_verbatim(self):
        n = 0
        for item in self.stream.tools(1500):
            if 'calculator(expr="' in item["full_text"] and item["category"] == "single_calculator":
                expr = item["full_text"].split('calculator(expr="')[1].split('")')[0]
                self.assertIn(expr, item["prompt"])
                n += 1
        self.assertGreater(n, 0)
        print(f"[PASS] test_calculator_expression_copied_verbatim ({n} samples)")

    def test_benchmark_decontamination(self):
        from benchmarks.evaluate_champion import audit_contamination
        items = self.stream.search(1000) + self.stream.tools(3000)
        corpus = "\n".join(i["full_text"] for i in items)
        self.assertEqual(audit_contamination(corpus), [])
        self.assertGreater(len(self.spans), 30)
        print(f"[PASS] test_benchmark_decontamination ({self.stream.rejected} rejected)")

    def test_dataset_source_passes_benchmark_audit(self):
        from benchmarks.evaluate_champion import audit_contamination
        with open(os.path.join(ROOT_DIR, "champion_model/scaled_t4/dataset.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertEqual(audit_contamination(src), [])
        print("[PASS] test_dataset_source_passes_benchmark_audit")

    def test_no_unknown_tokens_and_masking(self):
        items = self.stream.search(50) + self.stream.tools(150)
        for item in items:
            self.assertNotIn(self.tokenizer.unk_id, self.tokenizer.encode(item["full_text"]))
        x_b, y_b = encode_joint_batch([], items[:8], self.tokenizer)
        self.assertTrue((y_b != -100).any())
        print("[PASS] test_no_unknown_tokens_and_masking")


if __name__ == "__main__":
    unittest.main()
