"""
Comprehensive Unit Tests for Math GPT
====================================
Tests for Tokenizer, Dataset, Model Architecture, Causal Attention Masking,
Training Step Integration, and Configurations/Constants.
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import (
    TOKENIZER_CONSTANTS,
    MODEL_CONSTANTS,
    TRAINING_CONSTANTS,
    TESTING_CONSTANTS,
    load_config,
)
from src.tokenizer import MathTokenizer
from src.dataset import MathEquationGenerator, MathDataset
from src.model import MathGPT, GPTConfig, CausalSelfAttention


class TestConfigAndConstants(unittest.TestCase):
    """Verifies that config.json and constants.py are properly defined and loaded."""

    def test_config_json_loaded(self):
        raw_cfg = load_config()
        self.assertIn("tokenizer", raw_cfg)
        self.assertIn("model", raw_cfg)
        self.assertIn("training", raw_cfg)
        self.assertIn("testing", raw_cfg)

    def test_constants_structure(self):
        self.assertGreater(len(TOKENIZER_CONSTANTS.digits), 0)
        self.assertGreater(len(TOKENIZER_CONSTANTS.operators), 0)
        self.assertGreater(len(TOKENIZER_CONSTANTS.special_tokens), 0)
        self.assertGreater(len(TESTING_CONSTANTS.benchmark_prompts), 0)
        self.assertGreater(TESTING_CONSTANTS.model_settings.n_embd, 0)


class TestMathTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MathTokenizer(constants=TOKENIZER_CONSTANTS)

    def test_vocab_contents(self):
        for digit in TOKENIZER_CONSTANTS.digits:
            self.assertIn(digit, self.tokenizer.vocab)
        for op in TOKENIZER_CONSTANTS.operators:
            self.assertIn(op, self.tokenizer.vocab)
        for special in TOKENIZER_CONSTANTS.special_tokens:
            self.assertIn(special, self.tokenizer.vocab)

    def test_encode_decode_roundtrip(self):
        for expr in TESTING_CONSTANTS.sample_equations:
            encoded = self.tokenizer.encode(expr, add_bos=True, add_eos=True)
            self.assertEqual(encoded[0], self.tokenizer.bos_id)
            self.assertEqual(encoded[-1], self.tokenizer.eos_id)

            decoded = self.tokenizer.decode(encoded, skip_special=True)
            self.assertEqual(decoded, expr)

    def test_unknown_token_handling(self):
        for expr in TESTING_CONSTANTS.invalid_expressions:
            encoded = self.tokenizer.encode(expr)
            self.assertIn(self.tokenizer.unk_id, encoded)


class TestMathDataset(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MathTokenizer(constants=TOKENIZER_CONSTANTS)
        self.generator = MathEquationGenerator(seed=TESTING_CONSTANTS.seed)

    def test_generator_validity(self):
        eq = self.generator.generate_addition()
        parts = eq.split("=")
        self.assertEqual(len(parts), 2)
        lhs_terms = parts[0].split("+")
        self.assertEqual(int(lhs_terms[0]) + int(lhs_terms[1]), int(parts[1]))

    def test_place_value_thinking_formatter(self):
        from src.arithmetic import format_addition_place_value_thinking, extract_scratchpad_answer
        formatted = format_addition_place_value_thinking(32, 59)
        self.assertIn("<THINK>", formatted)
        self.assertIn("30", formatted)
        self.assertIn("50", formatted)
        self.assertIn("10", formatted)
        self.assertIn("1", formatted)
        _, ans = extract_scratchpad_answer(formatted)
        self.assertEqual(ans, "91")

    def test_dataset_item_shapes_and_masking(self):
        samples = TESTING_CONSTANTS.sample_equations
        block_size = TESTING_CONSTANTS.model_settings.block_size
        dataset = MathDataset(samples, self.tokenizer, block_size=block_size, mask_prompt=True)
        self.assertEqual(len(dataset), len(samples))

        x, y = dataset[0]
        self.assertEqual(x.shape, torch.Size([block_size]))
        self.assertEqual(y.shape, torch.Size([block_size]))

        # Check prompt masking: targets before '=' should be -100
        eq_token_id = self.tokenizer.token_to_id["="]
        if eq_token_id in x.tolist():
            eq_pos = (x == eq_token_id).nonzero(as_tuple=True)[0].item()
            for idx in range(eq_pos):
                self.assertEqual(y[idx].item(), -100)

    def test_commutativity_splits(self):
        from src.dataset import generate_commutativity_dataset
        train_eq, test_c_unseen, test_comm_seen, test_comm_unseen = generate_commutativity_dataset(
            min_val=0, max_val=10, train_ratio=0.8, seed=42
        )
        # Verify canonical ordering (a <= b) in train_eq
        for eq in train_eq:
            lhs = eq.split("=")[0]
            a, b = map(int, lhs.split("+"))
            self.assertLessEqual(a, b)

        # Verify no leakage of commuted pairs in train
        train_set = set(train_eq)
        for eq in test_comm_seen:
            self.assertNotIn(eq, train_set)
            lhs = eq.split("=")[0]
            a, b = map(int, lhs.split("+"))
            self.assertGreater(a, b)


class TestMathGPTModel(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MathTokenizer(constants=TOKENIZER_CONSTANTS)
        t_cfg = TESTING_CONSTANTS.model_settings
        self.config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=t_cfg.block_size,
            n_layer=t_cfg.n_layer,
            n_head=t_cfg.n_head,
            n_embd=t_cfg.n_embd,
            dropout=t_cfg.dropout,
        )
        self.model = MathGPT(self.config)

    def test_forward_pass_shapes(self):
        x = torch.randint(0, self.config.vocab_size, (2, 8))
        logits, loss = self.model(x)
        self.assertEqual(logits.shape, (2, 8, self.config.vocab_size))
        self.assertIsNone(loss)

        targets = torch.randint(0, self.config.vocab_size, (2, 8))
        logits, loss = self.model(x, targets)
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_causal_masking(self):
        """
        Ensure future tokens cannot influence earlier tokens.
        If we modify token at position 4, logits at positions 0..3 must remain unchanged.
        """
        self.model.eval()
        x1 = torch.tensor([[1, 4, 14, 5, 8, 10]])
        x2 = torch.tensor([[1, 4, 14, 5, 9, 11]])

        with torch.no_grad():
            logits1, _ = self.model(x1)
            logits2, _ = self.model(x2)

        diff = torch.max(torch.abs(logits1[0, :4, :] - logits2[0, :4, :])).item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_attention_extraction(self):
        self.model.eval()
        x = torch.randint(0, self.config.vocab_size, (1, 6))
        logits, _ = self.model(x, save_attention=True)
        attn_maps = self.model.get_attention_maps()
        self.assertEqual(len(attn_maps), self.config.n_layer)
        self.assertEqual(attn_maps[0].shape, (1, self.config.n_head, 6, 6))

    def test_generate(self):
        self.model.eval()
        prefix = torch.tensor([[1, 4, 14]])
        generated = self.model.generate(prefix, max_new_tokens=4, temperature=0.0)
        self.assertEqual(generated.shape, (1, 7))


class TestOptimizationStep(unittest.TestCase):

    def test_train_step(self):
        t_cfg = TESTING_CONSTANTS.model_settings
        config = GPTConfig(
            vocab_size=25,
            block_size=t_cfg.block_size,
            n_layer=1,
            n_head=1,
            n_embd=16,
        )
        model = MathGPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randint(0, 25, (2, 8))
        y = torch.randint(0, 25, (2, 8))

        optimizer.zero_grad()
        _, loss1 = model(x, y)
        loss1.backward()
        optimizer.step()

        _, loss2 = model(x, y)
        self.assertTrue(torch.is_tensor(loss2))

    def test_rope_architecture(self):
        config = GPTConfig(
            vocab_size=25,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
            pos_emb_type="rope",
        )
        model = MathGPT(config)
        self.assertFalse(hasattr(model.transformer, "wpe"))
        
        # Test forward pass with sequence length longer than block_size (relative position robustness)
        x = torch.randint(0, 25, (2, 24))
        y = torch.randint(0, 25, (2, 24))
        logits, loss = model(x, y, save_attention=True)
        self.assertEqual(logits.shape, (2, 24, 25))
        self.assertIsNotNone(loss)
        loss.backward()

        # Test generation beyond block_size
        prefix = torch.randint(0, 25, (1, 5))
        gen = model.generate(prefix, max_new_tokens=20, temperature=0.0)
        self.assertEqual(gen.shape, (1, 25))


class TestPromptMaskingAndFactory(unittest.TestCase):
    """Verifies that prompt masking targets are set to -100 up to '=' and factory works."""

    def test_encode_texts_to_batch_masking(self):
        from training.trainer import encode_texts_to_batch
        tokenizer = MathTokenizer()
        texts = ["12+34=46"]
        x, y = encode_texts_to_batch(texts, tokenizer, block_size=16, mask_prompt=True)
        # Verify that all targets up to and including '=' are -100
        eq_id = tokenizer.token_to_id["="]
        eq_idx = (x[0] == eq_id).nonzero(as_tuple=True)[0].item()
        for i in range(eq_idx + 1):
            self.assertEqual(y[0][i].item(), -100)
        # The token immediately following '=' should NOT be -100
        self.assertNotEqual(y[0][eq_idx + 1].item(), -100)

    def test_create_model_factory(self):
        from src.models import create_model, GPTConfig, LoopedGPTConfig, HierarchicalGPTConfig
        from src.models.baseline import MathGPT
        from src.models.looped import LoopedMathGPT
        from src.models.hierarchical import HierarchicalReasoningGPT

        m1 = create_model(GPTConfig(vocab_size=32, block_size=16, n_layer=1, n_head=1, n_embd=16))
        self.assertIsInstance(m1, MathGPT)

        m2 = create_model(LoopedGPTConfig(vocab_size=32, block_size=16, n_embd=16, n_head=1, n_core_layers=1))
        self.assertIsInstance(m2, LoopedMathGPT)

        m3 = create_model(HierarchicalGPTConfig(vocab_size=32, block_size=16, n_embd=16, n_head=1, n_fast_steps=2, n_slow_cycles=2))
        self.assertIsInstance(m3, HierarchicalReasoningGPT)


if __name__ == "__main__":
    unittest.main()

