"""
Phase 3 Training Runner: MicroMath-2M Architecture.
Trains compact ~1.36M parameter Transformer with NoPE, Abacus Decimal Place Embeddings,
and random shift jitter for 100% arithmetic length generalization.
Compliant with Dual NVIDIA T4 (Turing sm_75, FP16 AMP) and strictly < 2M parameters.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from training.trainer import get_training_arg_parser, train_condition


def main():
    parser = get_training_arg_parser(
        "Phase 3: MicroMath-2M (NoPE + Abacus Decimal Place Embeddings)"
    )
    parser.set_defaults(
        model_type="micromath",
        n_embd=192,
        n_head=4,
        n_layer=3,
        d_ffn=768,
        max_places=128,
        k_rand_shift=32,
        dropout=0.0,
        lr=5e-4,
        target_tokens=40_000_000,
        checkpoint_interval_tokens=10_000_000,
        batch_size=64,
        grad_accum_steps=1,
    )
    parser.add_argument(
        "--condition",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Data condition (default: 1 - Character Math with Scratchpad)",
    )
    args = parser.parse_args()
    train_condition(condition=args.condition, args=args)


if __name__ == "__main__":
    main()
