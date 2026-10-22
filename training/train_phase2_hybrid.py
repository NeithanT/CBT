"""
Phase 2 Training Runner: Hybrid SSM-Transformer (GLRU-GPT).
Trains HybridMathGPT with interleaved Gated Linear Recurrent Units and Causal Self-Attention.
Compliant with Dual NVIDIA T4 (Turing sm_75, FP16 AMP) and strictly < 20M non-embedding parameters.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from training.trainer import get_training_arg_parser, train_condition


def main():
    parser = get_training_arg_parser("Phase 2: Hybrid SSM-Transformer (GLRU-GPT)")
    parser.set_defaults(model_type="hybrid", n_layer=6, pattern="interleaved")
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
