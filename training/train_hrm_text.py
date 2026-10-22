"""
Phase 3 Training Runner: Compressed HRM-Text Architecture (Sapient Inc).
Paper: "HRM-Text: Efficient Pretraining Beyond Scaling" (arXiv:2605.20613).
Compressed to strictly < 2.0M non-embedding parameters (~1.44M).
Compliant with Dual NVIDIA T4 (Turing sm_75, FP16 AMP) via Cutlass SDPA.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from training.trainer import get_training_arg_parser, train_condition


def main():
    parser = get_training_arg_parser(
        "Phase 3: Compressed HRM-Text (Sapient Inc Hierarchical Foundation Architecture)"
    )
    parser.set_defaults(
        model_type="hrm_text",
        hidden_size=256,
        n_embd=256,
        num_heads=4,
        n_head=4,
        expansion=3.0,
        h_cycles=2,
        l_cycles=3,
        n_layers=2,
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
    parser.add_argument(
        "--h-cycles",
        type=int,
        default=2,
        help="Outer slow macro cycles (default: 2)",
    )
    parser.add_argument(
        "--l-cycles",
        type=int,
        default=3,
        help="Inner fast micro cycles (default: 3)",
    )
    parser.add_argument(
        "--expansion",
        type=float,
        default=3.0,
        help="SwiGLU intermediate expansion ratio (default: 3.0)",
    )
    args = parser.parse_args()
    train_condition(condition=args.condition, args=args)


if __name__ == "__main__":
    main()
