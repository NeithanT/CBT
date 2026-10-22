"""
Phase 2 Training Runner: Hierarchical Reasoning Model (Dual-Loop HRM)
Trains HierarchicalReasoningGPT with coupled Slow (H) and Fast (L) loops.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from training.trainer import get_training_arg_parser, train_condition


def main():
    parser = get_training_arg_parser("Phase 2: Hierarchical Reasoning Model (Dual-Loop HRM)")
    parser.set_defaults(model_type="hierarchical", n_fast_steps=4, n_slow_cycles=3)
    parser.add_argument("--condition", type=int, default=1, choices=[1, 2, 3, 4], help="Data condition (default: 1 - Character Math)")
    args = parser.parse_args()
    train_condition(condition=args.condition, args=args)


if __name__ == "__main__":
    main()
