"""
Condition 3 Training Runner: Subword BPE (50% Math Word-CoT + 50% Philosophy Text)
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from training.trainer import get_training_arg_parser, train_condition

def train(args=None):
    train_condition(condition=3, args=args)

def main():
    parser = get_training_arg_parser("Subword BPE (50% Math Word-CoT + 50% Philosophy Text)")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
