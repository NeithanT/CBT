"""
Condition 1 Training Runner: Character Math Tokenizer (100% Math with Scratchpad)
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from training.trainer import get_training_arg_parser, train_condition

def train(args=None):
    train_condition(condition=1, args=args)

def main():
    parser = get_training_arg_parser("Character Math Tokenizer (100% Math with Scratchpad)")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
