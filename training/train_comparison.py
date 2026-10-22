import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from training.trainer import get_training_arg_parser, train_condition

MODEL_MAP = {
    "model_a_char": 1,
    "model_b_bpe_math": 2,
    "model_c_bpe_philo_math": 3,
}

def main():
    parser = get_training_arg_parser("Comparative 3-Model Training")
    parser.add_argument(
        "--model_type",
        type=str,
        default="model_a_char",
        choices=list(MODEL_MAP.keys()),
    )
    args = parser.parse_args()
    cond = MODEL_MAP[args.model_type]
    train_condition(condition=cond, args=args)

if __name__ == "__main__":
    main()
