#!/usr/bin/env bash
# Distributed Cloud Research Orchestrator for 2x NVIDIA T4 GPUs
set -e

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PY_BIN="$VIRTUAL_ENV/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PY_BIN=".venv/bin/python"
else
    PY_BIN="python3"
fi

TORCHRUN_BIN="$(dirname "$PY_BIN")/torchrun"
if [ ! -f "$TORCHRUN_BIN" ]; then
    TORCHRUN_BIN="torchrun"
fi

print_header() {
    echo "================================================================================"
    echo "2x NVIDIA T4 RESEARCH ORCHESTRATOR"
    echo "Devices: $CUDA_VISIBLE_DEVICES | Python: $PY_BIN | Precision: FP16 AMP"
    echo "Scaling: Compute-Optimal 20:1 Ratio (400M tokens on ~20M parameters)"
    echo "================================================================================"
}

show_usage() {
    print_header
    echo "Usage: ./run_t4_research.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  verify              Run environment diagnostics and T4 benchmarks"
    echo "  model1 [opts]       Train Condition 1 (Char Math, 400M tokens, for however long needed)"
    echo "  model2 [opts]       Train Condition 2 (Subword BPE, 400M tokens, for however long needed)"
    echo "  model3 [opts]       Train Condition 3 (Subword BPE Philo-Math, 400M tokens)"
    echo "  model4 [opts]       Train Condition 4 (Place-Value Thinking with GRPO RL)"
    echo "  all [opts]          Train all 4 conditions sequentially (Models 1, 2, 3, 4)"
    echo "  benchmark [--all]   Run extrapolation benchmark across saved checkpoints"
    echo "  compare [type]      Run comparative training"
    echo "  test                Run unit test suite"
    echo "  interact            Launch interactive model inference CLI"
    echo ""
    echo "Examples:"
    echo "  ./run_t4_research.sh model1"
    echo "  ./run_t4_research.sh model1 --target_tokens 100000000"
    echo "  ./run_t4_research.sh all"
    echo "================================================================================"
}

CMD="${1:-help}"

case "$CMD" in
    verify)
        print_header
        "$PY_BIN" benchmarks/verify_environment.py
        ;;
    model1)
        print_header
        echo "Launching Condition 1 training (400M token budget)..."
        "$TORCHRUN_BIN" --nproc_per_node=2 training/train_model1_char.py "${@:2}"
        ;;
    model2)
        print_header
        echo "Launching Condition 2 training (400M token budget)..."
        "$TORCHRUN_BIN" --nproc_per_node=2 training/train_model2_bpe.py "${@:2}"
        ;;
    model3)
        print_header
        echo "Launching Condition 3 training (400M token budget)..."
        "$TORCHRUN_BIN" --nproc_per_node=2 training/train_model3_philo.py "${@:2}"
        ;;
    model4)
        print_header
        echo "Launching Condition 4 training (Place-Value Thinking with GRPO RL)..."
        "$TORCHRUN_BIN" --nproc_per_node=2 training/train_model4_rl.py "${@:2}"
        ;;
    all|all4)
        print_header
        echo "Starting 4-model study on 2x NVIDIA T4..."
        echo "[1/4] Condition 1: Character Math Tokenizer"
        "$TORCHRUN_BIN" --nproc_per_node=2 training/train_model1_char.py "${@:2}"
        echo "[2/4] Condition 2: Subword BPE Tokenizer"
        "$TORCHRUN_BIN" --nproc_per_node=2 training/train_model2_bpe.py "${@:2}"
        echo "[3/4] Condition 3: Subword BPE Philo-Math"
        "$TORCHRUN_BIN" --nproc_per_node=2 training/train_model3_philo.py "${@:2}"
        echo "[4/4] Condition 4: Place-Value Zero-Reduction Thinking with GRPO RL"
        "$TORCHRUN_BIN" --nproc_per_node=2 training/train_model4_rl.py "${@:2}"
        echo "All 4 conditions trained and evaluated."
        ;;
    benchmark)
        print_header
        "$PY_BIN" benchmarks/evaluate_checkpoints.py "${@:2}"
        ;;
    compare)
        MODEL_TYPE="${2:-model_a_char}"
        print_header
        "$PY_BIN" training/train_comparison.py --model_type "$MODEL_TYPE"
        ;;
    test)
        print_header
        "$PY_BIN" -m unittest tests/test_math_gpt.py
        ;;
    interact)
        print_header
        "$PY_BIN" benchmarks/interact.py "${@:2}"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $CMD"
        show_usage
        exit 1
        ;;
esac
