#!/usr/bin/env bash
# Phase 3 Distributed Research Orchestrator for 2x NVIDIA T4 GPUs
# Architecture: MicroMath-2M (NoPE + Abacus Decimal Place Embeddings)
set -e

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

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

GPU_COUNT=$("$PY_BIN" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -ge 2 ]; then
    DIST_CMD=("$TORCHRUN_BIN" "--nproc_per_node=2")
elif [ "$GPU_COUNT" -eq 1 ]; then
    DIST_CMD=("$TORCHRUN_BIN" "--nproc_per_node=1")
else
    DIST_CMD=("$PY_BIN")
fi

sync_kaggle_checkpoints() {
    if [ -d "/kaggle/working" ]; then
        mkdir -p "/kaggle/working/checkpoints"
        if [ -d "checkpoints" ]; then
            cp -r checkpoints/* "/kaggle/working/checkpoints/" 2>/dev/null || true
            echo "[INFO] Synced all checkpoints to /kaggle/working/checkpoints/"
        fi
    fi
}
trap sync_kaggle_checkpoints EXIT

print_header() {
    echo "================================================================================"
    echo "PHASE 3 RESEARCH ORCHESTRATOR: MICROMATH-2M LENGTH GENERALIZATION"
    echo "Hardware: 2x NVIDIA T4 (sm_75) | Precision: FP16 AMP | Python: $PY_BIN"
    echo "Active GPUs Detected: $GPU_COUNT"
    echo "================================================================================"
}

show_usage() {
    print_header
    echo "Usage: ./run_phase3_research.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train       Train MicroMath-2M on Condition 1 with compute-optimal tokens"
    echo "  evaluate    Run Extrapolation Benchmark on MicroMath-2M checkpoint"
    echo "  verify      Verify T4 runtime and dependencies"
    echo "  test        Run unit tests"
    echo ""
    echo "Examples:"
    echo "  ./run_phase3_research.sh train --target-tokens 40000000"
    echo "  ./run_phase3_research.sh evaluate --checkpoint checkpoints/micromath/micromath_c1_final.pt"
    echo "  ./run_phase3_research.sh test"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
    train|micromath)
        print_header
        echo "[RUN] Training Phase 3: MicroMath-2M..."
        "${DIST_CMD[@]}" training/train_phase3_micromath.py "$@"
        ;;
    hrm_rope|hrm)
        print_header
        echo "[RUN] Training Phase 3: HRM-RoPE..."
        "${DIST_CMD[@]}" training/train_phase3_hrm_rope.py "$@"
        ;;
    hrm_text)
        print_header
        echo "[RUN] Training Phase 3: Compressed HRM-Text (Sapient Inc)..."
        "${DIST_CMD[@]}" training/train_hrm_text.py "$@"
        ;;
    all)
        print_header
        echo "[RUN] Training both Phase 3 architectures sequentially..."
        echo ">>> 1/2: MicroMath-2M"
        "${DIST_CMD[@]}" training/train_phase3_micromath.py "$@"
        echo ">>> 2/2: HRM-RoPE"
        "${DIST_CMD[@]}" training/train_phase3_hrm_rope.py "$@"
        ;;
    evaluate)
        print_header
        echo "[RUN] Evaluating MicroMath-2M..."
        "$PY_BIN" benchmarks/evaluate_checkpoints.py "$@"
        ;;
    verify)
        print_header
        "$PY_BIN" benchmarks/verify_environment.py
        ;;
    test)
        print_header
        echo "[RUN] Executing unit test suite..."
        "$PY_BIN" -m unittest discover -s tests
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "[ERROR] Unknown command: $cmd"
        show_usage
        exit 1
        ;;
esac
