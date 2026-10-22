#!/usr/bin/env bash
# Phase 2 Distributed Research Orchestrator for 2x NVIDIA T4 GPUs
# Supports:
# 1. Harmonic Neural Register Machine (H-NRM: HPF + Bistable ALU)
# 2. Hybrid SSM-Transformer (GLRU-GPT: Parallel Scan SSM + Attention)
# 3. Abacus-Anchored Looped State Machine (ALSM-GPT: Relative Positional Anchoring)
# 4. Looped Transformer (Recurrent Depth / Astra Latent Compute)
# 5. Hierarchical Reasoning Model (Dual-Loop HRM)
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

# Detect available CUDA GPUs
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
    echo "PHASE 2 RESEARCH ORCHESTRATOR: BEYOND PURE SELF-ATTENTION"
    echo "Hardware: 2x NVIDIA T4 (sm_75) | Precision: FP16 AMP | Python: $PY_BIN"
    echo "Active GPUs Detected: $GPU_COUNT"
    echo "================================================================================"
}

show_usage() {
    print_header
    echo "Usage: ./run_phase2_research.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  all [opts]          Train all 5 Phase 2 models sequentially (H-NRM, Hybrid, ALSM, Looped, HRM)"
    echo "  novel [opts]        Train the 3 newly invented architectures sequentially (H-NRM, Hybrid, ALSM)"
    echo "  hnrm [opts]         Train Harmonic Neural Register Machine (H-NRM: HPF + Bistable ALU)"
    echo "  hybrid [opts]       Train Hybrid SSM-Transformer (GLRU-GPT: Parallel Scan SSM + Attention)"
    echo "  alsm [opts]         Train Abacus-Anchored Looped State Machine (ALSM-GPT)"
    echo "  looped [opts]       Train Looped Transformer (Recurrent Depth / Astra Latent Compute)"
    echo "  hrm [opts]          Train Hierarchical Reasoning Model (Dual-Loop HRM)"
    echo "  test                Run unit test suite for Phase 2 architectures"
    echo "  benchmark [opts]    Run extrapolation benchmark across Phase 2 checkpoints"
    echo ""
    echo "Options (passed directly to training scripts):"
    echo "  --target_tokens N   Target token budget (default: 400M tokens)"
    echo "  --batch_size N      Batch size per GPU (default: 64)"
    echo "  --lr FLOAT          Learning rate (default: 3e-4)"
    echo "  --log_interval N    Step logging interval (default: 50, concise single-line)"
    echo "  --condition N       Data condition: 1 (char+think), 2 (char), 3 (word+think), 4 (word)"
    echo "  --train_loops N     Training unrolled loops for Looped/ALSM Transformer (default: 6)"
    echo "  --eval_loops N      Inference unrolled loops for Looped/ALSM Transformer (default: 12)"
    echo "  --pattern STR       Mixer pattern for Hybrid model (interleaved, pure_ssm, pure_attn)"
    echo "  --base_freq FLOAT   Base frequency for HPF oscillators in H-NRM (default: 1.0)"
    echo "  --n_fast_steps N    Inner fast loop steps for HRM (default: 4)"
    echo "  --n_slow_cycles N   Outer slow loop cycles for HRM (default: 3)"
    echo "  --force_cpu         Force CPU execution for debugging"
    echo ""
    echo "Examples:"
    echo "  ./run_phase2_research.sh novel --target_tokens 400000000"
    echo "  ./run_phase2_research.sh hnrm"
    echo "  ./run_phase2_research.sh hybrid --pattern interleaved"
    echo "  ./run_phase2_research.sh alsm"
    echo "  ./run_phase2_research.sh all --target_tokens 100000000"
    echo "  ./run_phase2_research.sh test"
    echo "================================================================================"
}

CMD="${1:-help}"

case "$CMD" in
    hnrm)
        print_header
        echo "[INFO] Launching Phase 2 Harmonic Neural Register Machine (H-NRM) training..."
        "${DIST_CMD[@]}" training/train_phase2_hnrm.py "${@:2}"
        ;;
    hybrid)
        print_header
        echo "[INFO] Launching Phase 2 Hybrid SSM-Transformer training..."
        "${DIST_CMD[@]}" training/train_phase2_hybrid.py "${@:2}"
        ;;
    alsm)
        print_header
        echo "[INFO] Launching Phase 2 Abacus-Anchored Looped State Machine (ALSM-GPT) training..."
        "${DIST_CMD[@]}" training/train_phase2_alsm.py "${@:2}"
        ;;
    looped)
        print_header
        echo "[INFO] Launching Phase 2 Looped Transformer training..."
        "${DIST_CMD[@]}" training/train_phase2_looped.py "${@:2}"
        ;;
    hrm)
        print_header
        echo "[INFO] Launching Phase 2 Hierarchical Reasoning Model (HRM) training..."
        "${DIST_CMD[@]}" training/train_phase2_hrm.py "${@:2}"
        ;;
    novel)
        print_header
        echo "[INFO] Starting Phase 2 novel architectures sequential training suite..."
        echo "[1/3] Harmonic Neural Register Machine (H-NRM: HPF + Bistable ALU)"
        "${DIST_CMD[@]}" training/train_phase2_hnrm.py "${@:2}"
        echo "[2/3] Hybrid SSM-Transformer (GLRU-GPT: Parallel Scan SSM + Attention)"
        "${DIST_CMD[@]}" training/train_phase2_hybrid.py "${@:2}"
        echo "[3/3] Abacus-Anchored Looped State Machine (ALSM-GPT)"
        "${DIST_CMD[@]}" training/train_phase2_alsm.py "${@:2}"
        echo "[OK] Phase 2 novel architecture training runs complete."
        ;;
    all)
        print_header
        echo "[INFO] Starting Phase 2 full sequential training suite across all models..."
        echo "[1/5] Harmonic Neural Register Machine (H-NRM: HPF + Bistable ALU)"
        "${DIST_CMD[@]}" training/train_phase2_hnrm.py "${@:2}"
        echo "[2/5] Hybrid SSM-Transformer (GLRU-GPT: Parallel Scan SSM + Attention)"
        "${DIST_CMD[@]}" training/train_phase2_hybrid.py "${@:2}"
        echo "[3/5] Abacus-Anchored Looped State Machine (ALSM-GPT)"
        "${DIST_CMD[@]}" training/train_phase2_alsm.py "${@:2}"
        echo "[4/5] Looped Transformer (Recurrent Depth / Astra Latent Compute)"
        "${DIST_CMD[@]}" training/train_phase2_looped.py "${@:2}"
        echo "[5/5] Hierarchical Reasoning Model (Dual-Loop HRM)"
        "${DIST_CMD[@]}" training/train_phase2_hrm.py "${@:2}"
        echo "[OK] All Phase 2 model training runs complete."
        ;;
    test)
        print_header
        echo "[INFO] Running Phase 2 unit tests..."
        "$PY_BIN" -m unittest discover -s tests -v
        ;;
    benchmark)
        print_header
        echo "[INFO] Running extrapolation evaluation..."
        "$PY_BIN" benchmarks/evaluate_checkpoints.py "${@:2}"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "[ERROR] Unknown command: $CMD"
        show_usage
        exit 1
        ;;
esac
