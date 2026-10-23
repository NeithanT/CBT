#!/usr/bin/env bash
# ==============================================================================
# ScaledGPT (~18.95M Parameters) Unified Training & SFT Orchestrator
# Supports Dual NVIDIA T4 sm_75 FP16 AMP, Checkpoint Resumption, and SFT
# ==============================================================================
set -e

export PYTHONUNBUFFERED=1
export PYDEVD_DISABLE_FILE_VALIDATION=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Determine Python binary
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PY_BIN="$VIRTUAL_ENV/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PY_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
else
    PY_BIN="python"
fi

print_header() {
    echo "================================================================================"
    echo "SCALEDGPT (~18.95M) TRAINING ORCHESTRATOR"
    echo "Python: $PY_BIN | sm_75 FP16 AMP | Max Retained Checkpoints: 2"
    echo "================================================================================"
}

# Determine base directory structure (root vs inside scaled_t4)
if [ -f "sft_search.py" ]; then
    BASE_DIR="."
    DEFAULT_CKPT_DIR="checkpoints"
elif [ -f "champion_model/scaled_t4/sft_search.py" ]; then
    BASE_DIR="champion_model/scaled_t4"
    DEFAULT_CKPT_DIR="champion_model/scaled_t4/checkpoints"
else
    BASE_DIR="."
    DEFAULT_CKPT_DIR="checkpoints"
fi

# Resolve checkpoint path from argument $1 or search common paths
CKPT_INPUT="${1:-}"
RESOLVED_CKPT=""

if [ -n "$CKPT_INPUT" ] && [ -f "$CKPT_INPUT" ]; then
    RESOLVED_CKPT="$CKPT_INPUT"
else
    CANDIDATES=(
        "$CKPT_INPUT"
        "archive/checkpoints/champion_base.pt"
        "archive/checkpoints/champion_base.pt"
        "$DEFAULT_CKPT_DIR/scaled_t4_sft_all_tools.pt"
        "$DEFAULT_CKPT_DIR/scaled_t4_sft_search.pt"
        "checkpoints/scaled_t4_sft_search.pt"
        "champion_model/checkpoints/scaled_t4_sft_search_step_7000.zip"
        "/kaggle/working/checkpoints/scaled_t4/scaled_t4_sft_search.pt"
        "$DEFAULT_CKPT_DIR/scaled_t4_final.pt"
        "checkpoints/scaled_t4_final.pt"
        "/kaggle/working/checkpoints/scaled_t4/scaled_t4_final.pt"
        "$DEFAULT_CKPT_DIR/scaled_t4_final_timeout_step_45059.pt"
        "/kaggle/working/checkpoints/scaled_t4/scaled_t4_final_timeout_step_45059.pt"
        "champion_model/checkpoints/scaled_t4_final.zip"
        "champion_model/checkpoints/scaled_t4_final_timeout_step_45059.zip"
    )
    for c in "${CANDIDATES[@]}"; do
        if [ -n "$c" ] && [ -f "$c" ]; then
            RESOLVED_CKPT="$c"
            break
        fi
    done
fi

STEPS="${2:-12000}"
MODE="${3:-sft}"

show_help() {
    print_header
    echo "Usage: ./train.sh [CHECKPOINT_PATH] [STEPS] [MODE]"
    echo ""
    echo "Arguments:"
    echo "  CHECKPOINT_PATH  Path to base/resume checkpoint (.pt or .zip)"
    echo "  STEPS            Number of training/SFT steps (Default: 12000)"
    echo "  MODE             Operation mode: 'sft', 'pretrain', 'eval', 'demo' (Default: sft)"
    echo ""
    echo "Examples:"
    echo "  # 1. Run Universal Robust Annealing SFT from champion_base.pt (Default: 12000 steps)"
    echo "  ./train.sh archive/checkpoints/champion_base.pt"
    echo ""
    echo "  # 2. Run All-Tools Annealing SFT with custom steps"
    echo "  ./train.sh archive/checkpoints/champion_base.pt 12000 sft"
    echo ""
    echo "  # 2b. Curriculum V5 SFT from champion_final.pt -> champion_v3_final.pt (6.0 h SFT inside a 7 h session)"
    echo "  ./train.sh $DEFAULT_CKPT_DIR/champion_final.pt 200000 sft_v3"
    echo ""
    echo "  # 3. Resume full 10-hour pretraining (e.g., Session 2 up to 100k steps)"
    echo "  ./train.sh /kaggle/working/checkpoints/scaled_t4/scaled_t4_final_timeout_step_45059.pt 100000 pretrain"
    echo ""
    echo "  # 4. Run evaluation benchmark on checkpoint"
    echo "  ./train.sh $DEFAULT_CKPT_DIR/champion_final.pt 0 eval"
    echo ""
    echo "  # 5. Launch interactive terminal"
    echo "  ./train.sh $DEFAULT_CKPT_DIR/champion_final.pt 0 demo"
    echo "================================================================================"
}

if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ "$1" = "help" ]; then
    show_help
    exit 0
fi

print_header

if [ -n "$RESOLVED_CKPT" ]; then
    echo "[INFO] Using Checkpoint: $RESOLVED_CKPT"
else
    echo "[WARN] No existing checkpoint found. Starting from scratch."
fi

case "$MODE" in
    sft)
        echo "[INFO] Mode: Universal Robust All-Tools & Search Annealing SFT ($STEPS steps)"
        echo "[INFO] Curriculum V4: 45% Multi-Tool + 35% Search Copy + 20% Math Replay (1-25 Digits, k=45)"
        SAVE_CKPT="$DEFAULT_CKPT_DIR/champion_final.pt"
        
        EXTRA_ARGS=()
        if [ -n "$RESOLVED_CKPT" ]; then
            EXTRA_ARGS+=(--base-ckpt "$RESOLVED_CKPT")
        fi
        
        "$PY_BIN" "$BASE_DIR/sft_search.py" \
            --steps "$STEPS" \
            --batch-size 32 \
            --lr 5.0e-5 \
            --k-rand-shift 45 \
            --save-ckpt "$SAVE_CKPT" \
            --max-keep-ckpts 2 \
            "${EXTRA_ARGS[@]}" \
            "${@:4}"
        ;;

    sft_v3)
        # Curriculum V5 from the champion: time-bounded cosine (anneals fully within --max-hours)
        echo "[INFO] Mode: Curriculum V5 SFT -> champion_v3 (<= $STEPS steps, 6.0 h wall clock)"
        V3_BASE="${RESOLVED_CKPT}"
        if [ -z "$CKPT_INPUT" ] || [ ! -f "$CKPT_INPUT" ]; then
            V3_BASE="$DEFAULT_CKPT_DIR/champion_final.pt"
        fi
        "$PY_BIN" "$BASE_DIR/sft_search.py" \
            --curriculum v5 \
            --base-ckpt "$V3_BASE" \
            --steps "$STEPS" \
            --batch-size 32 \
            --lr 5.0e-5 \
            --k-rand-shift 45 \
            --max-hours 6.0 \
            --save-interval 2000 \
            --probe-interval 2000 \
            --save-ckpt "$DEFAULT_CKPT_DIR/champion_v3_final.pt" \
            --max-keep-ckpts 2 \
            "${@:4}"
        ;;

    pretrain|continue)
        echo "[INFO] Mode: Frontier Pretraining / Extension ($STEPS steps)"
        SAVE_CKPT="$DEFAULT_CKPT_DIR/scaled_t4_final.pt"
        
        EXTRA_ARGS=()
        if [ -n "$RESOLVED_CKPT" ]; then
            EXTRA_ARGS+=(--resume "$RESOLVED_CKPT")
        fi
        
        "$PY_BIN" "$BASE_DIR/train_and_eval.py" \
            --steps "$STEPS" \
            --anneal-steps 15000 \
            --batch-size 32 \
            --lr 5.0e-4 \
            --save-ckpt "$SAVE_CKPT" \
            --max-hours 10.0 \
            --max-keep-ckpts 2 \
            "${EXTRA_ARGS[@]}" \
            "${@:4}"
        ;;

    eval)
        echo "[INFO] Mode: Benchmark Evaluation"
        if [ -z "$RESOLVED_CKPT" ]; then
            echo "[FAIL] Evaluation requires a valid checkpoint path."
            exit 1
        fi
        "$PY_BIN" "$BASE_DIR/train_and_eval.py" \
            --eval-only \
            --resume "$RESOLVED_CKPT"
        ;;

    demo|interactive)
        echo "[INFO] Mode: Interactive Terminal"
        EXTRA_ARGS=()
        if [ -n "$RESOLVED_CKPT" ]; then
            EXTRA_ARGS+=(--ckpt "$RESOLVED_CKPT")
        fi
        "$PY_BIN" "$BASE_DIR/interactive.py" "${EXTRA_ARGS[@]}"
        ;;

    *)
        echo "[WARN] Unknown mode: $MODE. Defaulting to SFT."
        "$PY_BIN" "$BASE_DIR/sft_search.py" \
            --steps "$STEPS" \
            --base-ckpt "$RESOLVED_CKPT" \
            --max-keep-ckpts 2
        ;;
esac
