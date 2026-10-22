# Character Based Transformer (CBT, ~18.95M Parameters): Recurrent 2D-RoPE Architecture for Dual NVIDIA T4

## Overview

Character Based Transformer (CBT) scales the champion architecture to a robust parameter capacity (**~18.95M parameters**), strictly optimized for Dual NVIDIA T4 GPUs.

It unifies:
1. **Multi-Digit Column Arithmetic Length Extrapolation** from 1 to 50+ digits with 100% precision.
2. **Expanded Tool Calling Repertoire**: Single-tool calls, sequential multi-tool chaining, nested JSON schemas, error handling & recovery, multi-turn dialogues, and direct conversational routing.
3. **Dual NVIDIA T4 Hardware Efficiency**: Native Turing sm_75 FP16 AMP, O(1) SDPA causal attention, and PyTorch DistributedDataParallel (DDP).

---

## Architectural Specifications

| Parameter | Specification | Note |
| :--- | :--- | :--- |
| **Physical Layers ($L$)** | 6 | Physical transformer blocks |
| **Recurrent Passes ($R$)** | 3 | Effective depth = 18 layers at 0 parameter overhead |
| **Hidden Dimension ($d_{\text{model}}$)** | 512 | Wide channel capacity |
| **Attention Heads ($H$)** | 8 | Head dimension = 64 (ideal for sm_75 Tensor Cores) |
| **Max Sequence Length ($T$)** | 1024 | Accommodates 50-digit problems and multi-turn traces |
| **Max 2D-RoPE Coordinates** | 4096 | Precomputed frequency buffer with dynamic expansion |
| **Vocab Size** | 111 | Special discourse tags + ASCII characters |
| **Recurrence Residual Update** | $x = x + \frac{1}{\sqrt{R}} (x_{\text{res}} - x)$ | Dynamic stability scaling across passes |
| **Shift Augmentation** | $(p > 0) \cdot \Delta$ | Selective shift masking during training |
| **Weight Tying** | $W_{\text{te}} = W_{\text{lm\_head}}$ | Parameter efficiency & tied representation |
| **Total Parameters** | **18,944,512 (~18.95M)** | Target ~10M to ~20M parameter scale |
| **Non-Embedding Parameters** | **18,887,680 (~18.89M)** | Core transformer weights |

---

## Dual NVIDIA T4 Optimization Architecture

Turing sm_75 GPUs have specific hardware constraints that require strict configuration:

1. **Native FP16 AMP with GradScaler**:
   - Training uses `torch.amp.autocast('cuda', dtype=torch.float16)` combined with `torch.amp.GradScaler('cuda')`.
   - **No bfloat16**: Turing sm_75 lacks hardware bfloat16 support. Emulation is slow and unstable.
   - **No TF32**: TF32 is an Ampere sm_80+ feature. It is explicitly disabled via:
     ```python
     torch.backends.cuda.matmul.allow_tf32 = False
     torch.backends.cudnn.allow_tf32 = False
     ```
2. **Scaled Dot-Product Attention (SDPA)**:
   - Uses PyTorch native `F.scaled_dot_product_attention`.
   - In FP16 mode on sm_75, PyTorch engages Cutlass/Flash memory-efficient attention, keeping activation memory at $O(1)$ rather than $O(T^2)$.
3. **DistributedDataParallel (DDP) for 2x T4 GPUs**:
   - Launches via `torchrun --nproc_per_node=2 champion_model/scaled_t4/train_and_eval.py`.
   - Synchronizes gradients across GPUs via NCCL backend with non-overlapping data sharding.
   - Rank 0 handles all logging, benchmark evaluations, and checkpoint synchronization.
4. **Kaggle Checkpoint Mirroring**:
   - Automatically mirrors periodic checkpoints (`_step_{step}.pt` and `_final.pt`) to `/kaggle/working/checkpoints/scaled_t4/` for automatic persistence across Kaggle sessions.

---

## Expanded Capabilities

### 1. High-Digit Arithmetic Extrapolation (1 to 50 Digits)
- Standard in-distribution curriculum: 1 to 15 digits.
- Evaluated on 20 test cases spanning 4 difficulty tiers:
  - **Tier 1: In-Distribution (1-15 Digits)**: 6 test cases.
  - **Tier 2: Intermediate Extrapolation (16-25 Digits)**: 6 test cases.
  - **Tier 3: High Extrapolation (30-40 Digits)**: 4 test cases.
  - **Tier 4: Ultra Extrapolation (45-50 Digits & 50-Digit Cascade Carry)**: 4 test cases.

### 2. Expanded Tool Repertoire (6 Categories)
- **Single-Tool Calling**: `calculator`, `get_weather`, `search`, `get_time`, `unit_convert`.
- **Multi-Tool Sequential Chaining**: Search fact $\to$ Calculator metric $\to$ Synthesized answer; Weather comparison across cities; Timezone difference calculation.
- **Nested JSON Schemas**: Tools with structured nested dictionaries, arrays, and flags (`database_query`, `create_event`, `file_manager`, `fetch_api`).
- **Error Handling & Recovery**: Model reasons over error payloads (404, 503, ZeroDivisionError, TableNotFound) in `<THINK>` and provides graceful user responses or fallback queries.
- **Multi-Turn Dialogues**: Multi-turn conversations maintaining state and context across multiple `<USER>...</USER>` exchanges.
- **Direct Conversational Routing (No-Tool)**: Direct responses for greetings, conceptual questions, and safe refusal of invalid inputs.

### 3. Strict Loss Masking Policy
- **Math**: User prompt up to `<THINK>` is masked with target `-100`. Loss is computed strictly on scratchpad steps `(c+da+db)=...` and final answer.
- **Tools**: User prompts `<USER>...</USER>` and environment responses `<RESULT>...</RESULT>` are masked with `-100`. The model is supervised strictly on generating `<THINK>`, `<CALL>`, and `<ANSWER>`.

---

## Directory Structure

```
champion_model/scaled_t4/
├── __init__.py            # Package entrypoint
├── config.json            # Hyperparameter specification
├── model.py               # ScaledGPT architecture (~18.95M params, 2D-RoPE, R=3)
├── tokenizer.py           # ScaledTokenizer with discourse markers & ASCII tokens
├── dataset.py             # High-digit curriculum & expanded tool synthetic generator
├── train_and_eval.py      # DDP orchestrator, FP16 AMP trainer, benchmark evaluator
├── interactive.py         # Interactive CLI with mock tool execution
└── README.md              # Technical documentation
```

---

## Quickstart Commands

### 1. Run Unit Tests
```bash
python -m unittest tests/test_scaled_t4.py
```

### 2. Run Comprehensive Verification
```bash
python benchmarks/verify_scaled_t4.py
```

### 3. Launch 25-Hour Dual T4 Training (DDP)
```bash
torchrun --nproc_per_node=2 champion_model/scaled_t4/train_and_eval.py \
    --steps 50000 \
    --anneal-steps 15000 \
    --batch-size 32 \
    --lr 1.0e-3 \
    --max-hours 24.5 \
    --save-ckpt champion_model/scaled_t4/checkpoints/scaled_t4_final.pt
```

### 4. Evaluate Benchmarks
```bash
python champion_model/scaled_t4/train_and_eval.py \
    --eval-only \
    --save-ckpt champion_model/scaled_t4/checkpoints/scaled_t4_final.pt
```

### 5. Interactive Inference CLI
```bash
python champion_model/scaled_t4/interactive.py \
    --ckpt champion_model/scaled_t4/checkpoints/scaled_t4_final.pt
```

### 6. Universal Search Delegation SFT (Curriculum V2)
```bash
# Run 6,000 steps with 81.25% Search Copy Induction + <=10% Math Carry Replay
./train.sh champion_model/scaled_t4/checkpoints/scaled_t4_final.pt 6000 sft
```

For complete empirical benchmarks, training curves, and mechanistic autopsies, see [PHASE_4_EMPIRICAL_REPORT.md](../../PHASE_4_EMPIRICAL_REPORT.md).

---

## Zero Emojis Policy
All code, docstrings, console logs, and outputs strictly conform to standard ASCII status tags: `[OK]`, `[PASS]`, `[FAIL]`, `[WARN]`, `[INFO]`.
