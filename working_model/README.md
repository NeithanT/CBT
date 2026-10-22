# Working Model: HRM-RoPE (Hierarchical Reasoning Model with Decoupled 2D-RoPE)

This directory contains the self-contained, verified implementation and benchmark artifacts for **HRM-RoPE**, the first model in this research project to achieve multi-digit arithmetic length extrapolation up to 20 digits under compute-optimal scaling.

---

## 1. Breakthrough Summary

- **Parameter Count**: **1,574,144** non-embedding parameters (strictly $< 2.0\text{M}$).
- **Overall Benchmark Accuracy**: **91.67% (11/12 test cases passing)** across 1 to 20 digits.
- **In-Distribution Accuracy (1 to 12 digits)**: **100.0% (6/6)**.
- **Extreme Extrapolation Accuracy (18 & 20 digits)**: **100.0% (3/3)**.
- **20-Digit Zero Counting & Cascade Carry**: **100.0% (2/2)**.
- **Prior Baseline Result**: 0.0% extrapolation on all 9 previous architectures (Standard Transformer, Looped Transformer, GLRU-SSM, ALSM, HNRM).

---

## 2. Theoretical Architecture

HRM-RoPE resolves length extrapolation failure through two core mechanisms:

### 2.1 Decoupled 2D-RoPE Manifold
Standard 1D RoPE rotates query and key vectors monotonically with sequence index $t$, causing phase drift at unseen lengths $T > 192$.

HRM-RoPE decomposes each attention head's dimension ($d_k = 64$) into two orthogonal 32-dimensional subspaces:
$$\tilde{q}_i^T \tilde{k}_j = (q_i^{\text{macro}})^T R_{\text{place}}(p_i - p_j) k_j^{\text{macro}} + (q_i^{\text{micro}})^T R_{\text{step}}(\tau_i - \tau_j) k_j^{\text{micro}}$$

1. **Macro Coordinate ($p \in \mathbb{N}$)**:
   - Indexes the decimal place rank ($10^p$) of the column.
   - For tokens participating in the same column: $\Delta p = p_i - p_j = 0 \implies R_{\text{place}}(0) = I$ (Identity rotation).
   - Guarantees zero phase drift regardless of whether the equation has 2 or 20 digits.
2. **Micro Coordinate ($\tau \in [0, 9]$)**:
   - Indexes the sequential position within each 10-token column template `(c+a+b)=c's`.
   - Strictly bounded ($\tau \le 9$), preventing intra-block bag-of-words ambiguity while never encountering out-of-distribution rotational frequencies.

### 2.2 Dual-Timescale Recurrence
- **FastReasoningBlock** (Inner ALU): Iterates $T=2$ fast operational steps per token to execute local digit addition $(c_{\text{in}} + a + b) \to (c_{\text{out}}, s)$ modulated via FiLM conditioning.
- **SlowPlanningBlock** (Outer Controller): Executes $K=2$ slow strategic cycles to maintain the discrete carry state $c \in \{0, 1\}$ and track column termination, eliminating the "Zero-Loop" attractor basin.

---

## 3. Directory Layout

```
working_model/
|-- model.py          # Stand-alone HRM-RoPE PyTorch implementation
|-- tokenizer.py      # Character-level MathTokenizer
|-- arithmetic.py     # Canonical column scratchpad formatter and extractor
|-- inference.py      # Stand-alone CLI for evaluation, interactive inference, and benchmark
|-- config.json       # Exact model hyperparameters
|-- RESULTS.md        # Itemized benchmark traces and latencies
`-- README.md         # This documentation
```

---

## 4. Usage

### 4.1 Running Inference on a Saved Checkpoint
```bash
# Run the full 1-20 digit extrapolation benchmark:
python inference.py --checkpoint checkpoints/hrm_rope_c1_final.pt --benchmark

# Evaluate a specific multi-digit addition:
python inference.py --checkpoint checkpoints/hrm_rope_c1_final.pt --prompt "12345678901234567890+87654321098765432109="

# Interactive CLI session:
python inference.py --checkpoint checkpoints/hrm_rope_c1_final.pt --interactive
```

### 4.2 Training HRM-RoPE from Scratch (Compute-Optimal 400M Tokens)
Using the repository training orchestrator on Dual NVIDIA T4 GPUs:
```bash
./run_phase3_research.sh hrm_rope --target-tokens 400000000 --batch-size 64
```
Checkpoints will be saved periodically to `checkpoints/hrm_rope/`.

---

## 5. Parameter Count Verification

| Layer / Parameter Group | Dimension | Parameters |
| :--- | :--- | :--- |
| **Token Embedding (wte)** | $32 \times 256$ | 8,192 (tied with LM head) |
| **FastReasoningBlock Attention** | $4 \times 256 \times 256$ projections | 262,144 |
| **FastReasoningBlock MLP (GELU)** | $2 \times 256 \times 768$ | 393,216 |
| **FastReasoningBlock FiLM & Norms** | Projection ($256 \to 512$) + RMSNorms | 131,584 |
| **SlowPlanningBlock Attention** | $4 \times 256 \times 256$ projections | 262,144 |
| **SlowPlanningBlock MLP (GELU)** | $2 \times 256 \times 768$ | 393,216 |
| **SlowPlanningBlock Gating & Norms** | Gate projection ($512 \to 256$) + RMSNorms | 131,584 |
| **Final RMSNorm (ln_f)** | 256 | 256 |
| **Non-Embedding Total** | | **1,574,144** |
| **Total Parameter Count** | | **1,582,336** |
| **Parameter Ceiling Rule** | Strictly $< 2,000,000$ | **[PASS] Validated** |
