# Champion Model: 100k Recurrent 2D-RoPE Architecture (Math & Tools)

This directory contains the stand-alone codebase and verified checkpoints for the **Champion 100k Recurrent 2D-RoPE Architecture**.

This unified architectural paradigm achieves state-of-the-art results across two foundational reasoning domains at 1/15th the parameter footprint of prior baselines:
1. **Arithmetic Length Extrapolation**: **100.0% (14/14)** accuracy across 1 to 20 digits (exceeding HRM-RoPE 1.57M baseline).
2. **Natural Language Tool Calling & Routing**: **100.0% (50/50)** accuracy on structured tool invocation (calculator, weather, search, time, unit conversion, no-tool conversation).

---

## 1. Architectural Blueprint

The architecture is identical across both domains, utilizing:
- **Parameter Ceiling**: ~100k parameters (98,944 non-embedding parameters for Math; 99,840 for Tools).
- **Physical Layers ($L$)**: 2 Transformer blocks.
- **Hidden Dimension ($d_{\text{model}}$)**: 64.
- **Attention Heads ($n_{\text{head}}$)**: 4 (head dimension $d_k = 16$).
- **Recurrent Depth ($R=3$)**: Weight-tied recurrent unrolling across 3 passes per forward step with residual scaling ($1/\sqrt{R}$), achieving an effective depth of 6 Transformer layers with 0 additional parameters.
- **Decoupled 2D-RoPE**: Subspace decomposition separating Macro column/discourse coordinates from Micro step coordinates, eliminating phase drift.
- **Tied Embeddings**: `wte.weight == lm_head.weight`.

---

## 2. Directory Layout

```
champion_model/
|-- __init__.py                # Package exports
|-- evaluate_all.py            # Unified evaluation runner
|-- README.md                  # Champion model suite documentation
|-- scaled_t4/                 # Character Based Transformer (CBT, ~18.95M Parameters)
|   |-- model.py               # CBT architecture (2D-RoPE, R=3 unrolling)
|   |-- tokenizer.py           # ScaledTokenizer with tool schema support
|   |-- dataset.py             # 1-50 digit curriculum & multi-tool dataset generator
|   |-- interactive.py         # Live interactive CLI with mock tool execution
|   |-- sft_search.py          # 1-hop induction copy SFT training script
|   `-- config.json            # Model architecture configuration
|-- math_model/                # Ultra-Compact 100k Math Extrapolation Model
|   |-- model.py               # MathGPT architecture (2D-RoPE, R=3 recurrent unrolling)
|   |-- train_and_eval.py      # Training loop and 1-20 digit benchmark suite
|   |-- arithmetic.py          # Column scratchpad formatting and extraction
|   `-- config_100k_2d.json    # Hyperparameter configuration
`-- tool_model/                # 100k Natural Language Tool Calling
    |-- model.py               # ToolGPT architecture (Hierarchical 2D Discourse RoPE)
    |-- tokenizer.py           # Structured discourse tokenizer
    |-- bpe_tokenizer.py       # Subword tokenizer ablation
    |-- dataset.py             # Synthetic tool dialogues generator
    `-- train_and_eval.py      # ToolGPT training and 50-case benchmark evaluation
```

---

## 3. Verified Benchmark Results

### 3.1 Arithmetic Length Extrapolation (1 to 20 Digits)
Evaluated on `dense_100k_2d_rec3_full.pt`:

| Tier | Test Cases | Result | Status |
| :--- | :---: | :---: | :---: |
| In-Distribution (1–12 Digits) | 6 | 6/6 | `[PASS]` |
| Moderate Extrapolation (14–16 Digits) | 3 | 3/3 | `[PASS]` |
| Extreme Extrapolation (18–20 Digits) | 3 | 3/3 | `[PASS]` |
| 20-Digit Zero Counting & Cascade Carry | 2 | 2/2 | `[PASS]` |
| **Total Benchmark** | **14** | **14/14 (100.0%)** | `[FLAWLESS]` |

### 3.2 ToolGPT Tool Routing (50 Test Cases)
Evaluated on `tool_gpt_100k.pt`:

| Tool Category | Evaluated Cases | Accuracy | Status |
| :--- | :---: | :---: | :---: |
| `calculator` | 6 | 6/6 (100.0%) | `[PASS]` |
| `weather` | 8 | 8/8 (100.0%) | `[PASS]` |
| `search` | 5 | 5/5 (100.0%) | `[PASS]` |
| `time` | 10 | 10/10 (100.0%) | `[PASS]` |
| `unit_convert` | 13 | 13/13 (100.0%) | `[PASS]` |
| `no_tool` (Direct Conversation) | 8 | 8/8 (100.0%) | `[PASS]` |
| **Total Benchmark** | **50** | **50/50 (100.0%)** | `[FLAWLESS]` |

---

## 4. Execution & Reproduction Commands

### 4.1 Run Unified Evaluation (Both Math and Tools)
```bash
.venv/bin/python champion_model/evaluate_all.py
```

### 4.2 Run Only Math Evaluation
```bash
.venv/bin/python champion_model/evaluate_all.py --mode math
```
Or directly via the arithmetic runner:
```bash
.venv/bin/python champion_model/math_model/train_and_eval.py \
  --pos-emb-type rope_2d \
  --n-recur 3 \
  --eval-only \
  --save-ckpt champion_model/math_model/checkpoints/dense_100k_2d_rec3_full.pt
```

### 4.3 Run Only Tool Calling Evaluation
```bash
.venv/bin/python champion_model/evaluate_all.py --mode tools
```
Or directly via the tool runner:
```bash
.venv/bin/python champion_model/tool_model/train_and_eval.py \
  --eval-only \
  --save-ckpt champion_model/tool_model/checkpoints/tool_gpt_100k.pt
```

---

## 5. Python API Usage

```python
import torch
from champion_model import MathGPT, ToolGPT, ToolTokenizer

# 1. Load Math Champion
math_ckpt = torch.load("champion_model/math_model/checkpoints/dense_100k_2d_rec3_full.pt", map_location="cpu", weights_only=False)
math_model = MathGPT(math_ckpt["config"])
math_model.load_state_dict(math_ckpt["model_state_dict"])
math_model.eval()

# 2. Load Tool Champion
tool_tok = ToolTokenizer()
tool_ckpt = torch.load("champion_model/tool_model/checkpoints/tool_gpt_100k.pt", map_location="cpu", weights_only=False)
tool_model = ToolGPT(tool_ckpt["config"], tokenizer=tool_tok)
tool_model.load_state_dict(tool_ckpt["model_state_dict"])
tool_model.eval()
```
