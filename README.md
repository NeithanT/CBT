# 2D Embeddings for Length Generalization in Character-Based Transformers (CBT)

[![Paper](https://img.shields.io/badge/Paper-RESEARCH__PAPER.md-blue)](RESEARCH_PAPER.md)
[![Status](https://img.shields.io/badge/Status-Reproducible%20%5BPASS%5D-brightgreen)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Neithanvarvas%2FCBT-orange)](https://huggingface.co/Neithanvarvas/CBT)

**Primary Scientific Reference**: [RESEARCH_PAPER.md](RESEARCH_PAPER.md) | [RESEARCH_PAPER.pdf](RESEARCH_PAPER.pdf)  
**Target Hardware**: 2x NVIDIA T4 GPUs (Turing `sm_75`, 16 GB GDDR6 each, FP16 AMP)  
**Evaluated Scope**: 1 to 100 Digit Arithmetic Extrapolation (Structured & Random Multi-Carry) & Multi-Tool Agent Routing

---

## 1. Executive Summary & Findings

We study whether a small autoregressive character-level language model can extrapolate column-aligned multi-digit integer addition beyond the digit lengths seen in training when provided with rule-assigned two-dimensional (2D) coordinates $(p, \tau)$.

Small autoregressive Transformers trained from scratch on 1–12 digit addition with 1D Rotary Position Embeddings (1D-RoPE) score **0/30 at every unseen length** from 14 to 35 digits (and fail at 12 digits, the edge of their training range, with a character vocabulary); alternative architectures do the same or did not fit the task. In **Character-Based Transformer (CBT)**, tokens are atomic ASCII characters plus 14 structural control tags positioned on a 2D coordinate grid $(p, \tau)$ via **Decoupled 2D Rotary Position Embeddings (2D-RoPE)** and **Weight-Tied Recurrent Depth ($R = 3$)** (where our 100k factor ablation indicates that 2D coordinates account for the length-generalization gain while recurrence provides no measurable benefit for addition).

### Key Results Matrix (Greedy Decoding)

| Model | Non-embedding params | 4-8 digits (seen) | 12 digits (longest trained) | 14-25 digits (unseen) | 30-35 digits (unseen) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1D-RoPE character baseline** (`char_1d_rope_scratchpad_12d.pt`) | 18.9M | 60/60 | 0/30 | 0/150 | 0/60 |
| **2D-RoPE small model** (`hrm_2d_rope_k32_12d.pt`) | 1.57M | 60/60 | 30/30 | 150/150 | 47/60 |
| **CBT-18.95M** (V5 model, `cbt_18m_2d_rope_R3_k45_sftv5.pt`, HF: `champion_final.pt`) | 18.9M | 60/60 | 30/30 | 120/120 | 60/60 |
| **1D-RoPE twin of CBT-18.95M** (`cbt_18m_1d_rope_R3_k45_sftv5.pt`) | 18.9M | 60/60 | 29/30 | 117/120 | 0/60 |

Rows 1-2: free-running greedy decoding, exact final answer, 30 random problems per length (14-25 digits: lengths 14, 16, 18, 20, 25). Rows 3-4: teacher-forced greedy agreement with the gold scratchpad and answer (the condition for free-running greedy decoding to reproduce it), 30 problems per length (14-25 digits: lengths 14, 16, 20, 25). Results are in `results/baseline_rerun/`, `results/champion_v3_*.json` and `results/cbt1d_v5_*.json`. Other baselines (BPE, looped, hierarchical, hybrid, H-NRM, ALSM) are in the paper's Tables 1-2.

- **Length generalization (V5 model, $p_{\max} = 25 + k = 70$)**: teacher-forced, 30 problems per length. **150/150** at 4-16 digits, **60/60** at 20-25 (seen), **210/210** at 26-55 digits (unseen, 95% Wilson CI `[98.2, 100.0]%`), **25/30** at 60 and **19/30** at 70 digits (pooled 26-70: **254/270**). The remaining failures are local carry or digit errors at columns 59-70 of uniform sums; every cascade and dense-carry sum at 55-70 digits is exact. Traces for 80 or more digits exceed the 1,024-token context.
- **Tool use (V5 model)**: **26/26** exact calls on the phrasing suite, **18/18** on the OOD suite, **30/30** exact copies of entities absent from the training corpus. The V5 data was broadened after the previous model's failures on the first two suites were seen, so the unseen-30 set is the cleaner evidence.
- **1D-RoPE control (same architecture, V5 data and steps)**: on 26-70 digits the 2D model scores **254/270** and the 1D twin **0/270** (it fails from column 1 of the scratchpad); on tool use they are equal (74/74 vs 73/74 exact calls). So the 2D coordinates account for the arithmetic extrapolation, not for the tool-use results. Details: `results/cbt_2d_vs_1d_rope_ablation_report.md` and `results/champion_v3_performance_report.md`.

---

## 2. Champion Model: Character Based Transformer (CBT)

The repository champion model is **Character Based Transformer (CBT)** (~18.95M parameters), optimized for dual algorithmic reasoning and multi-tool agent execution:

- **Parameters**: 18,944,512 total (18,887,680 non-embedding)
- **Geometry**: d_model = 512, n_head = 8, head_dim = 64, d_ffn = 2048
- **Physical Layers**: L = 6
- **Recurrent Passes**: R = 3 (Effective Depth = 18 layers)
- **Positional Bias**: Decoupled 2D-RoPE with selective coordinate shifting
- **Training Trajectory**: pretraining on 1-15-digit addition and tool dialogues, search-copy and all-tools SFT, V4 SFT (12,000 steps), then V5 SFT (23,927 steps; decontaminated stream with arithmetic replay of 1-25 digits). The released weights are the V5 model. The previous V4 checkpoint is `checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sftv4.pt`.

### Model Weights on Hugging Face

The trained model checkpoint (`champion_final.pt`, ~75.8 MB) is available on Hugging Face:

- **Repository**: [https://huggingface.co/Neithanvarvas/CBT](https://huggingface.co/Neithanvarvas/CBT)
- **Checkpoint file**: `champion_final.pt`

To download and place the weights locally:
```bash
# Using huggingface_hub
.venv/bin/python benchmarks/download_from_hf.py  # saves checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sftv5.pt
```

---

## 3. Repository Topology

The repository follows a clean, modular structure:

```
LLM_Explanation/
├── README.md                      # Primary documentation and quickstart
├── RESEARCH_PAPER.md              # Complete scientific paper and theoretical analysis
├── AGENTS.md                      # Developer and agent operational guidelines
├── config.json                    # Canonical baseline hyperparameters
├── requirements.txt               # Pinned Python dependencies
├── run_t4_research.sh             # Unified CLI orchestrator for 2x T4 hardware
│
├── champion_model/                # Standalone Champion Model Suite
│   ├── checkpoints/               # Directory for champion weights (downloaded from HF)
│   ├── scaled_t4/                 # Character Based Transformer (CBT, 18.95M, 1-50 digits + tool calling)
│   │   ├── model.py               # CBT architecture (2D-RoPE + Recurrent unrolling)
│   │   ├── tokenizer.py           # ScaledTokenizer with tool schema support
│   │   ├── dataset.py             # Arithmetic and structured tool dataset generator
│   │   ├── interactive.py         # Live interactive terminal CLI
│   │   ├── sft_search.py          # Supervised Fine-Tuning pipeline
│   │   └── config.json            # Model architecture configuration
│   └── math_model/                # Ultra-compact 100k recurrent math model
│
├── src/                           # Core research library
│   ├── model.py                   # MathGPT reference implementation
│   ├── tokenizer.py               # Character math tokenizer
│   ├── bpe_tokenizer.py           # Subword BPE tokenizer wrapper
│   ├── arithmetic.py              # Column arithmetic logic and scratchpad formatting
│   ├── distributed.py             # DDP and FP16 utilities
│   └── constants.py               # Global dataclass configurations
│
├── benchmarks/                    # Evaluation and extrapolation suites
│   ├── extrapolation.py           # 1-12 vs 14-20 digit extrapolation tests
│   ├── evaluate_checkpoints.py    # Multi-checkpoint evaluation suite
│   ├── verify_environment.py      # Runtime and hardware diagnostic tests
│   └── interact.py                # Terminal inference runner
│
├── experiments/                   # Mechanistic ablations
│   ├── train_100k_factor_ablations.py # Factor ablation matrix
│   ├── scratchpad_mechanics.py    # State machine transition ablations
│   └── sample_efficiency_2d_rope.py # 2D-RoPE sample efficiency curves
│
├── tests/                         # Full unit test suite (100+ tests)
├── notebooks/                     # Interactive exploratory notebooks
└── results/                       # Empirical CSV leaderboards and JSON logs
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Linux OS (Ubuntu 22.04 LTS or Arch Linux recommended)
- Python 3.10+ (tested up to Python 3.14)
- PyTorch 2.1+ with CUDA or CPU support

### Setup
```bash
# Clone the repository
git clone https://github.com/NeithanT/CBT.git
cd CBT

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Quickstart & Interactive Inference

Inference is designed to be directly human-executable via the research orchestrator or Python CLI.

### 1. Execute a Single Arithmetic Problem
Test high-digit arithmetic directly from the command line:
```bash
./run_t4_research.sh interact --prompt "123+456="
```
Formatted Human-Readable Terminal Output:
```text
--------------------------------------------------------------------------------
Problem:    123+456 = ?
Scratchpad: (0+3+6)=09(0+2+5)=07(0+1+4)=05
Answer:     579
Validation: [PASS] (Expected: 579)
--------------------------------------------------------------------------------
[RAW TOKENS] <BOS>123+456=<THINK>(0+3+6)=09(0+2+5)=07(0+1+4)=05</THINK>=579<EOS>
```

### 2. Execute a Natural Language Tool Query
Test multi-tool routing and mock execution:
```bash
./run_t4_research.sh interact --prompt "What is the weather in Rome?"
```
Formatted Human-Readable Terminal Output:
```text
--------------------------------------------------------------------------------
Reasoning:   The user requests current weather information. I will query the get_weather tool.
Tool Call:   get_weather(city="Rome")
Tool Result: {"temp": 22, "condition": "Sunny", "city": "Rome"}
Response:    The weather in Rome is currently Sunny with a temperature of 22 degrees Celsius.
--------------------------------------------------------------------------------
```

### 3. Launch the Interactive Chat Terminal
Enter an ongoing dialogue session:
```bash
./run_t4_research.sh interact
```
```text
================================================================================
CHARACTER BASED TRANSFORMER (CBT, ~18.95M) INTERACTIVE TERMINAL
Type an arithmetic equation (e.g. '12345+67890=') or a natural question.
Type 'exit' or 'quit' to terminate.
================================================================================
User >> 
```

---

## 6. GitHub & Hugging Face Workflow

This project couples GitHub for version-controlled code, documentation, and evaluation suites with Hugging Face Hub for model weight distribution and deployment.

| Ecosystem | Destination | Purpose |
| :--- | :--- | :--- |
| **GitHub** | [https://github.com/NeithanT/CBT](https://github.com/NeithanT/CBT) | Open-source code, 2D-RoPE implementation, tests, and paper |
| **Hugging Face** | [https://huggingface.co/Neithanvarvas/CBT](https://huggingface.co/Neithanvarvas/CBT) | Pretrained champion weights (`champion_final.pt`), model card, and quickstart |

### 1. Pulling Model Weights from Hugging Face

Weights are automatically pulled if missing when launching inference, or can be pulled explicitly with any of the following methods:

**Method A: Dedicated CLI command (Recommended)**
```bash
./run_t4_research.sh download
```

**Method B: Python script**
```bash
python benchmarks/download_from_hf.py
```

**Method C: Hugging Face Python SDK**
```bash
.venv/bin/python benchmarks/download_from_hf.py  # saves checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sftv5.pt
```

### 2. Pushing New Weights to Hugging Face

To publish updated weights and model artifacts to Hugging Face:
```bash
# 1. Authenticate with your Hugging Face write token
hf auth login

# 2. Upload model bundle
python benchmarks/upload_to_hf.py --repo_id "Neithanvarvas/CBT"
```
This utility automatically bundles and verifies:
- Trained checkpoint (`champion_final.pt`, ~75.8 MB)
- Model card metadata (`README.md` with tags and benchmark leaderboard)
- Architecture code (`model.py`), tokenizer (`tokenizer.py`), and hyperparameters (`config.json`)
- Interactive CLI runner (`interactive.py`)

---

## 7. Verification & Test Suite

Verify complete test coverage across model architecture, 2D-RoPE coordinate generators, tokenizers, and loss masking:
```bash
# Run all unit tests
.venv/bin/python -m unittest discover -s tests

# Run runtime environment diagnostics
./run_t4_research.sh verify
```

---

## 8. Citation & References

If you use this codebase or architecture in your research, please cite:

```bibtex
@article{vargas2026attention,
  title={From Digits to Characters: Two-Dimensional Rotary Coordinates for Length Generalization in a Compact Character-Level Transformer},
  author={Vargas, Neithan},
  year={2026}
}
```
