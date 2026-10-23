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

Small autoregressive Transformers trained from scratch on 1–12 digit addition with 1D Rotary Position Embeddings (1D-RoPE) fall to **0.0% accuracy** on unseen 18- to 20-digit sums (after scoring 4.6–18.7% on moderate 14–16 digit extrapolation). In **Character-Based Transformer (CBT)**, tokens are atomic ASCII characters plus 14 structural control tags positioned on a 2D coordinate grid $(p, \tau)$ via **Decoupled 2D Rotary Position Embeddings (2D-RoPE)** and **Weight-Tied Recurrent Depth ($R = 3$)** (where our 100k factor ablation indicates that 2D coordinates account for the length-generalization gain while recurrence provides no measurable benefit for addition).

### Key Results Matrix (Greedy Decoding)

| Architecture | Total Params | Effective Depth | In-Dist (1–12/15d) | Unseen 16–40d Random ($n=180$) | Unseen 16–50d Random ($n=240$) | 50-Digit Cascade ($99\dots99 + 1$) | Multi-Tool Routing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Transformer (1D-RoPE)** | 18.91M | 6 layers | 98.4% (1–12d) | 0.0% (18–20d) | N/A (512-tok limit) | N/A (512-tok limit) | N/A |
| **HRM-RoPE (Decoupled 2D-RoPE)** | 1.56M | 8 cycles | 100.0% (1–12d) | 100.0% (18–20d) | N/A (512-tok limit) | 100.0% (20d) | N/A |
| **CBT-100k ($L=2, R=3$)** | 100,672 | 6 layers (unrolled) | 100.0% (1–12d) | 100.0% (14–20d) | N/A (512-tok limit) | 100.0% (20d) | N/A |
| **CBT-200k ($L=4, R=3$)** | 197.8k | 12 layers (unrolled) | 100.0% (1–12d) | 100.0% (14–20d) | N/A (512-tok limit) | 100.0% (20d) | 100.0% (50/50) |
| **CBT-18.95M (`champion_v2_final.pt`)** | **18.95M** | **18 layers (unrolled)** | -- | **100.0% (90/90, 16–55d)** | **100.0% (80/80)** | -- | **80.8% (21/26) / 56.7% exact on 30 unseen** |

- **Shift-Masking Horizon ($p_{\max} = N_{\text{train}} + k = 15 + 45 = 70$)**: On random multi-carry sums (`results/champion_v2_extrapolation_16_55d.json`, $n=10$ per length), **CBT-18.95M** achieves **100.0% (90/90; 95% Wilson CI `[95.9, 100.0]%`)** across 16–55 digits. Lengths beyond 55 digits were not evaluated for this checkpoint.
- **Unseen-Entity Tool Calling**: On 30 entity-disjoint unseen prompts (`benchmarks/evaluate_unseen_ood.py`), CBT-18.95M routes **96.7% (29/30)** to the intended tool and copies **56.7% (17/30; CI `[39.2, 72.6]%`)** of arguments exactly; this is not shown to exceed 50%.

---

## 2. Champion Model: Character Based Transformer (CBT)

The repository champion model is **Character Based Transformer (CBT)** (~18.95M parameters), optimized for dual algorithmic reasoning and multi-tool agent execution:

- **Parameters**: 18,944,512 total (18,887,680 non-embedding)
- **Geometry**: d_model = 512, n_head = 8, head_dim = 64, d_ffn = 2048
- **Physical Layers**: L = 6
- **Recurrent Passes**: R = 3 (Effective Depth = 18 layers)
- **Positional Bias**: Decoupled 2D-RoPE with selective coordinate shifting
- **Training Trajectory**: Pretrained on 247.3M tokens (loss `0.000094`), followed by 2-stage Supervised Fine-Tuning (SFT) for 1-hop induction copy and tool routing.

### Model Weights on Hugging Face

The trained model checkpoint (`champion_v2_final.pt`, ~75.8 MB) is available on Hugging Face:

- **Repository**: [https://huggingface.co/Neithanvarvas/CBT](https://huggingface.co/Neithanvarvas/CBT)
- **Checkpoint file**: `champion_v2_final.pt`

To download and place the weights locally:
```bash
# Using huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Neithanvarvas/CBT', filename='champion_v2_final.pt', local_dir='champion_model/scaled_t4/checkpoints')"
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
| **Hugging Face** | [https://huggingface.co/Neithanvarvas/CBT](https://huggingface.co/Neithanvarvas/CBT) | Pretrained champion weights (`champion_v2_final.pt`), model card, and quickstart |

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
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Neithanvarvas/CBT', filename='champion_v2_final.pt', local_dir='champion_model/scaled_t4/checkpoints')"
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
- Trained checkpoint (`champion_v2_final.pt`, ~75.8 MB)
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
  title={Attention Is Not All You Need for AGI: Empirical Limits of Pure Self-Attention on Algorithmic Length Generalization},
  author={Vargas, Neithan},
  year={2026}
}
```
