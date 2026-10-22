# Attention Is Not All You Need for AGI: Empirical Limits of Pure Self-Attention on Algorithmic Length Generalization

[![Paper](https://img.shields.io/badge/Paper-RESEARCH__PAPER.md-blue)](RESEARCH_PAPER.md)
[![Status](https://img.shields.io/badge/Status-Reproducible%20%5BPASS%5D-brightgreen)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Champion%20Weights-orange)](https://huggingface.co/)

**Primary Scientific Reference**: [RESEARCH_PAPER.md](RESEARCH_PAPER.md)  
**Target Hardware**: 2x NVIDIA T4 GPUs (Turing `sm_75`, 16 GB GDDR6 each, FP16 AMP)  
**Evaluated Scope**: 1 to 50 Digit Arithmetic Extrapolation & Multi-Tool Agent Routing

---

## 1. Executive Summary & Breakthrough Findings

Can an autoregressive language model internalize the underlying sequential state machine of arithmetic when trained up to $n$ digits, and extrapolate that algorithm invariantly to $n + x$ digits?

Standard autoregressive Transformers rely on soft pairwise routing across static depth $L$. When extrapolating beyond their training horizon ($n > 12$ digits), standard models fail catastrophically ($0.0\%$ extrapolation accuracy) due to:
1. **Positional Phase Drift**: Monotonic 1D Rotary Position Embeddings (1D-RoPE) rotate query and key representations out of phase as context length $T$ grows.
2. **Bounded Sequential Depth**: Static feed-forward layers cannot propagate cascading carry transitions ($99\dots99 + 1$) without state decay.

### The Solution: Decoupled 2D-RoPE + Recurrent Weight-Tied Depth

To solve these limitations, this repository introduces two complementary architectural inductive biases:
1. **Decoupled 2D Rotary Position Embeddings (2D-RoPE)**: Splits head dimension $d_k$ into orthogonal decimal-place ($\Delta p = 0$) and template-step ($\tau$) coordinates, enforcing invariant column matching independent of sequence length.
2. **Weight-Tied Recurrent Depth ($R = 3$)**: Unrolls physical Transformer layers recurrently across $R=3$ passes per step with residual scaling, providing deep multi-step state routing at zero parameter cost.

### Key Results Matrix

| Architecture | Total Params | Effective Depth | In-Distribution (1–12 Digits) | Extrapolation (14–20 Digits) | 50-Digit Cascade Carry ($99\dots99+1$) | Multi-Tool Routing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Transformer (1D-RoPE)** | 18.91M | 6 layers | 33.3% | 0.0% | 0.0% | 12.0% |
| **HRM-RoPE Baseline** | 1.57M | 8 cycles | 100.0% | 91.7% | 100.0% (20d) | N/A |
| **100k Recurrent 2D-RoPE** | 100,672 | 6 layers (unrolled) | 100.0% | 100.0% | 100.0% (20d) | 100.0% (50/50) |
| **ScaledGPT (Champion Model)** | **18.95M** | **18 layers (unrolled)** | **100.0% [PASS]** | **100.0% [PASS]** | **100.0% (50d) [PASS]** | **100.0% (26/26) [PASS]** |

---

## 2. Champion Model: ScaledGPT

The repository champion model is **ScaledGPT** (~18.95M parameters), optimized for dual algorithmic reasoning and multi-tool agent execution:

- **Parameters**: 18,944,512 total (18,887,680 non-embedding)
- **Geometry**: $d_{\text{model}} = 512$, $n_{\text{head}} = 8$, head dimension $= 64$, $d_{\text{ffn}} = 2048$
- **Physical Layers**: $L = 6$
- **Recurrent Passes**: $R = 3$ (Effective Depth $= 18$ layers)
- **Positional Bias**: Decoupled 2D-RoPE with selective coordinate shifting
- **Training Trajectory**: Pretrained on 247.3M tokens (loss `0.000094`), followed by 2-stage Supervised Fine-Tuning (SFT) for 1-hop induction copy and tool routing.

### Model Weights on Hugging Face

The trained model checkpoint (`champion_final.pt`, ~75.8 MB) is available on Hugging Face:

- **Repository**: `[Hugging Face Model Repository Link]`
- **Checkpoint file**: `champion_final.pt`

To download and place the weights locally:
```bash
# Using huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='<REPO_ID>', filename='champion_final.pt', local_dir='champion_model/scaled_t4/checkpoints')"
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
│   ├── scaled_t4/                 # ScaledGPT (18.95M, 1-50 digits + tool calling)
│   │   ├── model.py               # ScaledGPT architecture (2D-RoPE + Recurrent unrolling)
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
git clone https://github.com/<username>/LLM_Explanation.git
cd LLM_Explanation

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
SCALEDGPT CHAMPION (~18.95M) INTERACTIVE TERMINAL
Type an arithmetic equation (e.g. '12345+67890=') or a natural question.
Type 'exit' or 'quit' to terminate.
================================================================================
User >> 
```

---

## 6. Model Weights & Hugging Face Upload

### Downloading Pretrained Champion Weights
To download the champion weights from Hugging Face:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='<username>/scaled-gpt-recurrent-2d-rope', filename='champion_final.pt', local_dir='champion_model/scaled_t4/checkpoints')"
```

### Uploading Your Trained Weights to Hugging Face
To publish your local weights and model card to Hugging Face:
```bash
python benchmarks/upload_to_hf.py \
  --repo_id "<username>/scaled-gpt-recurrent-2d-rope" \
  --token "<your-hf-token>"
```
This utility automatically packages:
- Model weights (`champion_final.pt`, ~75.8 MB)
- Model card metadata (`README.md` with benchmark results)
- Architecture implementation (`model.py`)
- Tokenizer (`tokenizer.py`)
- Configuration hyperparameters (`config.json`)
- Interactive CLI (`interactive.py`)

---

## 6. Verification & Test Suite

Verify complete test coverage across model architecture, 2D-RoPE coordinate generators, tokenizers, and loss masking:
```bash
# Run all unit tests
.venv/bin/python -m unittest discover -s tests

# Run runtime environment diagnostics
./run_t4_research.sh verify
```

---

## 7. Citation & References

If you use this codebase or architecture in your research, please cite:

```bibtex
@article{vargas2026attention,
  title={Attention Is Not All You Need for AGI: Empirical Limits of Pure Self-Attention on Algorithmic Length Generalization},
  author={Vargas, Neithan},
  journal={DeepMind / TEC Advanced Research},
  year={2026}
}
```
