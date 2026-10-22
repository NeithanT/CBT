# AGENTS.md: Developer and Agent Guidelines

## 1. Project Context & Scientific Objective
- **Project Title**: Attention Is Not All You Need for AGI: Empirical Limits of Pure Self-Attention on Algorithmic Length Generalization
- **Primary Reference**: [RESEARCH_PAPER.md](RESEARCH_PAPER.md)
- **Central Hypothesis**: Can an autoregressive language model internalize the underlying sequential state machine of arithmetic when trained up to $n$ digits, and extrapolate that algorithm invariantly to $n+x$ digits?
- **Theoretical Grounding**: Standard self-attention relies on soft pairwise routing across static depth $L$, failing to compute unbounded sequential carry state transitions without degradation ("Math addition is not correct in LLM").
- **Hardware Platform**: 2x NVIDIA T4 GPUs (16 GB GDDR6 each, Turing `sm_75`).

---

## 2. Token Efficiency & Code Hygiene Rules

Future agents must adhere strictly to these engineering standards:

### Zero Emojis
- Do not introduce emojis in code, docstrings, commit messages, or markdown documentation.
- Use explicit technical status codes: `[OK]`, `[PASS]`, `[FAIL]`, `[WARN]`, `[INFO]`.

### Eliminate Redundancies
- Do not duplicate data generation, tokenization, or arithmetic formatting across scripts.
- Single source of truth:
  - Tokenizers -> `src.tokenizer`, `src.bpe_tokenizer`
  - Model Architecture -> `src.model`
  - Arithmetic Logic -> `src.arithmetic`
  - Datasets & Caches -> `src.dataset`
  - Distributed & AMP Utilities -> `src.distributed`
  - Configurations -> `src.constants`

### Token-Efficient Documentation
- Docstrings and comments must be high-signal, concise, and technical.
- Prohibit repetitive boilerplate (e.g., `# import torch`, `# set variable`, decorative line-art borders like `### ========== ###`).
- Keep function signatures fully typed (`typing.Tuple`, `typing.Optional`, `typing.List`, `typing.Dict`).

---

## 3. Directory Structure & File Placement Policy

Never add loose checkpoints, scratch files, or scripts to the project root. Follow this topology:

| Directory | Purpose | Rules |
| :--- | :--- | :--- |
| `src/` | Core library package | Self-contained, modular, zero loose runner scripts. |
| `training/` | Training pipelines | Use `src/` imports and DDP/AMP utilities. |
| `benchmarks/` | Evaluation & inference | Extrapolation suite (1-12 vs 14-20 digits), interactive CLI. |
| `experiments/` | Mechanistic ablations | Scratchpad, length scaling, commutativity studies. |
| `tests/` | Unit test suite | All architecture, tokenizer, dataset tests pass. |
| `checkpoints/` | Model weights (`.pt`) | Organized into `baseline/`, `model_1/`, `model_2/`, `model_3/`. |
| `data/` | Deterministic datasets | Reusable invariant cache (`shared_math_pairs.pt`), philosophy corpus. |
| `results/` | Evaluation metrics | Itemized CSVs, JSON logs, `leaderboard.csv`. |
| `notebooks/` | Interactive notebooks | Jupyter/Colab notebooks (`run_in_colab.ipynb`). |
| `archive/` | Historical backups | Isolate legacy code and zip archives here. |

Root directory must remain reserved strictly for top-level documentation, global configs (`config.json`, `requirements.txt`), and the primary orchestrator (`run_t4_research.sh`).

---

## 4. Efficient LLM Implementation Patterns

When modifying or extending the model architecture or data pipeline:

### 1. Flash Attention / SDPA
- Use `torch.nn.functional.scaled_dot_product_attention` for all forward passes.
- Do not manually compute $Q K^T$ softmax matrices during training (SDPA triggers Turing Cutlass kernels with $O(1)$ activation memory).
- Materialize explicit attention weights only when `save_attention=True` is requested.

### 2. Rotary Position Embeddings (RoPE)
- Maintain precomputed cosine/sine rotary caches (`precompute_rope_freqs`).
- Dynamically extend frequency tables if inference sequence length $T$ exceeds precomputed block size to prevent out-of-bounds indexing.

### 3. Hardware Precision (Turing `sm_75`)
- Target hardware is Dual NVIDIA T4. Turing architecture Tensor Cores require **FP16** (`torch.float16`).
- **Never use `bfloat16`** or enable TF32; Turing lacks native hardware support and incurs severe software emulation slowdowns.
- Always couple `torch.amp.autocast('cuda', dtype=torch.float16)` with `torch.amp.GradScaler('cuda')` to prevent underflow.

### 4. Loss Masking for Arithmetic Reasoning
- In autoregressive arithmetic datasets (`a+b=<THINK>...</THINK>=answer`), always set targets for prompt tokens (everything up to and including `=`) to `-100` (`CrossEntropyLoss` ignore index).
- Loss must penalize only reasoning steps and predicted solution tokens.

### 5. Weight Tying
- Maintain tied weights between embedding matrix and output projection:
  `self.transformer.wte.weight = self.lm_head.weight`
  This reduces memory overhead and improves parameter efficiency in compact models (< 20M parameters).

### 6. Deterministic Shared Data
- When benchmarking across conditions, all models must train on identical numerical equation pairs loaded from `data/shared_math_pairs.pt` using fixed random seeds.

### 7. Token-Budget Driven Training (Compute-Optimal Scaling)
- Training completion must be governed by token volume (`target_tokens`, defaulting to 400M tokens for ~20M parameter models under $\tau = 20:1$ scaling), not arbitrary runtime hour cutoffs.
- The model trains for however long is needed until reaching the token budget. Cosine learning rate decay is calibrated to `total_target_steps = target_tokens // tokens_per_step`. Periodic checkpoints are saved every 50M tokens.

---

## 5. Verification & Safety Guidelines

Before concluding any modification:
1. **Run Unit Tests**:
   ```bash
   .venv/bin/python -m unittest discover -s tests
   ```
2. **Verify Environment & T4 Runtime**:
   ```bash
   ./run_t4_research.sh verify
   ```
3. **Emoji Audit**: Verify no emojis were added:
   ```bash
   .venv/bin/python -c 'import os, re; p=re.compile("[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]"); assert not any(p.findall(open(os.path.join(r, f), errors="ignore").read()) for r, _, fs in os.walk(".") if not any(x in r for x in [".venv", "archive"]) for f in fs if f.endswith((".py", ".md", ".sh"))), "Emojis detected!"'
   ```
4. **Safety**: Never execute destructive file operations (`rm -rf`). Move obsolete artifacts to `archive/`.
