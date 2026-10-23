# Empirical Performance Report: CBT-18.95M Champion V2 (`champion_v2_final.pt`)

**Evaluation Date**: 2026-10-22  
**Checkpoint Path**: `champion_model/scaled_t4/checkpoints/champion_v2_final.pt`  
**Training Configuration**: 12,000 steps, Dual NVIDIA T4 (Turing `sm_75`, FP16 AMP), Final Training Loss: `0.0034`, Cosine LR: $5.0 \times 10^{-5} \to 2.0 \times 10^{-6}$, $k_{\text{rand\_shift}} = 45$ ($p_{\max} = 70$).  
**Evaluation Mode**: Deterministic greedy decoding ($\text{temperature} = 0.0$, $\arg\max$).

---

## 1. Arithmetic Extrapolation (16 to 55 Digits)

Evaluated via batched KV-cached 2D-RoPE inference on Intel Core Ultra 5 125H (`results/champion_v2_extrapolation_16_55d.json`). Stratified distribution per length: 70% uniform random multi-carry, 15% dense carry ($a_i + b_i \ge 9$), 15% cascading carry ($99\dots9 + b$).

| Digit Length ($d$) | Sample Size ($n$) | Target Answer Exact Match | `<THINK>` Scratchpad Exact | 95% Wilson Confidence Interval | Syntax / Readout Errors | CPU Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 101.8 tok/s |
| **20 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 108.6 tok/s |
| **25 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 109.4 tok/s |
| **30 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 98.5 tok/s |
| **35 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 97.6 tok/s |
| **40 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 90.5 tok/s |
| **45 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 91.6 tok/s |
| **50 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 83.7 tok/s |
| **55 Digits** | 10 | 10 / 10 (100.0%) | 10 / 10 (100.0%) | [72.2, 100.0]% | 0 | 83.7 tok/s |
| **Summary 16–50d** | **80** | **80 / 80 (100.0%)** | **80 / 80 (100.0%)** | **[95.4, 100.0]%** | **0** | **94.8 tok/s** |
| **Summary 16–55d** | **90** | **90 / 90 (100.0%)** | **90 / 90 (100.0%)** | **[95.9, 100.0]%** | **0** | **93.6 tok/s** |

### Statistical Limitations on Arithmetic
1. **Sample Size Constraints**: Evaluating $n=10$ per length limits empirical power. The 95% Wilson confidence interval lower bound for any individual length is **72.2%**.
2. **Comparison with Baseline Benchmark**:
   - Baseline (`champion_final.pt`, $n=240$ on 16–50d): 237 / 240 (98.75%, 95% CI [96.35, 99.58]%).
   - Champion V2 (`champion_v2_final.pt`, $n=80$ on 16–50d): 80 / 80 (100.0%, 95% CI [95.42, 100.0]%).
   - Two-tailed Fisher's exact test: $p = 1.000$ ($p > 0.05$). The confidence intervals overlap; the difference is not statistically significant at current sample size.

---

## 2. Multi-Tool Phrasing & Edge Cases Benchmark (26 Cases)

Evaluated across 9 tool schemas ([`benchmarks/evaluate_champion_v2.py`](file:///home/neithanvv/Desktop/Projects/LLM_Explanation/benchmarks/evaluate_champion_v2.py)):

- **Tool Routing Accuracy**: 21 / 26 (**80.8%**, 95% CI [62.1, 91.5]%)
- **Exact Argument Extraction Accuracy**: 19 / 26 (**73.1%**, 95% CI [53.9, 86.3]%)

### Sub-Domain Breakdown

| Category | Passed / Total | Accuracy | Failure Mode |
| :--- | :---: | :---: | :--- |
| **Weather (7 phrasings)** | 7 / 7 | 100.0% | None |
| **Time (3 timezones)** | 3 / 3 | 100.0% | None |
| **Unit Convert (2 formats)** | 2 / 2 | 100.0% | None |
| **Search (5 phrasings/nonces)** | 4 / 5 | 80.0% | `Look up quantum physics` routed to search but argument was `"cuantum Physics"` |
| **Calculator (3 arithmetic)** | 2 / 3 | 66.7% | `15 * 24` normalized spacing to `"15*344"` |
| **Database Query (Nested JSON)** | 0 / 1 | 0.0% | Thought generated, failed to emit closing `</CALL>` tag within token limit |
| **Fetch API** | 0 / 1 | 0.0% | Misrouted to `search` with hallucinated query |
| **Create Event** | 0 / 1 | 0.0% | Misrouted to `calculator` triggered by time format `"10:00"` |
| **File Manager** | 0 / 1 | 0.0% | Misrouted to `search` with garbled phonetic query |
| **Direct Conversational Chat** | 1 / 2 | 50.0% | False positive trigger on `"Hello, how are you today?"` $\to$ `<CALL>get_weather(city="Toro")</CALL>` |

### Logged Failures in Phrasing Suite
```
[FAIL] Nested DB Query  -> I will invoke database_query with a sested filter 
[FAIL] Fetch API        -> <CALL>search(query="EUTCEUTCDrich Grom Jetice")</CALL>
[FAIL] Create Event     -> <CALL>calculator(expr="1000-10")</CALL>
[FAIL] File Manager     -> <CALL>search(query="Vontjjt welbro Qorld Hello Vorld")</CALL>
[FAIL] Direct Chat 2    -> <CALL>get_weather(city="Toro")</CALL>
```

---

## 3. Zero-Shot Out-of-Distribution (OOD) Benchmark (18 Cases)

Evaluated on Table 5 test cases ([`benchmarks/evaluate_champion_v2.py`](file:///home/neithanvv/Desktop/Projects/LLM_Explanation/benchmarks/evaluate_champion_v2.py)):

- **OOD Tool Routing Accuracy**: 17 / 18 (**94.4%**, 95% CI [74.2, 99.0]%)
- **OOD Exact Argument Copy (Given Routed)**: 16 / 17 (**94.1%**, 95% CI [73.0, 99.0]%)
- **Headline Exact Match Overall**: 16 / 18 (**88.9%**, 95% CI [67.2, 96.9]%)

### Itemized Results

| Domain | Prompt | Predicted Output | Status |
| :--- | :--- | :--- | :---: |
| OOD Science | `<USER>What is quantum physics</USER>` | `<CALL>search(query="quantum physics")</CALL>` | `[PASS] (Exact)` |
| OOD Film/Term | `<USER>What is interstellar</USER>` | `<CALL>search(query="interstellar")</CALL>` | `[PASS] (Exact)` |
| OOD Economics | `<USER>Define hyperinflation</USER>` | `<CALL>search(query="hyperinflation")</CALL>` | `[PASS] (Exact)` |
| OOD Cuisine | `<USER>What is tiramisu?</USER>` | `<CALL>search(query="tiramisu")</CALL>` | `[PASS] (Exact)` |
| OOD Astronomy | `<USER>What are neutron stars</USER>` | `<CALL>search(query="neutron stars")</CALL>` | `[PASS] (Exact)` |
| OOD Weather (City) | `<USER>How is the weather in Reykjavik</USER>` | `<CALL>get_weather(city="Reykjavik")</CALL>` | `[PASS] (Exact)` |
| OOD Unit (Inverted) | `<USER>How many km is 42 miles</USER>` | `<CALL>unit_convert(value=42, from_unit="miles", to_unit="km")</CALL>` | `[PASS] (Exact)` |
| OOD Timezone | `<USER>What time is it in GMT</USER>` | `<CALL>get_time(timezone="GMT")</CALL>` | `[PASS] (Exact)` |
| OOD Philosophy | `<USER>What is existentialism?</USER>` | `<CALL>search(query="existentialism")</CALL>` | `[PASS] (Exact)` |
| OOD Mythology | `<USER>Tell me about Ragnarok</USER>` | `<CALL>search(query="Ragnarok")</CALL>` | `[PASS] (Exact)` |
| OOD Calculator | `<USER>Compute 84 * 19</USER>` | `<CALL>calculator(expr="84 *191")</CALL>` | `[FAIL] (Argument Drift)` |
| OOD Deep Learning | `<USER>What are transformers in deep learning</USER>` | `<CALL>search(query="transformers in deep learning")</CALL>` | `[PASS] (Exact)` |
| OOD Music/Instrument | `<USER>What is a theremin</USER>` | `<CALL>search(query="theremin")</CALL>` | `[PASS] (Exact)` |
| OOD History (2-Word) | `<USER>Who was Marcus Aurelius?</USER>` | `<CALL>search(query="Marcus Aurelius")</CALL>` | `[PASS] (Exact)` |
| OOD Weather (2-Word) | `<USER>What is the weather in Buenos Aires?</USER>` | `<CALL>get_weather(city="Buenos Aires")</CALL>` | `[PASS] (Exact)` |
| OOD Weather (2-Word) | `<USER>Weather on Kuala Lumpur</USER>` | `<CALL>get_weather(city="Kuala Lumpur")</CALL>` | `[PASS] (Exact)` |
| OOD Bio (With Digit) | `<USER>Explain crispr cas9</USER>` | `The user is asking a factual questing fabsemetical... unit_convert...` | `[FAIL] (Misrouted)` |
| OOD Architecture | `<USER>Search for brutalist architecture</USER>` | `<CALL>search(query="brutalist architecture")</CALL>` | `[PASS] (Exact)` |

### Caveats on OOD Copying
1. **Entity Inclusion**: `Marcus Aurelius`, `Buenos Aires`, `existentialism`, `brutalist architecture`, and `Ragnarok` were added to `SEARCH_ENTITIES` and `CITIES` during Curriculum V4 preparation. Exact reproduction of these spans reflects within-distribution retrieval rather than unassisted zero-shot generalization.
2. **Persistent Alphanumeric Defect**: Concepts containing embedded numbers (`crispr cas9`) reliably trigger numeric / unit conversion sub-routines, demonstrating an unresolved failure in syntactic routing when numbers appear inside semantic terms.
3. **Argument Copy Drift**: The calculator case (`84 * 19` $\to$ `84 *191`) shows that numeric copying is vulnerable to hallucinated suffix digits.

---

## 4. Head-to-Head Unseen Zero-Shot OOD Benchmark (30 Cases)

To remove any potential training data leakage, 30 strictly unseen entities were tested ([`benchmarks/evaluate_unseen_ood.py`](file:///home/neithanvv/Desktop/Projects/LLM_Explanation/benchmarks/evaluate_unseen_ood.py)). **Zero intersection was mathematically verified** against `CITIES`, `SEARCH_ENTITIES`, and `SEARCH_FACTS`.

### Aggregate Results: Baseline vs. Champion V2

| Evaluation Metric | Baseline `champion_final.pt` | Champion V2 `champion_v2_final.pt` | Statistical Significance |
| :--- | :---: | :---: | :---: |
| **Tool Routing Accuracy** | 28 / 30 (93.3%, 95% CI [78.7, 98.2]%) | **29 / 30 (96.7%, 95% CI [83.3, 99.4]%)** | $p = 1.000$ (No difference) |
| **Exact Copy Given Routed** | 4 / 28 (14.3%, 95% CI [5.7, 31.5]%) | **17 / 29 (58.6%, 95% CI [40.7, 74.5]%)** | $p = 0.0007$ (Significant gain over baseline) |
| **Headline Exact Match Overall** | 4 / 30 (13.3%, 95% CI [5.3, 29.7]%) | **17 / 30 (56.7%, 95% CI [39.2, 72.6]%)** | $p = 0.0008$ (Significant gain over baseline) |
| **One-Sided Binomial vs. 50% ($H_0: p \le 0.50$)** | $p \approx 1.000$ | **$p = 0.2923$** | **Cannot reject $H_0 \le 50\%$** |

### Question: Does Champion V2 Achieve Better Than 50%?
- **Point Estimate**: Nominally **yes** (56.7% headline exact match; 58.6% given correct tool routing).
- **Strict Statistical Rigor**: **No**. Because the 95% Wilson confidence interval spans **[39.2%, 72.6%]** (spanning below 50.0%) and the one-sided binomial test yields $p = 0.2923 > 0.05$, the model **cannot be claimed to be statistically significantly better than 50%**.

### Category-by-Category Exact Match Breakdown

| Sub-Category | Scope | Baseline `champion_final` | Champion V2 `champion_v2_final` |
| :--- | :--- | :---: | :---: |
| **Unseen Figures (6)** | Real historical figures (Spinoza, Hypatia, Beethoven, etc.) | 0 / 6 (0.0%) | **1 / 6 (16.7%)** |
| **Unseen Cities (6)** | Global cities (Addis Ababa, Ulaanbaatar, Belo Horizonte, etc.) | 0 / 6 (0.0%) | **2 / 6 (33.3%)** |
| **Unseen Science (6)** | Multi-word & technical terms (quarks, epigenetics, bose einstein) | 1 / 6 (16.7%) | **5 / 6 (83.3%)** |
| **Unseen Concepts (6)** | Foods and artifacts (croissant, baklava, clavichord, origami) | 2 / 6 (33.3%) | **5 / 6 (83.3%)** |
| **Unseen Nonces (6)** | Pure synthetic pseudowords (Zyntral Krelvok, Brixendale, etc.) | 1 / 6 (16.7%) | **4 / 6 (66.7%)** |
| **Total (30)** | Complete Unseen Benchmark | **4 / 30 (13.3%)** | **17 / 30 (56.7%)** |

