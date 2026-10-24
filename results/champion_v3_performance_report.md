# Clean Performance Report: CBT-18.95M Champion v3 (`champion_v3_final`)

**Evaluation Date**: 2026-10-23
**Checkpoint**: `champion_model/checkpoints/champion_v3_final.zip` (a `torch.save` file despite the extension; SHA-256 `c6fab36eaadbc402441691ded7ed9e26990f3718e6cefcb5a889691b7fc3b9e5`)
**Lineage**: v2 champion (`checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sftv4.pt`) + Curriculum V5 SFT
**Training Data (V5 SFT)**: decontaminated online stream (`CurriculumV5Stream`). Fractions per batch: 35% search copy, 20% math replay (operands 1-25 digits), the rest multi-tool and chat. Loss weight 2.0 on `<CALL>` argument targets, cosine LR with a time-aware schedule, k_rand_shift = 45. The run stopped at step 23,927 on the 6 h wall-clock budget.
**Checkpoint metadata**: held-out tool exact 98.4% (63/64), held-out math exact 100%
**Decoding**: Greedy (temperature = 0, argmax)
**Hardware**: CPU only

The harness, suites, seeds and scoring are the same as in the v2 report (`results/champion_performance_report.md`). v2 numbers below are taken from that report.

---

## 0. Contamination Audit

- **Harness audit**: `[PASS] Contamination audit: 0 free-text overlaps with training corpus` on every suite.
- **V5 corpus audit**: the exported V5 training corpus (`champion_model/checkpoints/champion_v3_final_corpus.txt.gz`, 642,102 lines, 219.6M characters) has 0 violations. None of the 30 unseen-30 expected arguments occur in it.
- **One prompt overlap, disclosed**: the phrasing prompt `Thank you!` (Direct Chat 1) occurs in training. In the V5 corpus it appears only as a later dialogue turn (21,652 times), never as an opening turn. The V4 cache, which v3 inherits through v2, has 1,730 opening turns with it. This case is in-distribution for both v2 and v3. Excluding it, v3 phrasing is 25/25.
- Arithmetic lengths 20 and 25 are **seen**. Lengths 26-70 are **unseen**.

---

## 1. Arithmetic Extrapolation

Test problems: 70% uniform random operands, 15% dense carry (digits 5-9), 15% cascade (`99...9 + b`), seed `42 + digits`, 30 problems per length. These are the same problems used for v2.

### 1.1 Scoring Method

The scoring method is **teacher-forced greedy agreement**, as in the v2 report. One forward pass runs over the gold sequence `a+b=<THINK>...</THINK>=sum<EOS>`. A problem is correct only if the argmax equals gold at every position after the prompt. Under greedy decoding this is the same condition as an exact free-run match: if every argmax equals gold, greedy decoding reproduces gold exactly, and if any argmax differs, greedy decoding departs from gold at that position.

### 1.2 Results

| Digits | Status | n | v2 Exact | v3 Exact | v3 95% Wilson CI | Fisher p (v2 vs v3) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 20 | Seen | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 25 | Seen | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 26 | Unseen (+1) | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 30 | Unseen (+5) | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 35 | Unseen (+10) | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 40 | Unseen (+15) | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 45 | Unseen (+20) | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 50 | Unseen (+25) | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 55 | Unseen (+30) | 30 | 30 / 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 1 |
| 60 | Unseen (+35) | 30 | 24 / 30 | **25 / 30 (83.3%)** | [66.4, 92.7]% | 1.000 |
| 70 | Unseen (+45) | 30 | 24 / 30 | **19 / 30 (63.3%)** | [45.5, 78.1]% | 0.252 |
| **Seen (20, 25)** | | **60** | 60 / 60 | **60 / 60 (100.0%)** | [94.0, 100.0]% | 1 |
| **Unseen 26-55** | | **210** | 210 / 210 | **210 / 210 (100.0%)** | [98.2, 100.0]% | 1 |
| **Unseen 60-70** | | **60** | 48 / 60 | **44 / 60 (73.3%)** | [61.0, 82.9]% | 0.518 |
| **Unseen 26-70 (pooled)** | | **270** | 258 / 270 | **254 / 270 (94.1%)** | [90.6, 96.3]% | 0.561 |

Fisher p values are two-sided exact tests. `<THINK>` scratchpad exact equals answer exact at every length.

### 1.3 Failure Localization (v3, 55/60/70 digits)

For each of the 90 problems at 55, 60 and 70 digits, the first position where the argmax departs from gold was recorded (`results/champion_v3_arith_failure_analysis.json`). The per-length counts reproduce the sweep exactly (30, 25, 19).

| Property | Finding |
| :--- | :--- |
| Problem type | All 16 failures are **uniform random**. Every cascade (17/17) and dense-carry (9/9) problem at 55-70 digits is exact, including 9 cascades at 70 digits. |
| Region | All 16 first errors are inside the `<THINK>` scratchpad. None is in the final answer copy. |
| Column | First errors fall at columns 59-70. **No problem has an error in columns 1-58.** |
| Absolute position | First errors fall at token positions 705-840. All 55-digit sequences (up to 733 tokens) are exact. Some 70-digit sequences of 928 tokens are also exact, so there is no hard context cutoff. |
| Wrong tokens | 1-4 per problem under teacher forcing. In free-run decoding the first error would propagate. |

Error types at the first wrong token:

| Type | Count | Example (context -> gold / predicted) |
| :--- | :---: | :--- |
| Spurious carry-in (`0` -> `1`) after a column with carry-out 0 | 8 | `...(0+3+4)=07(` -> `0` / `1` |
| Column sum digit wrong | 5 | `...(1+9+2)=1` -> `2` / `6` |
| Extra operand digit in place of `)` | 3 | `...(1+3+6` -> `)` / `7` |

### 1.4 Interpretation

- v3 matches v2 at 100% on every length from 20 to 55. The 210/210 unseen result at 26-55 is unchanged.
- At 60 and 70 digits, the differences from v2 are not significant: 70d 24/30 -> 19/30, p = 0.25; pooled 26-70, p = 0.56. The data does not support calling the V5 SFT an arithmetic regression or an improvement. Its effect on arithmetic is within sampling noise.
- The failures share one signature. Each problem is correct up to about column 58 and then makes a local state error. Half are the carry flag failing to copy the previous column's carry-out. Column index and absolute position are confounded in this test (longer operands give both more columns and later positions), so it cannot say which of the two drives the breakdown. Both lie well beyond the 25-digit training maximum and the 55-digit range that remains exact.
- Cascade problems, which need the longest carry chain, are perfect at 70 digits. Uniform problems, which need the carry state to change from column to column, are the ones that fail. This is consistent with the hypothesis in `AGENTS.md`: the per-step carry state degrades with distance from the training regime, not the ability to repeat one pattern.

---

## 2. Multi-Tool Phrasing & Edge Cases (26 Cases, 9 Schemas)

| Metric | v2 | v3 | Fisher p |
| :--- | :---: | :---: | :---: |
| Tool routing | 21 / 26 (80.8%, [62.1, 91.5]%) | **26 / 26 (100.0%, [87.1, 100.0]%)** | 0.0506 |
| Exact argument extraction | 13 / 26 (50.0%, [32.1, 67.9]%) | **26 / 26 (100.0%, [87.1, 100.0]%)** | 3.3e-05 |
| Exact, excluding `Thank you!` | 12 / 25 | 25 / 25 (100.0%, [86.7, 100.0]%) | |

v3 fixes every v2 failure:

- City drift: Lagos, Accra and Yerevan are now exact.
- Time-zone codes: NZST, SAST and WAT are now exact.
- Search: "Johannes Kepler" is now exact.
- Calculator: `37 * 113` keeps all its digits.
- The nested DB query now emits a `<CALL>`.
- Fetch API, Create Event and File Manager are routed to the correct schema.
- "Hello, how are you today?" no longer triggers `get_weather`.

---

## 3. Zero-Shot OOD Copying (18 Fresh Prompts)

| Metric | v2 | v3 | Fisher p |
| :--- | :---: | :---: | :---: |
| Tool routing | 18 / 18 (100.0%) | 18 / 18 (100.0%, [82.4, 100.0]%) | 1 |
| Headline exact match | 10 / 18 (55.6%, [33.7, 75.4]%) | **18 / 18 (100.0%, [82.4, 100.0]%)** | 0.0029 |

All eight v2 drift cases are now exact: Tbilisi, WAT, nihilism, Yggdrasil, Hatshepsut, Santo Domingo, Ho Chi Minh City and `crispr cas12`. The last one is the digit-embedded case that failed in every earlier report.

---

## 4. Head-to-Head Unseen Zero-Shot Benchmark (30 Cases)

Protocol unchanged (`benchmarks/evaluate_unseen_ood.py`). 0 of the 30 expected arguments occur in the V5 corpus.

| Metric | Pre-SFT base | v2 | v3 |
| :--- | :---: | :---: | :---: |
| Tool routing | 28 / 30 | 29 / 30 | **30 / 30 (100.0%, [88.6, 100.0]%)** |
| Headline exact match | 4 / 30 (13.3%) | 17 / 30 (56.7%, [39.2, 72.6]%) | **30 / 30 (100.0%, [88.6, 100.0]%)** |
| One-sided binomial vs 50% | p = 1.000 | p = 0.292 | **p = 9.3e-10** |

Per-category headline exact (v3): figures 6/6, cities 6/6, science 6/6, concepts 6/6, nonces 6/6.

v3 vs v2 headline exact: 30/30 vs 17/30, two-sided Fisher p = 4.6e-05. v3 now exceeds the 50% majority threshold with high significance. v2 did not.

---

## 5. Limitations

1. **Template overlap is not removed.** Prompt templates appear throughout training. The tool-use claims are about novel entities under known templates, as in the v2 report.
2. **Closed-vocabulary arguments are exempt** from the audit (unit names, time zones), as in the v2 report.
3. **Small per-suite samples.** At 100% observed accuracy, the 95% CI lower bounds are 87.1% (n = 26), 82.4% (n = 18) and 88.6% (n = 30). A perfect score on these suites shows high accuracy, not that the true rate is 100%.
4. **The V5 stream was designed against these suites.** It is decontaminated at the entity level, but its prompt phrasings were broadened after the v2 failures were seen. The phrasing and OOD gains therefore partly reflect targeted coverage of those failure categories. The unseen-30 set, whose arguments are absent from the corpus, is the cleanest evidence of generalization.
5. **Column vs position is confounded** in the arithmetic failure analysis (Section 1.4).
6. **Single run, single seed.** Seed variance is not measured. The 70-digit difference between v2 and v3 is within the range that seed noise could explain.
7. **Position-encoding attribution**: see `results/cbt_2d_vs_1d_rope_ablation_report.md`. A 1D-RoPE twin trained on the same V5 data scores 0/270 at 26-70 digits, against 254/270 for this model, and matches it on tool use.

---

## 6. Raw Outputs

| File | Content |
| :--- | :--- |
| `results/champion_v3_phrasing.json` / `.log` | Phrasing suite |
| `results/champion_v3_ood.json` / `.log` | OOD suite |
| `results/champion_v3_arithmetic.json`, `results/champion_v3_arith.log` | Teacher-forced arithmetic, 20-70 digits |
| `results/champion_v3_arith_failure_analysis.json` / `.log` | First-error localization at 55/60/70 digits |
| `results/champion_v3_unseen_ood.json`, `results/unseen_ood_champion_v3.log` | Unseen-30 |

Commands:

```
CKPT=champion_model/checkpoints/champion_v3_final.zip
.venv/bin/python benchmarks/evaluate_champion.py --ckpt $CKPT --suite arithmetic --samples-per-length 30 --arith-mode teacher_forced --out-json results/champion_v3_arithmetic.json
.venv/bin/python benchmarks/evaluate_champion.py --ckpt $CKPT --suite phrasing --out-json results/champion_v3_phrasing.json
.venv/bin/python benchmarks/evaluate_champion.py --ckpt $CKPT --suite ood --out-json results/champion_v3_ood.json
.venv/bin/python benchmarks/evaluate_unseen_ood.py --ckpt $CKPT --out-json results/champion_v3_unseen_ood.json
```
