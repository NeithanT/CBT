# Clean Performance Report: CBT-18.95M Champion (`champion_final.pt`)

**Evaluation Date**: 2026-10-23
**Checkpoint**: `checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sftv4.pt` (Hugging Face file name: `champion_final.pt`)
**Training Data (champion SFT)**: `champion_model/scaled_t4/data/sft_dataset_v4.pt` (45% multi-tool, 35% search copy, 20% math replay with operands of 1-25 digits, k_rand_shift = 45, p_max = 70)
**Decoding**: Greedy (temperature = 0, argmax)
**Hardware**: CPU only (no CUDA device available)

This report replaces `archive/champion_contaminated_2026-10-22/champion_performance_report.md`. The earlier report is withdrawn for the reasons in Section 0.

---

## 0. Why the Earlier Report Was Replaced

A contamination audit of the champion SFT training cache found three problems with the earlier evaluation:

1. **OOD and phrasing entities were in training.** Strings such as `Marcus Aurelius` (182 occurrences), `Buenos Aires` (639), `Reykjavik` (638), `Ragnarok` (304), `Kuala Lumpur` (563), `tomatoes` (39) and `Alexander Graham Bell` (277) appear in the champion training data. The earlier OOD and phrasing numbers therefore measured in-distribution copying, not generalization.
2. **16-25 digit arithmetic was in-distribution.** The champion math replay contains operands of 1-25 digits. The earlier "16-55 digit extrapolation" table was not a clean extrapolation test below 26 digits.
3. **The arithmetic harness was broken.** `extract_scratchpad_answer` returns a `(reasoning, answer)` tuple, and the earlier loop compared that tuple to an integer. Its arithmetic scores were not valid.

The rerun in this report addresses all three:

- Every free-text expected value in the phrasing, OOD and unseen-30 suites is checked against the training corpus, and the run aborts on any overlap. Result: **0 overlaps** (`[PASS] Contamination audit`). Closed-vocabulary schema enumerations (units, time zones) and template words (`What is`, `Define`, `Look up`) are exempt and disclosed in Section 5.
- Arithmetic lengths 20 and 25 are reported as **seen**. Lengths 26-70 are reported as **unseen**.
- Arithmetic is scored by teacher-forced greedy agreement (Section 1.1).

Old results (archived): `archive/champion_contaminated_2026-10-22/`.

---

## 1. Arithmetic Extrapolation

Test problems: 70% uniform random operands, 15% dense carry (digits 5-9), 15% cascade (`99...9 + b`), with a fixed seed per length. 30 problems per length.

### 1.1 Scoring Method

Scoring uses **teacher-forced greedy agreement**: for each problem, a single forward pass runs over the gold scratchpad `a+b=<THINK>...</THINK>=sum<EOS>`. A problem is counted correct only if the argmax equals the gold token at every position after the prompt. This holds exactly when greedy autoregressive decoding would reproduce the gold sequence.

This method was adopted because `ScaledGPT.generate` has no KV cache and the coordinate parser runs Python loops over every token, so free-running generation at 26-70 digits was impractical on CPU (about 4 tok/s). Free-running decoding was validated against this method on the same problems:

| Digits | Mode | Exact | Agreement with teacher-forced |
| :---: | :--- | :---: | :--- |
| 20 (seen) | Free-run, n = 2 | 2 / 2 | Matches teacher-forced on the same seeds |
| 26 (unseen) | Free-run, n = 2 | 2 / 2 | Matches teacher-forced on the same seeds |

The free-run validation is small (n = 2 per length). Its think-column check had a bug (the reference string still lacked `<THINK>`), so only its answer column is reported.

### 1.2 Results

| Digits | Status | n | Exact Match | 95% Wilson CI | `<THINK>` Scratchpad Exact |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 20 | Seen | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 25 | Seen | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 26 | Unseen (+1) | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 30 | Unseen (+5) | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 35 | Unseen (+10) | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 40 | Unseen (+15) | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 45 | Unseen (+20) | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 50 | Unseen (+25) | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 55 | Unseen (+30) | 30 | 30 / 30 (100.0%) | [88.6, 100.0]% | 30 / 30 |
| 60 | Unseen (+35) | 30 | **24 / 30 (80.0%)** | [68.2, 88.2]% | 24 / 30 |
| 70 | Unseen (+45) | 30 | **24 / 30 (80.0%)** | [68.2, 88.2]% | 24 / 30 |
| **Seen (20, 25)** | | **60** | **60 / 60 (100.0%)** | [94.0, 100.0]% | 60 / 60 |
| **Unseen 26-55** | | **210** | **210 / 210 (100.0%)** | [98.2, 100.0]% | 210 / 210 |
| **Unseen 60-70** | | **60** | **48 / 60 (80.0%)** | [68.2, 88.2]% | 48 / 60 |
| **Unseen 26-70 (pooled)** | | **270** | **258 / 270 (95.6%)** | [92.4, 97.4]% | 258 / 270 |

### 1.3 Interpretation

- Accuracy is 100% across all 210 problems at 26-55 digits. Lengths 26-55 lie beyond the 25-digit training ceiling and within the shift-masked coordinate support (p_max = 70).
- Accuracy drops at 60 and 70 digits (80%, 6 failures each). Both lengths are inside p_max = 70, so this is not explained by coordinate range alone. The cause is not established. The per-problem failure records were not saved, so the failure mode (digit error, carry error, or length/EOS error) is unknown.
- The earlier claim of a sharp 45-50 digit boundary is not supported by this run: 45 and 50 digits both scored 30/30.

---

## 2. Multi-Tool Phrasing & Edge Cases (26 Cases, 9 Schemas)

Greedy generation, audited prompts and entities (Section 0).

- **Tool routing**: 21 / 26 (80.8%, CI [62.1, 91.5]%)
- **Exact argument extraction**: 13 / 26 (50.0%, CI [32.1, 67.9]%)

| Category | Result | Notes |
| :--- | :---: | :--- |
| Weather (7) | 4 / 7 exact, 3 routed | Exact: Nairobi, Hanoi, Lima, Bloumplik. Drift: "Lagnes" (Lagos), "Auccra" (Accra), "Rerevan" (Yerevan) |
| Time (3) | 0 / 3 exact, 3 routed | Zone codes drift: "HST" (NZST), "AuctT" (SAST), "AtT" (WAT) |
| Unit Convert (2) | 2 / 2 | |
| Search (5) | 4 / 5 exact | Drift: "Joannes Kepler" |
| Calculator (3) | 2 / 3 exact | "37 * 13" for "37 * 113" (dropped digit) |
| Nested DB Query | 0 / 1 | Model emitted prose ("I will invoke database_query with a sested filter") and no `<CALL>` |
| Fetch API | 0 / 1 | Misrouted to `search`, garbled query |
| Create Event | 0 / 1 | Misrouted to `search`, garbled query |
| File Manager | 0 / 1 | Misrouted to `search`, garbled query |
| Direct Chat (2) | 1 / 2 | "Hello, how are you today?" triggers `get_weather(city="Toro")` |

---

## 3. Zero-Shot OOD Copying (18 Fresh Prompts)

- **Tool routing**: 18 / 18 (100.0%, CI [82.4, 100.0]%)
- **Headline exact match**: 10 / 18 (55.6%, CI [33.7, 75.4]%)

| Result | Prompts |
| :--- | :--- |
| Exact (10) | bioluminescence, aerogel, stagflation, ceviche, quasars, km-to-miles inversion, calculator `61 * 27`, convolutional networks, harpsichord, art nouveau |
| Routed, drifted (8) | Tbilisi -> "Thilisisi"; WAT -> "ATC"; nihilism -> "niiilism"; Yggdrasil -> "zggdrasil"; Hatshepsut -> "watshepsut"; Santo Domingo -> "Santo"; Ho Chi Minh City -> "Cith Cith"; crispr cas12 -> "crispr cos1212" |

The drift is consistent with copying multi-word and novel strings character by character with occasional substitutions. The `crispr cas12` case shows the digit-embedded failure that was also observed in the earlier report.

---

## 4. Head-to-Head Unseen Zero-Shot Benchmark (30 Cases)

Protocol unchanged from `benchmarks/evaluate_unseen_ood.py`. The 30 expected arguments were verified against the training corpus: **0 overlaps**. Both models were rerun in this session.

| Metric | Pre-SFT base `champion_base.pt` | Champion `champion_final.pt` |
| :--- | :---: | :---: |
| Tool routing | 28 / 30 (93.3%, [78.7, 98.2]%) | 29 / 30 (96.7%, [83.3, 99.4]%) |
| Exact copy given routed | 4 / 28 (14.3%, [5.7, 31.5]%) | 17 / 29 (58.6%, [40.7, 74.5]%) |
| Headline exact match | 4 / 30 (13.3%, [5.3, 29.7]%) | 17 / 30 (56.7%, [39.2, 72.6]%) |
| One-sided binomial vs 50% | p = 1.000 | p = 0.292 (not significant) |

Per-category headline exact match (champion): figures 1/6, cities 2/6, science 5/6, concepts 5/6, nonces 4/6.

Conclusion: the champion is higher than the baseline on headline exact match (17/30 vs 4/30, two-sided Fisher's exact p = 0.0009). It does not reach a significant majority over 50% (headline 56.7%, CI [39.2, 72.6]%).

---

## 5. Limitations

1. **Template overlap is not removed.** Prompt templates (`What is X`, `Weather in X`, `Who was X`) appear throughout the training data. The audit removes entity overlap only. Generalization claims are about novel entities under known templates.
2. **Closed-vocabulary arguments are exempt.** Unit names (`km`, `miles`) and time zones are tool-schema enumerations and are not audited. The time-zone codes used in the phrasing and OOD suites (`NZST`, `SAST`, `WAT`) are outside the training list, but the training data has a fixed 10-zone list, so the codes were not seen as values.
3. **Small per-length samples in arithmetic.** n = 30 gives a 95% CI lower bound of 88.6% at 100% observed accuracy. The 60- and 70-digit results (24/30 each) are reported at n = 30 only.
4. **Teacher-forced scoring is an equivalence, not a generation test.** Free-run agreement was checked only at n = 2 per length, at 20 and 26 digits.
5. **Single training run and single seed** for every number here. Variance across seeds is not measured.
6. **Inference cost.** The coordinate parser runs Python loops inside every forward pass, and `generate` has no KV cache. Free-running generation is therefore slow (about 4 tok/s single-stream on CPU, measured here). This affects throughput claims and is not a model property.

---

## 6. Raw Outputs

| File | Content |
| :--- | :--- |
| `results/champion_clean_phrasing.json` | Phrasing suite |
| `results/champion_clean_ood.json` | OOD suite |
| `results/champion_clean_arithmetic.json` | Teacher-forced arithmetic, 20-70 digits |
| `results/champion_freerun_validation.json` | Free-run check, 20 and 26 digits |
| `results/unseen_ood_champion.json` | Unseen-30, champion |
| `results/unseen_ood_base.json` | Unseen-30, baseline |
| `results/*.log` | Console logs for each run |

Commands:

```
.venv/bin/python benchmarks/evaluate_champion.py --suite arithmetic --samples-per-length 30 --arith-mode teacher_forced --out-json results/champion_clean_arithmetic.json
.venv/bin/python benchmarks/evaluate_champion.py --suite phrasing --out-json results/champion_clean_phrasing.json
.venv/bin/python benchmarks/evaluate_champion.py --suite ood --out-json results/champion_clean_ood.json
.venv/bin/python benchmarks/evaluate_unseen_ood.py --ckpt checkpoints/cbt/cbt_18m_2d_rope_R3_k45_sftv4.pt --out-json results/unseen_ood_champion.json
```
