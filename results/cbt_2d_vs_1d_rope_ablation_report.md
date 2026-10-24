# Position-Encoding Ablation: 2D-RoPE vs 1D-RoPE (CBT-18.95M, Curriculum V5)

**Evaluation Date**: 2026-10-24
**2D model**: `champion_model/checkpoints/champion_v3_final.zip` (v3; full results in `results/champion_v3_performance_report.md`)
**1D model**: `champion_model/checkpoints/1d/cbt_18m_1d_rope_R3_k45_sftv5.zip` (`torch.save` file; identical weights to `_best.pt`)
**Decoding**: Greedy (temperature = 0, argmax). **Hardware**: CPU for all local evaluations; Kaggle 2x T4 for training.

**Question**: does decoupled 2D-RoPE (place p / step tau coordinates) cause the length generalization of the CBT champion, or would standard 1D-RoPE do the same with the same architecture, data and training recipe?

**Answer**: the coordinates make the difference. With identical V5 data, the 1D twin scores 0/270 on every unseen arithmetic length (26-70 digits). The 2D model scores 254/270. The two models are equal on tool use.

---

## 1. Setup

| | 2D lineage (v3) | 1D twin |
| :--- | :--- | :--- |
| Architecture | ScaledGPT, 6 layers x R=3, d=512, 8 heads, block 1024 | identical |
| Parameters | 18,944,512 | 18,944,512 |
| Position encoding | Decoupled 2D-RoPE: half of head_dim rotated by place p, half by step tau | Standard RoPE over the full head_dim at positions 0..T-1 |
| Shift augmentation (k) | Random shift of p > 0 during training | None (a uniform offset is an exact no-op under RoPE; proved by `TestRope1DAblation`) |
| Pretrain | 50,000-step recipe; the session stopped at step 45,059 (`scaled_t4_final_timeout_step_45059`) | Same recipe, all 50,000 steps completed |
| Search-copy SFT (7k) + all-tools SFT (4k) | yes | skipped (code superseded, not replayable; neither stage contains arithmetic) |
| V4 SFT | 12,000 steps, LR 5e-5 | identical |
| V5 SFT | 23,927 steps (6.0 h time budget) | 23,927 steps (step-matched; 2.7 h) |
| V5 training data | `champion_v3_final_corpus.txt.gz` | Byte-identical after decompression (SHA-256 of the decompressed text matches) |

The treatment is the full 2D coordinate scheme, including its shift augmentation. That augmentation has no 1D equivalent, so this ablation cannot separate the two.

---

## 2. Results

Both models were scored with the same harness, seeds and problems. Fisher p values are two-sided exact tests.

### 2.1 Arithmetic (teacher-forced, 30 problems per length, seed 42 + digits)

| Digits | Status | 2D (v3) | 1D twin | Fisher p |
| :---: | :--- | :---: | :---: | :---: |
| 20 | Seen | 30 / 30 | 28 / 30 | 0.49 |
| 25 | Seen | 30 / 30 | 29 / 30 | 1.0 |
| 26 | Unseen (+1) | 30 / 30 | **0 / 30** | 1.7e-17 |
| 30 | Unseen (+5) | 30 / 30 | **0 / 30** | |
| 35 | Unseen (+10) | 30 / 30 | **0 / 30** | |
| 40 | Unseen (+15) | 30 / 30 | **0 / 30** | |
| 45 | Unseen (+20) | 30 / 30 | **0 / 30** | |
| 50 | Unseen (+25) | 30 / 30 | **0 / 30** | |
| 55 | Unseen (+30) | 30 / 30 | **0 / 30** | |
| 60 | Unseen (+35) | 25 / 30 | **0 / 30** | |
| 70 | Unseen (+45) | 19 / 30 | **0 / 30** | |
| **Seen (20, 25)** | | **60 / 60** [94.0, 100.0]% | **57 / 60** [86.3, 98.3]% | 0.24 |
| **Unseen 26-55** | | **210 / 210** [98.2, 100.0]% | **0 / 210** [0.0, 1.8]% | 1.9e-125 |
| **Unseen 26-70** | | **254 / 270** [90.6, 96.3]% | **0 / 270** [0.0, 1.4]% | 1.0e-135 |

The 1D twin goes from 29/30 at 25 digits to 0/30 at 26 digits, one digit past the training maximum. It fails every problem type at 26 digits, including the three cascades. The training-time held-out check agrees: 1D V5 solved only the 25-digit problem of four (25%, every checkpoint from step 12,000 on), and 2D held 100%.

### 2.2 Where the 1D twin fails

First wrong token under teacher forcing (`results/cbt1d_v5_arith_failure_analysis.json`):

| Problem type | First error | Example |
| :--- | :--- | :--- |
| Uniform / dense (124 failures at 26, 30, 40, 55, 70 digits) | Column 1 (93), 2 (29) or 3 (2): reading an operand digit at the start of the scratchpad | `...=<THINK>(0+4+1)=05(0+` -> gold `3`, predicted `7` |
| Cascade (26 failures, same lengths) | Column 22-27, always `0` written for an operand `9`: the model stops reading the operand near the 25-digit training maximum | `...=10(1+9+0)=10(1+` -> gold `9`, predicted `0` |

Compare the 2D model (`results/champion_v3_arith_failure_analysis.json`): it has no error in columns 1-58 on any problem, and its failures are local carry errors at columns 59-70 of 60/70-digit uniform problems.

Interpretation, limited to what the data shows:
- **1D does not degrade gradually. It fails from the first column.** On longer operands, it mis-reads a digit in the first one to three columns. With 1D positions, the distance from a scratchpad column back to the operand digit it needs grows with operand length. At 26+ digits that distance is outside the training range. 2D place coordinates index digits by place value, so that lookup does not depend on operand length. This is consistent with the result, but the analysis did not find a single consistent wrong-place offset in the mis-read digits, so the exact lookup error is not established.
- **1D cascades fail near the training boundary** (columns 22-27). The model behaves as if the operand of 9s ends at about 25 digits. This is a learned length limit, not a decay of the carry state.

### 2.3 Tool use

| Suite | 2D (v3) | 1D twin | Fisher p |
| :--- | :---: | :---: | :---: |
| Phrasing routing (26) | 26 / 26 | 25 / 26 [81.1, 99.3]% | 1.0 |
| Phrasing exact (26) | 26 / 26 | 25 / 26 [81.1, 99.3]% | 1.0 |
| OOD routing (18) | 18 / 18 | 18 / 18 | 1.0 |
| OOD exact (18) | 18 / 18 | 18 / 18 [82.4, 100.0]% | 1.0 |
| Unseen-30 routing | 30 / 30 | 30 / 30 | 1.0 |
| Unseen-30 exact | 30 / 30 | 30 / 30 [88.6, 100.0]% | 1.0 |

The single 1D failure is Fetch API. The model writes `<CALL>fetch_api(endpoint="/v3/telemetry", params={"region": "APAC", "metric": "thite": ...`: it picks the right tool and endpoint, then the `params` dict breaks and the call never closes. Tool routing and argument copying do not depend on the position-encoding choice at this sample size.

### 2.4 1D stage progression (Kaggle fast benchmarks, free-run, separate problem sampler)

| Stage | Phrasing exact | OOD exact | Unseen-30 exact | Arithmetic 16 / 20 / 25 / 30-55 |
| :--- | :---: | :---: | :---: | :--- |
| Pretrain | 17 / 26 | 13 / 18 | 26 / 30 | 0 / 0 / 0 / 0 |
| V4 SFT | 17 / 26 | 14 / 18 | 25 / 30 | 29 / 26 / 0 / 0 |
| V5 SFT | 25 / 26 | 18 / 18 | 30 / 30 | 30 / 30 / 26 / 0 |

These numbers come from `benchmarks/evaluate_random_16_to_100d.py`, which samples different problems from the teacher-forced suite (25 digits: 26/30 here vs 29/30 in Section 2.1). The pretrain arithmetic score of 0 is not compared with 2D, because the 2D pretrain checkpoint was not scored with this tool.

---

## 3. Limitations

1. **Single seed per model.** The difference in unseen arithmetic (254/270 vs 0/270) is far too large for seed variance to explain. The small differences at seen lengths and on tool use are within noise.
2. **The lineages are not identical before V4.** The 2D model had two extra tool SFT stages (11k steps, no arithmetic), and its pretraining stopped at step 45,059 of 50,000. The 1D twin completed all 50,000 pretraining steps. Neither difference gives 2D more arithmetic training.
3. **Shift and coordinates are confounded** (Section 1). The result shows that "2D-RoPE with shift augmentation" generalizes and "1D-RoPE" does not. It does not show which component is necessary.
4. **The mechanism is inferred.** The failure localization shows where the 1D twin fails (first operand reads; cascades at the training boundary). It does not test the explanation directly, for example with attention maps.

---

## 4. Raw Outputs and Commands

| File | Content |
| :--- | :--- |
| `results/cbt1d_v5_arithmetic.json`, `results/cbt1d_v5_arith.log` | Teacher-forced arithmetic, 20-70 digits (local) |
| `results/cbt1d_v5_arith_failure_analysis.json` / `.log` | First-error localization at 20-70 digits |
| `results/cbt1d_v5_phrasing_local.json`, `results/cbt1d_v5_phrasing.log` | Phrasing suite (local rerun, matches Kaggle) |
| `results/cbt1d_v5_phrasing.json`, `results/cbt1d_v5_ood.json`, `results/cbt1d_v5_unseen_ood.json`, `results/cbt1d_v5_arith_16_55d.json` | Kaggle fast benchmarks on V5 |
| `results/cbt1d_v5_sft_train.log` | V5 SFT training log (held-out history) |
| `champion_model/checkpoints/1d/cbt_1d_ablation_outputs.zip`, `cbt_1d_ablation_outputsv5.zip` | All Kaggle outputs (pretrain, V4, V5 checkpoints, logs, stage benchmarks) |

```
CKPT=champion_model/checkpoints/1d/cbt_18m_1d_rope_R3_k45_sftv5.zip
.venv/bin/python benchmarks/evaluate_champion.py --ckpt $CKPT --suite arithmetic --samples-per-length 30 --arith-mode teacher_forced --device cpu --out-json results/cbt1d_v5_arithmetic.json
.venv/bin/python benchmarks/evaluate_champion.py --ckpt $CKPT --suite phrasing --device cpu --out-json results/cbt1d_v5_phrasing_local.json
```

Training: `notebooks/kaggle_cbt_1d_ablation.ipynb` (Stages 1-3, `--pos-emb rope_1d`).
