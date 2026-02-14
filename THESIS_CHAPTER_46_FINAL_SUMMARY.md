# Chapter 4.6 — Robustness Evaluation Summary (LOBO / LODO)

This repository contains the CNN experiments used as a benchmark in the ESPI modal classification thesis.  
The primary “source-of-truth” for the overall thesis results (Hybrid RF, Pattern RF, dataset specification) is maintained in the repository:
- `https://github.com/GeorgeSpy/espi-classification-models_2`

This file summarizes what is included here, and how it should be interpreted.

---

## 1. What this repository covers

- A ResNet-18 baseline for modal classification on unwrapped phase maps.
- Multiple runs reflecting different training configurations (e.g., loss functions, label smoothing, balancing, and seeds).
- Aggregated results and protocol notes in `thesis_results.json`.

This repository should be treated as the reference for the CNN benchmark only, not as the canonical location for Hybrid RF results.

---

## 2. Dataset versions and evaluation protocols

Two dataset “universes” appear in the archived CNN runs:

### 2.1 Final verified dataset (used for thesis headline reporting)
- Verified classification set: **N = 3,443**
- Canonical split: **70 / 15 / 15** (Train / Val / Test)

The thesis headline CNN benchmark is reported from the run referenced in `thesis_results.json`.

### 2.2 Legacy dataset (kept for traceability)
Some older CNN runs were executed before the final dataset cleaning step (e.g., total N around 4,446).  
These runs are retained for reproducibility of intermediate development, but they are not used in the thesis headline tables.

---

## 3. CNN benchmark results (single split)

The CNN benchmark numbers cited in the thesis correspond to the “final verified dataset” runs summarized in `thesis_results.json`.

- **ResNet-18 (benchmark run):** Accuracy **93.76%**, Macro-F1 **88.11%**
- **Legacy run example:** Accuracy **91.90%**, Macro-F1 **77.06%** (different dataset version)

For exact run identifiers, configuration, and support values, refer to:
- `thesis_results.json`
- the per-run `metrics.json` / `split.json` artifacts in the corresponding run folders

---

## 4. LOBO / LODO robustness evaluation

Robustness protocols are evaluated across multiple training runs (one model per held-out partition):
- LOBO: leave one frequency band out
- LODO: leave one domain/dataset out

Aggregated summaries are reported in `thesis_results.json` and the thesis document.  
Per-fold artifacts remain available in the run folders for inspection.

---

## 5. Notes on statistical tests

Any statistical tests recorded in this repository are specific to the comparisons described in `thesis_results.json`.  
They should not be interpreted as global significance claims across all thesis models unless explicitly documented in the thesis results repository.

---

## 6. Reproducibility

To reproduce the CNN benchmark:
1. Use the environment and dependencies described in this repository.
2. Select the benchmark run referenced in `thesis_results.json`.
3. Verify the dataset version (final verified vs legacy) using the run’s `split.json`.
4. Recompute metrics from predictions if needed using the scripts in this repository.

---