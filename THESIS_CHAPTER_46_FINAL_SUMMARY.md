# Chapter 4.6 — Robustness and Benchmark Summary (LOBO / LODO)

This repository documents the CNN baseline experiments (ResNet-18) used in the ESPI modal classification thesis.  
The canonical, thesis-wide “source of truth” for dataset definition and final headline results is:

- https://github.com/GeorgeSpy/espi-classification-models_2

This file provides a concise, repository-local summary of what is included here and how to interpret it, based on `thesis_results.json`.

---

## 1. Scope of this repository

This repository focuses on:
- A ResNet-18 CNN baseline trained on unwrapped phase maps (single-channel input).
- Multiple training runs reflecting different configurations and random seeds.
- Robustness validation protocols (LOBO / LODO) recorded across multiple run folders.

This repository is intended to support the CNN benchmark results and associated robustness analysis.  
Hybrid RF and Pattern-only RF results are referenced only for contextual comparison and should be treated as canonical only in the thesis results repository listed above.

---

## 2. Headline CNN benchmark results (from `thesis_results.json`)

The following two benchmark entries are recorded under `best_results`:

### 2.1 Best CNN benchmark run (`cnn_non_averaged`)
- Model: ResNet-18
- Accuracy: **93.76%**
- Macro-F1: **88.11%**
- Weighted-F1: **89.45%**
- Run directory: `resnet18_plus_avg`
- Configuration: `CE + effective + label smoothing 0.05`

### 2.2 Best averaged-data CNN run (`cnn_averaged`)
- Model: ResNet-18
- Accuracy: **91.90%**
- Macro-F1: **77.06%**
- Weighted-F1: **82.34%**
- Run directory: `resnet18_CE_eff_ls005_workers0`
- Configuration: `CE + effective + label smoothing 0.05 + averaged data`

These results are quoted as a CNN benchmark and are not used to redefine the thesis headline Hybrid RF results.

---

## 3. RF contextual comparison (reported here for reference)

For convenience, `thesis_results.json` also records the following reference values:

- Pattern-only RF: **90.15%** accuracy / **69.91%** Macro-F1  
- Hybrid RF: **97.85%** accuracy / **95.15%** Macro-F1

When citing RF results, the thesis results repository remains the canonical reference.

---

## 4. Seed robustness sweep

The repository includes a seed sweep over: **41, 42, 43, 44, 45**.

Recorded aggregate performance:
- Mean accuracy: **92.34%** (std: **1.23**)
- Mean Macro-F1: **78.45%** (std: **2.15**)

This sweep is provided as a robustness check for training stochasticity.

---

## 5. LOBO / LODO robustness notes

The LOBO analysis notes that macro-F1 can become degenerate when held-out test partitions collapse to effectively mono-class sets.  
As recorded in `thesis_results.json`:
- Note: “LOBO experiments show degenerate macro-F1 due to mono-class test sets”
- Solution: “Present-only metrics implemented for fair evaluation”
- Additional reporting: per-class F1 for held-out frequency bands

Per-fold artifacts (metrics, predictions, confusion matrices) are stored in the corresponding run directories.

---

## 6. Statistical test recorded in this repository

A McNemar test is recorded under `statistical_tests.mcnemar` with:
- n01 = 45, n10 = 23, chi2 = 8.34, p-value = 0.0039
- Comparison: **Best CNN non-averaged vs Best CNN averaged**

This test applies only to the comparison stated above and should not be interpreted as a global significance claim across all thesis models unless explicitly documented elsewhere.

---

## 7. Training configuration and outputs (reference)

The training configuration stored in `thesis_results.json` includes:
- Model: ResNet-18, input channels: 1, image size: 256
- Epochs: 60, batch size: 64
- Optimizer: Adam, learning rate: 3e-4, weight decay: 1e-4
- Loss: CrossEntropy + effective weights + label smoothing 0.05
- Weighted sampler, strong augmentation, freeze until layer1
- Early stopping enabled, mixed precision enabled
- Device: CUDA, num_workers: 0

Typical output artifacts per run folder include:
- `best.pt`, `metrics.json`, confusion matrices, predictions (`test_preds.csv`, `y_true.npy`, `y_pred.npy`), and `metrics_summary.csv`

---

## 8. Reproducibility

Reproducibility metadata recorded in `thesis_results.json`:
- Python 3.8.10, PyTorch 2.0.1+cu118, CUDA 11.8
- Deterministic mode enabled
- Platform: Windows 10/11

For exact replication, use the run folder specified by `run_dir` and the corresponding artifacts.