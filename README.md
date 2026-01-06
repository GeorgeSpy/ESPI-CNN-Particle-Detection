# ESPI-CNN-Particle-Detection

A comprehensive Deep Learning framework for classifying modal patterns in Electronic Speckle Pattern Interferometry (ESPI) images. This repository contains the training and evaluation scripts used for the thesis **"A Deep Learning Approach for Modal Pattern Recognition in ESPI"**.

## 🧠 Models & Strategies

The project implements two main architectures:
*   **SimpleCNN**: A lightweight, custom CNN (≈1.2M params) trained from scratch.
*   **ResNet18**: Transfer learning from ImageNet, adapted for 1-channel grayscale input.

It supports advanced training strategies to handle class imbalance and data scarcity:
*   **LOBO (Leave-One-Band-Out)**: Tests generalization to unseen frequency ranges.
*   **LODO (Leave-One-Dataset-Out)**: Tests generalization to unseen experimental boards (W01/W02/W03).
*   **MC-LOBO**: Monte Carlo cross-validation with per-class frequency gaps.
*   **Imbalance Handling**: Weighted Random Sampling, Focal Loss, and Class-Weighted Cross Entropy.

## 📂 Repository Structure

*   `train_cnn.py`: **Main Training Script.** Standard training loop with support for Stratified, LOBO, and LODO splits.
*   `train_cnn_mc.py`: **Monte Carlo Training.** Runs multiple random seed variations for robust statistical evaluation.
*   `train_cnn_mclobo.py`: **MC-LOBO Training.** Specialized script for Monte Carlo evaluation with specific "Left-Out" frequency gaps per class.
*   `requirements.txt`: Python dependencies.

## 🛠 Installation

```bash
pip install -r requirements.txt
```

*(Requires Python 3.8+, PyTorch, and CUDA for GPU acceleration)*

## 🚀 Usage

### 1. Standard Training (SimpleCNN)

```bash
python train_cnn.py \
    --labels_csv data/labels.csv \
    --run_dir runs/simple_baseline \
    --model simple \
    --epochs 60 \
    --batch_size 64 \
    --augment strong
```

### 2. ResNet18 Training

```bash
python train_cnn.py \
    --labels_csv data/labels.csv \
    --run_dir runs/resnet18_baseline \
    --model resnet18 \
    --lr 3e-4 \
    --freeze_until layer2 \
    --epochs 50
```

### 3. LOBO Evaluation (Leave 500-525 Hz out)

```bash
python train_cnn.py \
    --labels_csv data/labels.csv \
    --run_dir runs/lobo_500_525 \
    --lobo_band 500 525 \
    --epochs 50
```

### 4. LODO Evaluation (Leave Board W02 out)

```bash
python train_cnn.py \
    --labels_csv data/labels.csv \
    --run_dir runs/lodo_W02 \
    --lodo_holdout W02 \
    --epochs 50
```

## 📊 Data Format

The scripts expect a CSV file (`--labels_csv`) with at least:
*   **path**: Absolute or relative path to the image (16-bit PNG or .npy).
*   **label**: Integer label (0-5).

*Optional columns for advanced splits:*
*   **freq_hz**: Frequency in Hz (required for LOBO).
*   **dataset_id**: Experiment ID like 'W01', 'W02' (required for LODO).

## 📄 License

Academic/Research License. Please cite appropriately if used.
