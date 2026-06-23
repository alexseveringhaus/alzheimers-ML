# ML Research Project: Alzheimer's Disease Classification

A personal project applying machine learning to Alzheimer's disease diagnosis using real-world neuroimaging data from the [ADNI](https://adni.loni.usc.edu/) (Alzheimer's Disease Neuroimaging Initiative) dataset. Built four progressively more complex models to explore how different ML approaches and data modalities (tabular ROI features vs. raw 3D PET scans) affect classification performance.

**Tech stack:** Python · PyTorch · scikit-learn · pandas · NumPy · nibabel · matplotlib

## Models

| Model | Input | Approach |
|-------|-------|----------|
| Logistic Regression + Random Forest | Tabular ROI features | Baseline classifiers, feature importance |
| Feedforward Neural Network | Tabular ROI features | 3-layer MLP with early stopping |
| Longitudinal Logistic Regression | Multi-visit tabular data | Engineered trajectory features (baseline + slope) |
| 2D CNN | Raw PET scan images | Convolutional network on extracted 2D slices |

## Results

All tabular models use a stratified 80/20 train/test split with 5-fold cross-validation on the training portion. CV AUC (mean ± std) reflects generalization across folds; Test AUC is the held-out final evaluation.

| Model | Feature Set | CV AUC | Test AUC | Test Accuracy |
|-------|-------------|--------|----------|---------------|
| Logistic Regression | Amyloid | 0.825 ± 0.084 | 0.982 | 96.0% |
| Logistic Regression | Tau | 0.895 ± 0.077 | 0.992 | 96.0% |
| Logistic Regression | Combined | 0.892 ± 0.061 | 1.000 | 98.7% |
| Random Forest | Amyloid | 0.837 ± 0.081 | 0.966 | 92.0% |
| Random Forest | Tau | 0.863 ± 0.070 | 0.987 | 97.3% |
| Random Forest | Combined | 0.864 ± 0.071 | 0.980 | 94.7% |
| Neural Network | Combined | 0.894 ± 0.060 | 0.997 | 96.0% |
| Longitudinal LR | Baseline tau | 0.797 ± 0.167 | 0.875 | 81.3% |
| Longitudinal LR | Tau slope | 0.698 ± 0.176 | 0.708 | 81.3% |
| Longitudinal LR | Baseline + slope | 0.814 ± 0.128 | 0.854 | 81.3% |

## Project Structure

```
Alzheimers_ML/
├── main.py                         # Unified entrypoint — runs all models or a single one
├── requirements.txt
├── results.ipynb                   # Notebook: results table + plots rendered inline
│
├── data/
│   ├── data_extraction.py          # Shared ETL: loads CSVs, merges on patient ID, creates binary label
│   ├── UCBERKELEY_AMY_6MM.csv      # Amyloid PET measurements (13 cortical ROIs per subject)
│   ├── UCBERKELEY_TAU_6MM.csv      # Tau PET measurements (13 cortical ROIs per subject)
│   ├── DXSUM.csv                   # Diagnosis records (baseline + longitudinal visits)
│   └── Tau_PET_Images/
│       └── nifti_output_with_RID/  # ~600 preprocessed 3D NIfTI scan files
│
├── LogisticRegression/model.py
├── NeuralNet/neural_network.py
├── LongitudinalTauBC/
│   ├── data_extraction.py
│   └── model.py
├── 2D-CNN/
│   ├── image_data.py
│   └── cnn.py
│
├── results/
│   ├── metrics.json                # Saved accuracy, CV AUC, and test AUC for all models
│   └── plots/                      # Feature importance and training curve PNGs
└── checkpoints/
    └── nn_best.pt                  # Best neural network weights (saved during training)
```

## Usage

```bash
python main.py              # run all models and save results
python main.py --model nn   # run a single model (choices: logreg, nn, longitudinal, cnn)
```

Results are written to `results/metrics.json` and plots to `results/plots/`. The neural network checkpoint is saved to `checkpoints/nn_best.pt`.

Individual scripts can also be run directly from the project root:

```bash
python LogisticRegression/model.py
python NeuralNet/neural_network.py
python LongitudinalTauBC/model.py
python 2D-CNN/cnn.py
```

## Model Details

### Logistic Regression + Random Forest

**File:** [LogisticRegression/model.py](LogisticRegression/model.py)

Baseline models used to establish a performance floor and probe which imaging biomarkers are most predictive. The same pipeline runs on three feature sets — amyloid-only, tau-only, and combined — with both a Logistic Regression and a Random Forest, producing six result sets in a single run.

- `StandardScaler` + model composed in a `sklearn.Pipeline` — scaler is fit only on training data within each CV fold, preventing leakage
- Stratified 80/20 train/test split + 5-fold CV on the training portion
- Logistic regression coefficients extracted per feature set as a feature importance proxy, saved as a bar chart to `results/plots/`
- Reports CV AUC, test AUC, test accuracy, and full classification report

---

### Feedforward Neural Network

**File:** [NeuralNet/neural_network.py](NeuralNet/neural_network.py)

3-layer MLP trained on the combined amyloid+tau feature vector. Adds regularization techniques absent from the baseline models.

**Architecture:**
```
Input → Linear(64) → ReLU → Dropout(0.3) → Linear(32) → ReLU → Linear(2)
```

**Engineering decisions:**
- **Dropout (p=0.3)** — randomly zeroes activations during training to prevent co-adaptation of neurons
- **L2 regularization** via Adam's `weight_decay=1e-5` to penalize large weights
- **Early stopping** (patience=30): monitors validation loss each epoch and halts when it stops improving, then restores the best weights — avoids overfitting without fixing a static epoch count
- **5-fold CV** on training data for reliable AUC estimates; final model trained on the full training set
- Best weights saved to `checkpoints/nn_best.pt`; training curve saved to `results/plots/`

---

### Longitudinal Tau Regression

**Files:** [LongitudinalTauBC/data_extraction.py](LongitudinalTauBC/data_extraction.py) · [LongitudinalTauBC/model.py](LongitudinalTauBC/model.py)

Cross-sectional models treat each subject's scan as an independent snapshot. This model uses multi-visit data to engineer **trajectory features** — capturing not just where a subject is, but how fast they're changing — which better reflects disease progression dynamics.

**Feature engineering:**
- For each subject × ROI pair, compute:
  - **Baseline value** — mean measurement at time=0
  - **Accumulation slope** — linear regression coefficient fit across all available timepoints
- Only subjects with ≥3 timepoints are included (ensures slope estimates are reliable)
- Median imputation for remaining NaNs; scaler fit inside a Pipeline to prevent leakage
- Three feature sets evaluated independently: baseline-only, slope-only, and baseline+slope

---

### 2D CNN on PET Image Slices

**Files:** [2D-CNN/image_data.py](2D-CNN/image_data.py) · [2D-CNN/cnn.py](2D-CNN/cnn.py)

Instead of using pre-extracted ROI values, this model learns directly from raw 3D PET scan volumes — no hand-crafted features. The key challenge is converting ~600 high-dimensional volumetric scans into a format suitable for supervised learning.

**Data pipeline (`image_data.py`):**
- Loads NIfTI files using `nibabel`
- Extracts axial slices 30–65 per volume (avoids noisy skull/neck regions), turning ~600 3D scans into ~17,000 2D samples
- Per-slice z-score normalization: `(x - μ) / (σ + 1e-6)`
- **Train/test split is subject-level** (stratified by diagnosis) — slices from the same subject never appear in both train and test, preventing data leakage

**Architecture:**
```
Conv2d(1→16, 3×3) → ReLU → MaxPool2d(2,2)
Conv2d(16→32, 3×3) → ReLU → MaxPool2d(2,2)
Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2,2)
Flatten → Linear(25600→128) → ReLU → Linear(128→2)
```

- GPU-accelerated via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- DataLoader with `batch_size=32` and shuffling for training
- 15 epochs, Adam (lr=1e-4), CrossEntropyLoss

## Setup

```bash
pip install -r requirements.txt
```

Data files are not included in this repository as they are governed by ADNI's data use agreement. Access can be requested at [adni.loni.usc.edu](https://adni.loni.usc.edu/).

## Motivation

Alzheimer's disease affects over 55 million people worldwide and currently has no cure. Early and accurate diagnosis is critical for treatment planning and clinical trial enrollment. This project was driven by a genuine interest and personal connection to the problem - using ML on real clinical imaging data felt like a meaningful way to develop applied ML skills while working on something that I'm passionate about.
