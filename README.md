# Alzheimer's Disease Classification — ML Research Project

A personal project applying machine learning to Alzheimer's disease diagnosis using real-world neuroimaging data from the [ADNI](https://adni.loni.usc.edu/) (Alzheimer's Disease Neuroimaging Initiative) dataset. Built four progressively more complex models to explore how different ML approaches and data modalities (tabular ROI features vs. raw 3D PET scans) affect classification performance.

**Tech stack:** Python · PyTorch · scikit-learn · pandas · NumPy · nibabel · matplotlib

## Models

| Model | Input | Approach |
|-------|-------|----------|
| Logistic Regression + Random Forest | Tabular ROI features | Baseline classifiers, feature importance |
| Feedforward Neural Network | Tabular ROI features | 3-layer MLP with early stopping |
| Longitudinal Logistic Regression | Multi-visit tabular data | Engineered trajectory features (baseline + slope) |
| 2D CNN | Raw PET scan images | Convolutional network on extracted 2D slices |

## Project Structure

```
Alzheimers_ML/
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
└── 2D-CNN/
    ├── image_data.py
    └── cnn.py
```

## Model Details

### Logistic Regression + Random Forest

**File:** [LogisticRegression/model.py](LogisticRegression/model.py)

Baseline models used to establish a performance floor and probe which imaging biomarkers are most predictive. The same pipeline runs on three feature sets — amyloid-only, tau-only, and combined — with both a Logistic Regression and a Random Forest, producing six result sets in a single run.

- `StandardScaler` normalization (required for logistic regression convergence; also applied to RF for consistency)
- Stratified 80/20 train/test split
- Logistic regression coefficients extracted per feature set as a lightweight feature importance proxy
- Reports accuracy, ROC AUC, and full classification report

```bash
python LogisticRegression/model.py
```

---

### Feedforward Neural Network

**File:** [NeuralNet/neural_network.py](NeuralNet/neural_network.py)

3-layer MLP trained on the combined amyloid+tau feature vector. Adds regularization techniques absent from the baseline models.

**Architecture:**
```
Input → Linear(64) → ReLU → Dropout(0.3) → Linear(32) → ReLU → Linear(2)
```

**Engineering decisions:**
- **Dropout (p=0.3)** for regularization — randomly zeroes activations during training to prevent co-adaptation of neurons
- **L2 regularization** via Adam's `weight_decay=1e-5` to penalize large weights
- **Early stopping** (patience=30): monitors validation loss and halts training when it stops improving, then restores the best checkpoint — avoids overfitting without fixing a static epoch count
- Separate validation split carved from training data; test set held out entirely until final evaluation

```bash
python NeuralNet/neural_network.py
```

---

### Longitudinal Tau Regression

**Files:** [LongitudinalTauBC/data_extraction.py](LongitudinalTauBC/data_extraction.py) · [LongitudinalTauBC/model.py](LongitudinalTauBC/model.py)

Cross-sectional models treat each subject's scan as an independent snapshot. This model uses multi-visit data to engineer **trajectory features** — capturing not just where a subject is, but how fast they're changing — which better reflects disease progression dynamics.

**Feature engineering:**
- For each subject × ROI pair, compute:
  - **Baseline value** — mean measurement at time=0
  - **Accumulation slope** — linear regression coefficient fit across all available timepoints
- Only subjects with ≥3 timepoints are included (ensures slope estimates are reliable)
- Median imputation for remaining NaNs
- Three feature sets evaluated: baseline-only, slope-only, baseline+slope

```bash
python LongitudinalTauBC/model.py
```

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

```bash
cd 2D-CNN && python cnn.py
```

## Setup

```bash
pip install torch numpy pandas scikit-learn matplotlib nibabel
```

Data files are not included in this repository as they are governed by ADNI's data use agreement. Access can be requested at [adni.loni.usc.edu](https://adni.loni.usc.edu/).

## Motivation

Alzheimer's disease affects over 55 million people worldwide and currently has no cure. Early and accurate diagnosis is critical for treatment planning and clinical trial enrollment. This project was driven by a genuine interest in the problem — using ML on real clinical imaging data felt like a meaningful way to develop applied ML skills while working on something that matters.
