import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib.pyplot as plt

from data.data_extraction import extract_data, amy_rois, tau_rois

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
CKPT_DIR    = Path(__file__).parent.parent / "checkpoints"


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1     = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, 32)
        self.out     = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 400,
    patience: int = 30,
) -> tuple[nn.Module, list[float], list[float], int]:
    """Train one fold with early stopping. Returns best model, loss curves, and stop epoch."""
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val,   dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    y_vl = torch.tensor(y_val,   dtype=torch.long)

    model     = SimpleNeuralNetwork(X_tr.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_loss    = np.inf
    patience_counter = 0
    early_stop_epoch = 0
    best_state       = None
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), y_vl).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state       = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            early_stop_epoch = epoch
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, early_stop_epoch


def save_training_plot(train_losses: list[float], val_losses: list[float], early_stop_epoch: int) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Validation Loss")
    if early_stop_epoch > 0:
        plt.axvline(early_stop_epoch, color="red", linestyle="--", label="Early Stop")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Neural Network — Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "nn_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def model() -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    _, _, combo = extract_data()

    X_raw = pd.concat([
        combo[[col + "_x" for col in amy_rois]],
        combo[[col + "_y" for col in tau_rois]],
    ], axis=1).values
    y = combo["y"].values

    # Held-out test set — scaler fit only on train data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5-fold CV on training portion
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs: list[float] = []

    for fold_train_idx, fold_val_idx in cv.split(X_train_full, y_train_full):
        X_fold_tr, X_fold_vl = X_train_full[fold_train_idx], X_train_full[fold_val_idx]
        y_fold_tr, y_fold_vl = y_train_full[fold_train_idx], y_train_full[fold_val_idx]

        scaler = StandardScaler()
        X_fold_tr = scaler.fit_transform(X_fold_tr)
        X_fold_vl = scaler.transform(X_fold_vl)

        fold_model, _, _, _ = train_fold(X_fold_tr, y_fold_tr, X_fold_vl, y_fold_vl)

        fold_model.eval()
        with torch.no_grad():
            probs = torch.softmax(
                fold_model(torch.tensor(X_fold_vl, dtype=torch.float32)), dim=1
            )[:, 1].numpy()
        cv_aucs.append(roc_auc_score(y_fold_vl, probs))

    # Final model trained on all training data, evaluated on held-out test set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled  = scaler.transform(X_test)

    # Use a validation split from training data only for early stopping on the final model
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_train_scaled, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    final_model, train_losses, val_losses, early_stop_epoch = train_fold(X_tr, y_tr, X_vl, y_vl)

    final_model.eval()
    with torch.no_grad():
        test_logits = final_model(torch.tensor(X_test_scaled, dtype=torch.float32))
        probs = torch.softmax(test_logits, dim=1)[:, 1].numpy()
        preds = torch.argmax(test_logits, dim=1).numpy()

    accuracy = accuracy_score(y_test, preds)
    test_auc = roc_auc_score(y_test, probs)

    save_training_plot(train_losses, val_losses, early_stop_epoch)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), CKPT_DIR / "nn_best.pt")

    print(f"\nNeural Network Results:")
    print(f"  CV AUC:        {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test AUC:      {test_auc:.4f}")

    return {
        "NeuralNetwork": {
            "feature_set":   "combined",
            "cv_auc_mean":   round(float(np.mean(cv_aucs)), 4),
            "cv_auc_std":    round(float(np.std(cv_aucs)), 4),
            "test_accuracy": round(float(accuracy), 4),
            "test_auc":      round(float(test_auc), 4),
        }
    }


if __name__ == "__main__":
    model()
