import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent)) # Ensure project root is in path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib.pyplot as plt

from data.data_extraction import extract_data, amy_rois, tau_rois

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.3) # Dropout layer for regularization/prevent overfitting (some neurons randomly turned off during training)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2) # Binary classification (AD vs. CN)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

def plot_training(epochs, losses):
    plt.plot(range(epochs), losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Training Loss over Epochs')
    plt.show()

def plot_validation(train_losses, val_losses, early_stop_epoch):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    if early_stop_epoch > 0: # only plot if early stopping occurred
        plt.axvline(early_stop_epoch, color="red", linestyle="--", label="Early Stop")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

if __name__ == "__main__":
    # Ensure outputs can be reproduced
    torch.manual_seed(42)
    np.random.seed(42)

    amy, tau, combo = extract_data()

    # Input features and target variable
    X_amy = combo[[col + '_x' for col in amy_rois]]  
    X_tau = combo[[col + '_y' for col in tau_rois]]
    X_combo = pd.concat([X_amy, X_tau], axis=1)
    y = combo['y'].values

    # Feature scaling - ensures all input features contribute equally (more stable gradient descent)
    scaler = StandardScaler()
    X_combo = scaler.fit_transform(X_combo)

    # Split into training, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X_combo, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)
    
    # Model, loss, optimizer
    model = SimpleNeuralNetwork(X_combo.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001) # Use weight decay for L2 regularization (avoids overfitting/over-reliance on large weights)
    
    # Training loop
    epochs = 400 # Number of times running through the entire dataset
    best_val_loss = np.inf
    patience = 30 # Number of epochs to wait for improvement before stopping
    patience_counter = 0 # Current counter
    early_stop_epoch = 0 # Used in plotting

    # Used for plotting training/validation loss curves
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        
        # Train 
        model.train()
        optimizer.zero_grad() # Reset gradients

        logits = model(X_train) # Forward pass
        loss = criterion(logits, y_train) # Measure loss between predictions and true labels

        loss.backward() # Back propagation - take error rate of forward prop, feed it back through the network to update weights
        optimizer.step()

        train_losses.append(loss.item()) # For plotting

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val) # Validation forward pass
            val_loss = criterion(val_logits, y_val)
            val_losses.append(val_loss.item()) # For plotting
        
        # Check for early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1 # No improvement this epoch

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            early_stop_epoch = epoch
            break
    
    # Restore best model
    model.load_state_dict(best_state)
    
    # Testing
    model.eval()
    with torch.no_grad(): # No backpropagation during testing
        test_logits = model(X_test)
        probs = torch.softmax(test_logits, dim=1)[:, 1].numpy()
        preds = torch.argmax(test_logits, dim=1).numpy()

    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\nNeural Network Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC : {auc:.3f}")
    plot_validation(train_losses, val_losses, early_stop_epoch)