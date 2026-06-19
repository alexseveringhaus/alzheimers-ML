import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))       # local image_data module
sys.path.insert(0, str(Path(__file__).parent.parent)) # project root

from sklearn.model_selection import train_test_split
from image_data import extract, TauSliceDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam


class TauCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 20 * 20, 128)
        self.fc2   = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main() -> None:
    df       = extract()
    subjects = df["PTID"].unique()

    # Split at subject level to prevent slice-level data leakage
    train_ids, test_ids = train_test_split(
        subjects,
        test_size=0.2,
        stratify=df.groupby("PTID")["label"].first(),
        random_state=42,
    )

    train_loader = DataLoader(TauSliceDataset(df[df["PTID"].isin(train_ids)]),  batch_size=32, shuffle=True)
    test_loader  = DataLoader(TauSliceDataset(df[df["PTID"].isin(test_ids)]),   batch_size=32, shuffle=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = TauCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(15):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y   = X.to(device), y.to(device)
            preds  = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    print(f"Slice-level test accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
