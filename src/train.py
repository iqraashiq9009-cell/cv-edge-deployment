"""
src/train.py

Baseline model training (Section 3.1) and shared accuracy-evaluation utility (reused for
every variant -- baseline, pruned, quantized, combined -- so accuracy is always measured the
same way on the same fixed test set, per Section 3.4's fairness requirement).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42


def make_dataloaders(train_images, train_labels, test_images, test_labels, batch_size=256):
    # Normalize to [0, 1] and add channel dimension: (N, 28, 28) -> (N, 1, 28, 28)
    X_train = torch.from_numpy(train_images).float().unsqueeze(1) / 255.0
    y_train = torch.from_numpy(train_labels).long()
    X_test = torch.from_numpy(test_images).float().unsqueeze(1) / 255.0
    y_test = torch.from_numpy(test_labels).long()

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    return train_loader, test_loader


def train_baseline(model, train_loader, epochs=4, lr=1e-3, device="cpu"):
    torch.manual_seed(SEED)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        total_loss, n_correct, n_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n_correct += (out.argmax(1) == yb).sum().item()
            n_total += xb.size(0)

        train_loss = total_loss / n_total
        train_acc = n_correct / n_total
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4)})
        print(f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}")

    return history


@torch.no_grad()
def evaluate_accuracy(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    n_correct, n_total = 0, 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        n_correct += (out.argmax(1) == yb).sum().item()
        n_total += xb.size(0)
    return n_correct / n_total


def fine_tune(model, train_loader, epochs=1, lr=1e-4, device="cpu"):
    """Brief recovery fine-tuning pass after pruning (Section 3.2 requirement to report
    accuracy with and without a recovery pass)."""
    torch.manual_seed(SEED)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    return model
