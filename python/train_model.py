import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ONNX_PATH = os.path.join(MODEL_DIR, "ik_model.onnx")

EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PATIENCE = 15


class IKModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    def to_tensors(df):
        X = torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32)
        Y = torch.tensor(df[["t1", "t2", "t3"]].values, dtype=torch.float32)
        return X, Y

    X_train, Y_train = to_tensors(train_df)
    X_val, Y_val = to_tensors(val_df)
    X_test, Y_test = to_tensors(test_df)

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val),
                            batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, Y_test),
                             batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} "
                      f"(best val_loss={best_val_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            test_loss += loss.item() * len(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    angle_errors = np.abs(preds - targets)
    mean_angle_err = angle_errors.mean()
    max_angle_err = angle_errors.max()

    print(f"\n=== Test Results ===")
    print(f"  Test MSE loss:       {test_loss:.6f}")
    print(f"  Mean angle error:    {mean_angle_err:.4f} rad ({np.degrees(mean_angle_err):.2f}°)")
    print(f"  Max angle error:     {max_angle_err:.4f} rad ({np.degrees(max_angle_err):.2f}°)")

    return test_loss


def export_onnx(model, path):
    model.eval()
    dummy = torch.randn(1, 3)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    print(f"\nModel exported to: {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, test_loader = load_data()
    print(f"Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    model = IKModel().to(device)
    print(f"\nModel:\n{model}\n")

    print("Training...")
    model = train(model, train_loader, val_loader, device)

    evaluate(model, test_loader, device)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ik_model.pth"))
    print(f"PyTorch model saved to: {MODEL_DIR}/ik_model.pth")

    export_onnx(model, ONNX_PATH)


if __name__ == "__main__":
    main()
