"""
Supervised training using pre-split CWRU data.

Uses recording-based train/test split files to prevent data leakage.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.models.cnn_classifier import CNNClassifier


def load_split_data(train_path, test_path):
    """Load pre-split train and test data"""
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    X_train = torch.tensor(train_data["X"], dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_data["y"], dtype=torch.long)
    
    X_test = torch.tensor(test_data["X"], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_data["y"], dtype=torch.long)
    
    return X_train, y_train, X_test, y_test


def train_supervised_split(
    train_path="data/processed/cwru_train.npz",
    test_path="data/processed/cwru_test.npz",
    ssl_ckpt=None,  # Optional: path to SSL pretrained weights
    epochs=30,
    batch_size=128,
    lr=1e-3,
    out_path="results/supervised/cnn_cwru_split.pt"
):
    """
    Train CNN classifier on recording-split data.
    
    Args:
        train_path: Path to train npz file
        test_path: Path to test npz file
        ssl_ckpt: Optional path to SSL pretrained encoder weights
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        out_path: Path to save trained model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load pre-split data
    print("\n📂 Loading pre-split data...")
    X_train, y_train, X_test, y_test = load_split_data(train_path, test_path)
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Create dataloaders
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Initialize model
    model = CNNClassifier(num_classes=4).to(device)
    
    # Optionally load SSL pretrained weights
    if ssl_ckpt and os.path.exists(ssl_ckpt):
        model.encoder.load_state_dict(torch.load(ssl_ckpt, map_location=device))
        print(f"✅ Loaded SSL pretrained encoder from: {ssl_ckpt}")
    else:
        print("🔸 Training from scratch (no SSL pretraining)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    test_accs = []
    best_acc = 0.0
    
    print("\n" + "=" * 50)
    print("🚀 Starting Training (Recording-Split)")
    print("=" * 50)
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())
        
        acc = accuracy_score(y_true, y_pred)
        test_accs.append(acc)
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(model.state_dict(), out_path)
        
        print(f"Epoch [{epoch:2}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS (Recording-Split Evaluation)")
    print("=" * 50)
    
    print(f"\n✅ Best Test Accuracy: {best_acc:.4f}")
    print(f"💾 Model saved to: {out_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Normal", "Ball", "Inner_Race", "Outer_Race"]
    
    print("\n📋 Confusion Matrix:")
    print("                 Predicted")
    print("                 " + "  ".join([f"{c[:4]:>5}" for c in class_names]))
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:12} {row}")
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True)
    
    ax2.plot(test_accs)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Test Accuracy (Recording-Split)")
    ax2.grid(True)
    
    plt.tight_layout()
    
    plot_path = out_path.replace(".pt", "_curves.png")
    plt.savefig(plot_path)
    print(f"\n📈 Training curves saved to: {plot_path}")
    plt.close()
    
    return best_acc, model


if __name__ == "__main__":
    train_supervised_split()
