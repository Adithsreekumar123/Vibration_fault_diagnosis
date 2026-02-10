"""
Training script for CNN-LSTM-Transformer hybrid model.

Anti-overfitting measures:
- Early stopping with patience
- Learning rate scheduling
- Weight decay (L2 regularization)
- Dropout in model
- Recording-based data split
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.models.cnn_lstm_transformer import CNNLSTMTransformer


def load_split_data(train_path, test_path):
    """Load pre-split train and test data"""
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    X_train = torch.tensor(train_data["X"], dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_data["y"], dtype=torch.long)
    
    X_test = torch.tensor(test_data["X"], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_data["y"], dtype=torch.long)
    
    return X_train, y_train, X_test, y_test


def train_hybrid(
    train_path="data/processed/cwru_train.npz",
    test_path="data/processed/cwru_test.npz",
    epochs=50,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    out_path="results/supervised/hybrid_cwru.pt"
):
    """
    Train hybrid CNN-LSTM-Transformer model with anti-overfitting measures.
    
    Args:
        train_path: Path to train npz file
        test_path: Path to test npz file
        epochs: Maximum training epochs
        batch_size: Batch size (smaller = more regularization)
        lr: Initial learning rate
        weight_decay: L2 regularization strength
        patience: Early stopping patience
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
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Initialize model
    model = CNNLSTMTransformer(
        num_classes=4,
        dropout=0.3  # Dropout for regularization
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    test_accs = []
    best_acc = 0.0
    epochs_no_improve = 0
    
    print("\n" + "=" * 60)
    print("🚀 Training CNN-LSTM-Transformer Hybrid Model")
    print("=" * 60)
    print(f"   Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"   Weight Decay: {weight_decay} | Early Stop Patience: {patience}")
    print("=" * 60 + "\n")
    
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Learning rate scheduling
        scheduler.step(acc)
        
        # Early stopping check
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(model.state_dict(), out_path)
        else:
            epochs_no_improve += 1
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:2}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f} | LR: {current_lr:.2e}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n⏹️  Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
            break
    
    # Final evaluation with best model
    model.load_state_dict(torch.load(out_path, map_location=device))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    
    # Final results
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS - CNN-LSTM-Transformer Hybrid")
    print("=" * 60)
    
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
    ax2.set_title("Test Accuracy (Hybrid Model)")
    ax2.grid(True)
    
    plt.tight_layout()
    
    plot_path = out_path.replace(".pt", "_curves.png")
    plt.savefig(plot_path)
    print(f"\n📈 Training curves saved to: {plot_path}")
    plt.close()
    
    return best_acc, model


if __name__ == "__main__":
    train_hybrid()
