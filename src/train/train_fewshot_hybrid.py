"""
Few-shot training for Hybrid (CNN-LSTM-Transformer) model.

Fine-tunes the pre-trained hybrid model on a small percentage of
Paderborn labels to improve cross-domain accuracy.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.models.cnn_lstm_transformer import CNNLSTMTransformer


def create_fewshot_split(X, y, fewshot_pct=0.05, random_state=42):
    """Split data into few-shot train and test sets."""
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=fewshot_pct, 
        stratify=y, 
        random_state=random_state
    )
    return X_train, y_train, X_test, y_test


def train_hybrid_fewshot(
    pretrained_path="results/supervised/hybrid_cwru.pt",
    paderborn_path="data/processed/paderborn_windows.npz",
    fewshot_pct=0.05,
    epochs=30,
    batch_size=32,
    lr=1e-4,  # Lower LR for fine-tuning
    out_path=None
):
    """
    Fine-tune pre-trained hybrid model on Paderborn few-shot data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Set output path
    if out_path is None:
        pct_str = f"{int(fewshot_pct * 100)}pct"
        out_path = f"results/supervised/hybrid_fewshot_{pct_str}.pt"
    
    # Load Paderborn data
    print(f"\n📂 Loading Paderborn data...")
    data = np.load(paderborn_path)
    X_all, y_all = data["X"], data["y"]
    
    # Create few-shot split
    X_train, y_train, X_test, y_test = create_fewshot_split(
        X_all, y_all, fewshot_pct=fewshot_pct
    )
    
    print(f"   Few-shot train: {len(X_train)} samples ({fewshot_pct:.0%})")
    print(f"   Test: {len(X_test)} samples")
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create dataloaders
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Load pre-trained model
    print(f"\n📥 Loading pre-trained hybrid model...")
    model = CNNLSTMTransformer(num_classes=4, dropout=0.3).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    # Freeze early layers, fine-tune later layers
    # (Optional: can be enabled for more stable training)
    # for param in model.cnn.parameters():
    #     param.requires_grad = False
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\n{'='*50}")
    print(f"🚀 HYBRID + FEW-SHOT ({fewshot_pct:.0%} Paderborn)")
    print(f"{'='*50}")
    print(f"   Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"{'='*50}\n")
    
    best_acc = 0.0
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
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
        
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(model.state_dict(), out_path)
        
        print(f"Epoch [{epoch:2}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {acc:.2%}")
    
    # Final evaluation
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
    
    print(f"\n{'='*50}")
    print(f"📊 FINAL RESULTS - HYBRID + {fewshot_pct:.0%} FEW-SHOT")
    print(f"{'='*50}")
    print(f"\n✅ Best Accuracy: {best_acc:.2%}")
    print(f"💾 Model saved: {out_path}")
    
    class_names = ["Normal", "Ball", "Inner", "Outer"]
    print(f"\n📋 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return best_acc, model


if __name__ == "__main__":
    # Train 5% few-shot
    train_hybrid_fewshot(fewshot_pct=0.05)
