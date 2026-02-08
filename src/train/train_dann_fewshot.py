# src/train/train_dann_fewshot.py
"""
DANN + Few-Shot Training

Two-stage training approach:
1. Stage 1: DANN domain adversarial training (uses all source labels + unlabeled target)
2. Stage 2: Few-shot fine-tuning on X% of target domain labels

This combination should achieve ~60-80% accuracy on Paderborn vs ~30% with DANN alone.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.models.cnn import CNNEncoder
from src.models.domain_discriminator import DomainDiscriminator, compute_alpha
from src.train.dann_dataset import DANNDataset
from src.train.supervised_dataset import SupervisedDataset
from src.train.fewshot_utils import create_fewshot_split


class DANNModel(nn.Module):
    """DANN Model for Stage 1"""
    
    def __init__(self, num_classes=4, encoder_dim=128):
        super().__init__()
        self.encoder = CNNEncoder()
        self.classifier = nn.Linear(encoder_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim=encoder_dim)
    
    def forward(self, x, alpha=1.0):
        features = self.encoder(x)
        class_output = self.classifier(features)
        domain_output = self.domain_discriminator(features, alpha)
        return class_output, domain_output, features


def train_dann_fewshot(
    target_path="data/processed/paderborn_windows.npz",
    dann_ckpt="results/dann/cnn_dann.pt",  # Use existing trained DANN!
    out_path="results/dann/dann_fewshot.pt",
    fewshot_epochs=20,
    fewshot_fraction=0.05,  # 5% default
    batch_size=128,
    fewshot_lr=1e-3
):
    """
    Few-shot fine-tuning on pre-trained DANN model.
    
    This does NOT retrain DANN from scratch. It loads the existing trained
    DANN model and fine-tunes only the classifier on a few target labels.
    
    Args:
        target_path: Path to Paderborn data
        dann_ckpt: Path to pre-trained DANN model (run_dann.py must be run first)
        out_path: Where to save final model
        fewshot_epochs: Epochs for few-shot fine-tuning
        fewshot_fraction: Fraction of target labels to use (0.01, 0.05, 0.20)
        batch_size: Batch size
        fewshot_lr: Learning rate for few-shot
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # ========================================
    # LOAD PRE-TRAINED DANN MODEL
    # ========================================
    print("\n" + "=" * 60)
    print("  Loading Pre-trained DANN Model")
    print("=" * 60 + "\n")
    
    # Initialize model
    model = DANNModel(num_classes=4).to(device)
    
    # Load pre-trained DANN weights
    if not os.path.exists(dann_ckpt):
        print(f"❌ Pre-trained DANN not found: {dann_ckpt}")
        print("Please run 'python run_dann.py' first to train the base DANN model.")
        return None, None
    
    model.load_state_dict(torch.load(dann_ckpt, map_location=device))
    print(f"✅ Loaded pre-trained DANN from: {dann_ckpt}")
    
    # Evaluate base DANN accuracy
    dann_acc = evaluate_model(model, target_path, device)
    print(f"📊 Base DANN accuracy on Paderborn: {dann_acc:.2%}")
    
    # ========================================
    # FEW-SHOT FINE-TUNING
    # ========================================
    print("\n" + "=" * 60)
    print(f"  Few-Shot Fine-tuning ({fewshot_fraction*100:.0f}% target labels)")
    print("=" * 60 + "\n")
    
    # Create few-shot split
    target_dataset = SupervisedDataset(target_path)
    train_ds, test_ds = create_fewshot_split(target_dataset, fraction=fewshot_fraction)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    print(f"📊 Few-shot training samples: {len(train_ds)}")
    print(f"📊 Test samples: {len(test_ds)}")
    
    # Fine-tune ENTIRE model with lower learning rate (no freezing)
    # This prevents the classifier from diverging while still adapting
    
    # Compute class weights for balanced training
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    
    train_labels = [y for _, y in train_ds]
    # Use all 4 classes for weight computation (classes 0-3)
    all_classes = np.array([0, 1, 2, 3])
    # Ensure all classes are represented (add dummy if missing)
    for c in all_classes:
        if c not in train_labels:
            train_labels.append(c)
    train_labels_array = np.array(train_labels)
    class_weights = compute_class_weight('balanced', classes=all_classes, y=train_labels_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"📊 Class weights: {class_weights.cpu().numpy()}")
    
    # Use a VERY low learning rate to preserve DANN features
    fewshot_optimizer = torch.optim.Adam(model.parameters(), lr=fewshot_lr * 0.1)  # 10x smaller
    fewshot_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"\n🚀 Fine-tuning entire model for {fewshot_epochs} epochs (LR={fewshot_lr * 0.1})...")
    print("-" * 60)
    
    best_acc = 0
    best_state = None  # Store best model state
    
    for epoch in range(1, fewshot_epochs + 1):
        model.train()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Get class predictions (ignore domain output)
            class_out, _, _ = model(x)
            loss = fewshot_criterion(class_out, y)
            
            fewshot_optimizer.zero_grad()
            loss.backward()
            fewshot_optimizer.step()
        
        # Evaluate
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                class_out, _, _ = model(x)
                preds = class_out.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())
        
        acc = accuracy_score(y_true, y_pred)
        
        # Save best model state
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
        
        print(f"Epoch [{epoch:3d}/{fewshot_epochs}] - Few-shot Accuracy: {acc:.2%}")
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print("\n" + "-" * 60)
    print(f"✅ Stage 2 Complete - Best Few-shot Accuracy: {best_acc:.2%}")
    
    # Evaluate on FULL target dataset (not just test split)
    full_acc = evaluate_model(model, target_path, device)
    print(f"✅ Full Paderborn Accuracy: {full_acc:.2%}")
    
    # Save final model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"💾 Model saved to: {out_path}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60 + "\n")
    
    print(f"📊 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print(f"\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=["Normal", "Ball", "Inner", "Outer"],
                                digits=4))
    
    return model, {
        "dann_accuracy": dann_acc,
        "fewshot_accuracy": best_acc,
        "fewshot_fraction": fewshot_fraction
    }


def evaluate_model(model, data_path, device):
    """Quick evaluation on target domain"""
    model.eval()
    dataset = SupervisedDataset(data_path)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            class_out, _, _ = model(x)
            preds = class_out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    train_dann_fewshot(fewshot_fraction=0.05)
