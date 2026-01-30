# src/train/train_dann.py
"""
DANN (Domain Adversarial Neural Network) Training

This implements domain adversarial training to learn features that:
1. Are good for fault classification (on source domain)
2. Are domain-invariant (can't distinguish CWRU from Paderborn)

The gradient reversal layer (GRL) makes the encoder learn to CONFUSE
the domain discriminator, resulting in transferable features.
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


class DANNModel(nn.Module):
    """
    Complete DANN Model with:
    - Shared CNN Encoder
    - Fault Classifier (for source domain)
    - Domain Discriminator (for both domains)
    """
    
    def __init__(self, num_classes=4, encoder_dim=128):
        super().__init__()
        
        # Shared feature extractor
        self.encoder = CNNEncoder()
        
        # Fault classifier (only used for source domain)
        self.classifier = nn.Linear(encoder_dim, num_classes)
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(input_dim=encoder_dim)
    
    def forward(self, x, alpha=1.0):
        """
        Forward pass
        
        Args:
            x: Input signal (batch, 1, 4096)
            alpha: GRL strength for domain discriminator
        
        Returns:
            class_output: Fault class predictions
            domain_output: Domain predictions (0=source, 1=target)
            features: Encoded features (for analysis)
        """
        # Extract features
        features = self.encoder(x)
        
        # Fault classification
        class_output = self.classifier(features)
        
        # Domain discrimination (with gradient reversal)
        domain_output = self.domain_discriminator(features, alpha)
        
        return class_output, domain_output, features


def train_dann(
    source_path="data/processed/cwru_windows.npz",
    target_path="data/processed/paderborn_windows.npz",
    ssl_ckpt="results/ssl/encoder_cwru_ssl.pt",
    out_path="results/dann/cnn_dann.pt",
    epochs=50,
    batch_size=128,
    lr=1e-3,
    lambda_domain=0.5
):
    """
    Train DANN model
    
    Args:
        source_path: Path to CWRU data
        target_path: Path to Paderborn data
        ssl_ckpt: Path to SSL pretrained encoder
        out_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        lambda_domain: Weight for domain loss (0.5 = balanced)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # Create dataset and loader
    dataset = DANNDataset(source_path, target_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    model = DANNModel(num_classes=4).to(device)
    
    # Load SSL pretrained weights
    if os.path.exists(ssl_ckpt):
        model.encoder.load_state_dict(torch.load(ssl_ckpt))
        print(f"✅ Loaded SSL pretrained encoder from: {ssl_ckpt}")
    else:
        print("⚠️ SSL checkpoint not found, training from scratch")
    
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    
    print(f"\n🚀 Starting DANN training for {epochs} epochs...")
    print(f"   Lambda (domain loss weight): {lambda_domain}")
    print("-" * 60)
    
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        total_class_loss = 0
        total_domain_loss = 0
        total_loss = 0
        
        # Compute alpha for this epoch (GRL scheduling)
        alpha = compute_alpha(epoch, epochs)
        
        for source_x, source_y, source_domain, target_x, target_domain in loader:
            # Move to device
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            source_domain = source_domain.to(device).unsqueeze(1)
            target_x = target_x.to(device)
            target_domain = target_domain.to(device).unsqueeze(1)
            
            # Forward pass on source
            source_class_out, source_domain_out, _ = model(source_x, alpha)
            
            # Forward pass on target
            _, target_domain_out, _ = model(target_x, alpha)
            
            # Classification loss (only on source)
            class_loss = class_criterion(source_class_out, source_y)
            
            # Domain loss (on both source and target)
            domain_loss_source = domain_criterion(source_domain_out, source_domain)
            domain_loss_target = domain_criterion(target_domain_out, target_domain)
            domain_loss = domain_loss_source + domain_loss_target
            
            # Total loss
            loss = class_loss + lambda_domain * domain_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += loss.item()
        
        n_batches = len(loader)
        avg_class_loss = total_class_loss / n_batches
        avg_domain_loss = total_domain_loss / n_batches
        avg_total_loss = total_loss / n_batches
        
        # Evaluate on target domain (Paderborn) every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            target_acc = evaluate_on_target(model, target_path, device)
            
            print(f"Epoch [{epoch:3d}/{epochs}] | "
                  f"Class Loss: {avg_class_loss:.4f} | "
                  f"Domain Loss: {avg_domain_loss:.4f} | "
                  f"α: {alpha:.3f} | "
                  f"Paderborn Acc: {target_acc:.2%}")
            
            if target_acc > best_acc:
                best_acc = target_acc
                # Save best model
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(model.state_dict(), out_path)
        else:
            print(f"Epoch [{epoch:3d}/{epochs}] | "
                  f"Class Loss: {avg_class_loss:.4f} | "
                  f"Domain Loss: {avg_domain_loss:.4f} | "
                  f"α: {alpha:.3f}")
    
    print("-" * 60)
    print(f"✅ DANN training complete!")
    print(f"   Best Paderborn Zero-shot Accuracy: {best_acc:.2%}")
    print(f"   Model saved to: {out_path}")
    
    return model


def evaluate_on_target(model, target_path, device):
    """Quick evaluation on target domain"""
    model.eval()
    
    dataset = SupervisedDataset(target_path)
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


def evaluate_dann_full(
    model_path="results/dann/cnn_dann.pt",
    target_path="data/processed/paderborn_windows.npz"
):
    """
    Full evaluation of DANN model with all metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = DANNModel(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print(f"✅ Loaded DANN model from: {model_path}")
    print("🔹 Evaluating on Paderborn (zero-shot)...\n")
    
    # Load data
    dataset = SupervisedDataset(target_path)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    y_true, y_pred, all_probs = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            class_out, _, _ = model(x)
            probs = torch.softmax(class_out, dim=1)
            preds = class_out.argmax(dim=1).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Ball", "Inner", "Outer"],
        digits=4
    )
    
    print(f"🎯 Zero-shot Accuracy on Paderborn: {acc:.4f} ({acc:.2%})\n")
    print("📊 Confusion Matrix:")
    print(cm)
    print("\n📄 Classification Report:")
    print(report)
    
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": all_probs
    }


if __name__ == "__main__":
    train_dann()
