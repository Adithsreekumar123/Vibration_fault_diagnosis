# run_ssl_combined.py
"""
Run SSL Pretraining on Combined CWRU + Paderborn Data

This trains the encoder on BOTH source and target domain data
(without labels) to learn domain-invariant features.
"""

import os
import torch
from torch.utils.data import DataLoader

from src.ssl.combined_ssl_dataset import CombinedSSLDataset
from src.models.cnn import CNNEncoder
from src.ssl.simclr import SimCLR, nt_xent_loss


def train_ssl_combined(
    source_path="data/processed/cwru_windows.npz",
    target_path="data/processed/paderborn_windows.npz",
    out_path="results/ssl/encoder_combined_ssl.pt",
    epochs=50,
    batch_size=128,
    lr=1e-3
):
    """
    Train SSL encoder on combined CWRU + Paderborn data.
    
    Args:
        source_path: Path to CWRU data
        target_path: Path to Paderborn data
        out_path: Where to save the pretrained encoder
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # Create combined dataset
    print("\n📦 Loading combined SSL dataset...")
    dataset = CombinedSSLDataset(source_path, target_path, balance=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    encoder = CNNEncoder().to(device)
    model = SimCLR(encoder).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n🚀 Starting combined SSL training for {epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            
            # Get projections
            z1 = model(x1)
            z2 = model(x2)
            
            # Contrastive loss
            loss = nt_xent_loss(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{epochs}] - SSL Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(encoder.state_dict(), out_path)
    
    print("-" * 60)
    print(f"✅ Combined SSL training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Encoder saved to: {out_path}")
    
    return encoder


if __name__ == "__main__":
    os.makedirs("results/ssl", exist_ok=True)
    
    train_ssl_combined(
        source_path="data/processed/cwru_windows.npz",
        target_path="data/processed/paderborn_windows.npz",
        out_path="results/ssl/encoder_combined_ssl.pt",
        epochs=50
    )
