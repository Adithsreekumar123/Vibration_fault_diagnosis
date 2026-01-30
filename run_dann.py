# run_dann.py
"""
Run DANN (Domain Adversarial Neural Network) Training

This trains the model to learn domain-invariant features that
work on both CWRU (source) and Paderborn (target) datasets.
"""

import os
from src.train.train_dann import train_dann, evaluate_dann_full

if __name__ == "__main__":
    # Create output directory
    os.makedirs("results/dann", exist_ok=True)
    
    # Train DANN model
    print("=" * 60)
    print("  DANN (Domain Adversarial Neural Network) Training")
    print("=" * 60)
    
    model = train_dann(
        source_path="data/processed/cwru_windows.npz",
        target_path="data/processed/paderborn_windows.npz",
        ssl_ckpt="results/ssl/encoder_cwru_ssl.pt",
        out_path="results/dann/cnn_dann.pt",
        epochs=50,
        batch_size=128,
        lr=1e-3,
        lambda_domain=0.5  # Balance between classification and domain loss
    )
    
    # Full evaluation
    print("\n" + "=" * 60)
    print("  DANN Model Evaluation on Paderborn (Zero-shot)")
    print("=" * 60 + "\n")
    
    results = evaluate_dann_full(
        model_path="results/dann/cnn_dann.pt",
        target_path="data/processed/paderborn_windows.npz"
    )
