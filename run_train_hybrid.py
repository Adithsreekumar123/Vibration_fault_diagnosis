"""
Run CNN-LSTM-Transformer Hybrid model training.

This script:
1. Uses pre-split data (recording-based, no leakage)
2. Trains hybrid model with regularization
3. Implements early stopping to prevent overfitting
"""

from src.data.preprocessing_split import preprocess_with_split
from src.train.train_hybrid import train_hybrid
import os


def main():
    print("\n" + "=" * 60)
    print("🧠 CNN-LSTM-TRANSFORMER HYBRID MODEL")
    print("   Local patterns + Temporal memory + Global attention")
    print("=" * 60 + "\n")
    
    # Check if split data exists, if not create it
    train_path = "data/processed/cwru_train.npz"
    test_path = "data/processed/cwru_test.npz"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("📌 STEP 1: Creating recording-based data split...\n")
        preprocess_with_split(
            data_root="data/raw",
            out_dir="data/processed",
            test_size=0.2,
            random_state=42
        )
    else:
        print("✅ Using existing recording-split data\n")
    
    # Train hybrid model
    print("📌 STEP 2: Training Hybrid Model...\n")
    best_acc, model = train_hybrid(
        train_path=train_path,
        test_path=test_path,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        patience=10,
        out_path="results/supervised/hybrid_cwru.pt"
    )
    
    print("\n" + "=" * 60)
    print(f"🏆 HYBRID MODEL ACCURACY: {best_acc:.4f}")
    print("=" * 60)
    
    print("\n💡 Anti-overfitting measures used:")
    print("   • Recording-based data split (no leakage)")
    print("   • Dropout (30%)")
    print("   • Weight decay (L2 regularization)")
    print("   • Early stopping (patience=10)")
    print("   • Learning rate scheduling")
    print("   • Gradient clipping")


if __name__ == "__main__":
    main()
