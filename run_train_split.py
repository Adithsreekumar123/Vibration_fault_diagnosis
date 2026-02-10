"""
Run recording-based training pipeline.

This script:
1. Preprocesses CWRU data with recording-based split
2. Trains CNN classifier on proper splits
3. Reports metrics without data leakage
"""

from src.data.preprocessing_split import preprocess_with_split
from src.train.train_supervised_split import train_supervised_split


def main():
    print("\n" + "=" * 60)
    print("🎯 RECORDING-BASED TRAINING PIPELINE")
    print("   Prevents data leakage for accurate evaluation")
    print("=" * 60 + "\n")
    
    # Step 1: Preprocess with recording-based split
    print("📌 STEP 1: Preprocessing with recording-based split...\n")
    train_path, test_path = preprocess_with_split(
        data_root="data/raw",
        out_dir="data/processed",
        test_size=0.2,
        random_state=42
    )
    
    # Step 2: Train on proper splits
    print("\n📌 STEP 2: Training on recording-split data...\n")
    best_acc, model = train_supervised_split(
        train_path=train_path,
        test_path=test_path,
        ssl_ckpt=None,  # Set to SSL checkpoint path if available
        epochs=30,
        batch_size=128,
        lr=1e-3,
        out_path="results/supervised/cnn_cwru_split.pt"
    )
    
    print("\n" + "=" * 60)
    print(f"🏆 FINAL ACCURACY (No Data Leakage): {best_acc:.4f}")
    print("=" * 60)
    
    print("\n💡 Note: This accuracy is more realistic than random-split!")
    print("   If lower than before, it means the old method had data leakage.\n")


if __name__ == "__main__":
    main()
