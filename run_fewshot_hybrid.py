"""
Run Hybrid + Few-Shot training.

Fine-tunes the pre-trained hybrid model on 5% and 20% Paderborn labels.
"""

from src.train.train_fewshot_hybrid import train_hybrid_fewshot


def main():
    print("\n" + "=" * 60)
    print("🧠 HYBRID + FEW-SHOT TRAINING")
    print("   Fine-tuning CNN-LSTM-Transformer on Paderborn")
    print("=" * 60)
    
    # Train with 5% labels
    print("\n" + "=" * 60)
    print("📌 PHASE 1: 5% Few-Shot")
    print("=" * 60)
    
    acc_5pct, _ = train_hybrid_fewshot(
        pretrained_path="results/supervised/hybrid_cwru.pt",
        paderborn_path="data/processed/paderborn_windows.npz",
        fewshot_pct=0.05,
        epochs=30,
        batch_size=32,
        lr=1e-4,
        out_path="results/supervised/hybrid_fewshot_5pct.pt"
    )
    
    # Train with 20% labels
    print("\n" + "=" * 60)
    print("📌 PHASE 2: 20% Few-Shot")
    print("=" * 60)
    
    acc_20pct, _ = train_hybrid_fewshot(
        pretrained_path="results/supervised/hybrid_cwru.pt",
        paderborn_path="data/processed/paderborn_windows.npz",
        fewshot_pct=0.20,
        epochs=30,
        batch_size=32,
        lr=1e-4,
        out_path="results/supervised/hybrid_fewshot_20pct.pt"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 HYBRID + FEW-SHOT RESULTS SUMMARY")
    print("=" * 60)
    print(f"   Hybrid (no fine-tune):    ~17%")
    print(f"   Hybrid + 5% Few-Shot:     {acc_5pct:.2%}")
    print(f"   Hybrid + 20% Few-Shot:    {acc_20pct:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
