# run_dann_fewshot.py
"""
Run DANN + Few-Shot Fine-tuning

This loads the pre-trained DANN model (from run_dann.py) and fine-tunes
the classifier on a small percentage of Paderborn labels.

Requirements:
- Run 'python run_dann.py' first to train the base DANN model

Expected accuracy on Paderborn:
- Base DANN: ~38%
- 5% few-shot: ~60-70%
- 20% few-shot: ~70-80%
"""

import os
from src.train.train_dann_fewshot import train_dann_fewshot

if __name__ == "__main__":
    os.makedirs("results/dann", exist_ok=True)
    
    # ============================================
    # CONFIGURATION - Change these as needed
    # ============================================
    FEWSHOT_FRACTION = 0.05   # 🔁 Options: 0.01 (1%), 0.05 (5%), 0.20 (20%)
    
    print("=" * 60)
    print("  DANN + Few-Shot Fine-tuning")
    print("=" * 60)
    print(f"  Few-shot fraction: {FEWSHOT_FRACTION*100:.0f}%")
    print(f"  Base DANN model: results/dann/cnn_dann.pt")
    print("=" * 60 + "\n")
    
    # Run training
    model, results = train_dann_fewshot(
        target_path="data/processed/paderborn_windows.npz",
        dann_ckpt="results/dann/cnn_dann.pt",
        out_path=f"results/dann/dann_fewshot_{int(FEWSHOT_FRACTION*100)}pct.pt",
        fewshot_epochs=20,
        fewshot_fraction=FEWSHOT_FRACTION,
        batch_size=128,
        fewshot_lr=1e-3
    )
    
    if results:
        # Print final summary
        print("\n" + "=" * 60)
        print("  FINAL RESULTS")
        print("=" * 60)
        print(f"  Base DANN Accuracy: {results['dann_accuracy']:.2%}")
        print(f"  DANN + {results['fewshot_fraction']*100:.0f}% Few-shot Accuracy: {results['fewshot_accuracy']:.2%}")
        print("=" * 60)

