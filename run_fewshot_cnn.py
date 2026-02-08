# run_fewshot_cnn.py
"""
Run CNN Few-Shot Fine-tuning on Paderborn

This fine-tunes the CWRU-trained CNN on a small % of Paderborn data.
The model is saved to: results/supervised/cnn_fewshot_{pct}pct.pt
"""

from src.train.train_fewshot import train_fewshot

if __name__ == "__main__":
    # Change fraction: 0.01 (1%), 0.05 (5%), 0.20 (20%)
    FRACTION = 0.20
    
    print("=" * 60)
    print(f"  CNN Few-Shot Training ({FRACTION*100:.0f}% of Paderborn)")
    print("=" * 60)
    
    acc, model = train_fewshot(fraction=FRACTION)

