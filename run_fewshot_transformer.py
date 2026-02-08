# run_fewshot_transformer.py
"""
Run Transformer Few-Shot Fine-tuning on Paderborn

This fine-tunes the CWRU-trained Transformer on a small % of Paderborn data.
The model is saved to: results/supervised/transformer_fewshot_{pct}pct.pt
"""

from src.train.train_fewshot_transformer import train_fewshot_transformer

if __name__ == "__main__":
    # Change fraction: 0.01 (1%), 0.05 (5%), 0.20 (20%)
    FRACTION = 0.05
    
    print("=" * 60)
    print(f"  Transformer Few-Shot Training ({FRACTION*100:.0f}% of Paderborn)")
    print("=" * 60)
    
    acc, model = train_fewshot_transformer(fraction=FRACTION)

