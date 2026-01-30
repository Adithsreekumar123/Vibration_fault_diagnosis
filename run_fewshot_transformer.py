# run_fewshot_transformer.py

from src.train.train_fewshot_transformer import train_fewshot_transformer

if __name__ == "__main__":
    train_fewshot_transformer(
        fraction=0.05,   # 5% few-shot
        epochs=15,
        lr=1e-3
    )
