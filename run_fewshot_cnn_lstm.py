# run_fewshot_cnn_lstm.py

from src.train.train_fewshot_cnn_lstm import train_fewshot_cnn_lstm

if __name__ == "__main__":
    train_fewshot_cnn_lstm(
        fraction=0.05,   # 5% few-shot
        epochs=15,
        lr=1e-3
    )
