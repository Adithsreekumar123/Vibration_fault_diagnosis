import os
from src.train.train_ssl import train_ssl

if __name__ == "__main__":
    os.makedirs("results/ssl", exist_ok=True)

    train_ssl(
        data_path="data/processed/cwru_windows.npz",
        out_path="results/ssl/encoder_cwru_ssl.pt"
    )
