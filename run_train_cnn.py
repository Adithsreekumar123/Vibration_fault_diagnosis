from src.train.train_supervised import train_supervised

if __name__ == "__main__":
    train_supervised(
        data_path="data/processed/cwru_windows.npz",
        ssl_ckpt="results/ssl/encoder_cwru_ssl.pt"
    )
