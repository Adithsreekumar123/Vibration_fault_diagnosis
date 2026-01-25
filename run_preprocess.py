from src.data.preprocessing import preprocess_all

if __name__ == "__main__":
    preprocess_all(
        data_root="data/raw",
        out_dir="data/processed"
    )
