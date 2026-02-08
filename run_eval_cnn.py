from src.eval.eval_paderborn import evaluate_paderborn

if __name__ == "__main__":
    evaluate_paderborn(
        model_ckpt="results/supervised/cnn_cwru_supervised.pt",
        data_path="data/processed/paderborn_windows.npz"
    )
