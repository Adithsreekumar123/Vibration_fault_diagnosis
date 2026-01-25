from src.train.train_fewshot import train_fewshot

if __name__ == "__main__":
    train_fewshot(
        fraction=0.20  # 🔁 change to 0.05 for 5% experiment
    )
