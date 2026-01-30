# run_eval_cnn_lstm.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.train.supervised_dataset import SupervisedDataset
from src.models.cnn_lstm_classifier import CNNLSTMClassifier


def evaluate_cnn_lstm_paderborn():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- Load Paderborn dataset --------
    dataset = SupervisedDataset("data/processed/paderborn_windows.npz")
    loader = DataLoader(dataset, batch_size=128)

    # -------- Load CNN-LSTM model --------
    model = CNNLSTMClassifier(num_classes=4).to(device)
    model.load_state_dict(
        torch.load("results/supervised/cnn_lstm_cwru.pt", map_location=device)
    )
    model.eval()

    print("✅ Loaded CNN-LSTM trained on CWRU")
    print("🔹 Evaluating on Paderborn (zero-shot)...")

    # -------- Evaluation --------
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n🎯 Zero-shot Accuracy (CNN-LSTM): {acc:.4f}")
    print("\n📊 Confusion Matrix:\n", cm)
    print("\n📄 Classification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    evaluate_cnn_lstm_paderborn()
