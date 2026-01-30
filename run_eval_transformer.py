# run_eval_transformer.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.train.supervised_dataset import SupervisedDataset
from src.models.transformer_classifier import TransformerClassifier


def eval_transformer_paderborn():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SupervisedDataset("data/processed/paderborn_windows.npz")
    loader = DataLoader(dataset, batch_size=128)

    model = TransformerClassifier(num_classes=4).to(device)
    model.load_state_dict(
        torch.load("results/supervised/transformer_cwru.pt", map_location=device)
    )
    model.eval()

    print("✅ Loaded Transformer trained on CWRU")
    print("🔹 Evaluating on Paderborn (zero-shot)...")

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Zero-shot Accuracy (Transformer): {acc:.4f}")
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\n📄 Classification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    eval_transformer_paderborn()
