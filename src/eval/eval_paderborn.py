import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.train.supervised_dataset import SupervisedDataset
from src.models.cnn_classifier import CNNClassifier


def evaluate_paderborn(
    model_ckpt,
    data_path="data/processed/paderborn_windows.npz",
    batch_size=128
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = SupervisedDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = CNNClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    print("✅ Loaded CWRU-trained model")
    print("🔹 Evaluating on Paderborn (zero-shot)...\n")

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"🎯 Zero-shot Accuracy on Paderborn: {acc:.4f}\n")
    print("📊 Confusion Matrix:")
    print(cm)

    print("\n📄 Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Ball", "Inner", "Outer"]
    ))
