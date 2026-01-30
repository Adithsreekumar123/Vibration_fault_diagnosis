# src/train/train_fewshot_cnn_lstm.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.train.supervised_dataset import SupervisedDataset
from src.train.fewshot_utils import create_fewshot_split
from src.models.cnn_lstm_classifier import CNNLSTMClassifier


def train_fewshot_cnn_lstm(
    data_path="data/processed/paderborn_windows.npz",
    pretrained_ckpt="results/supervised/cnn_lstm_cwru.pt",
    fraction=0.05,      # 5% few-shot
    epochs=15,
    batch_size=128,
    lr=1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- Dataset --------
    dataset = SupervisedDataset(data_path)
    train_ds, test_ds = create_fewshot_split(dataset, fraction=fraction)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # -------- Model --------
    model = CNNLSTMClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
    print("✅ Loaded CNN-LSTM trained on CWRU")

    # ❄️ Freeze CNN encoder ONLY (train LSTM + classifier)
    for p in model.cnn.parameters():
        p.requires_grad = False

    # -------- Loss & Optimizer --------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_acc = 0.0

    # -------- Training --------
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- Evaluation --------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                preds = model(x).argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())

        acc = accuracy_score(y_true, y_pred)
        best_acc = max(best_acc, acc)
        print(f"Epoch [{epoch}/{epochs}] - Few-shot Acc (CNN-LSTM): {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (Few-shot CNN-LSTM):\n", cm)
    print(f"\n✅ Best Few-shot Accuracy (CNN-LSTM): {best_acc:.4f}")

    return best_acc

