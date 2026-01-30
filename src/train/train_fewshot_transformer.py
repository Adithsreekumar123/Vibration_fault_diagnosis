# src/train/train_fewshot_transformer.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.train.supervised_dataset import SupervisedDataset
from src.train.fewshot_utils import create_fewshot_split
from src.models.transformer_classifier import TransformerClassifier


def train_fewshot_transformer(
    data_path="data/processed/paderborn_windows.npz",
    pretrained_ckpt="results/supervised/transformer_cwru.pt",
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
    model = TransformerClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
    print("✅ Loaded Transformer trained on CWRU")

    # ❄️ Freeze embedding + transformer encoder
    for p in model.embedding.parameters():
        p.requires_grad = False
    for p in model.transformer.parameters():
        p.requires_grad = False

    # -------- Loss & Optimizer --------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
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

        print(f"Epoch [{epoch}/{epochs}] - Few-shot Acc (Transformer): {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (Few-shot Transformer):\n", cm)
    print(f"\n✅ Best Few-shot Accuracy (Transformer): {best_acc:.4f}")

    return best_acc
