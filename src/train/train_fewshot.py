import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.train.supervised_dataset import SupervisedDataset
from src.models.cnn_classifier import CNNClassifier
from src.train.fewshot_utils import create_fewshot_split


def train_fewshot(
    data_path="data/processed/paderborn_windows.npz",
    pretrained_ckpt="results/supervised/cnn_cwru_supervised.pt",
    out_path=None,  # Will be auto-generated if None
    fraction=0.05,   # 5% few-shot
    epochs=20,
    batch_size=128,
    lr=3e-4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-generate output path if not provided
    if out_path is None:
        pct = int(fraction * 100)
        out_path = f"results/supervised/cnn_fewshot_{pct}pct.pt"

    dataset = SupervisedDataset(data_path)
    train_ds, test_ds = create_fewshot_split(dataset, fraction=fraction)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    print(f"📊 Few-shot training: {len(train_ds)} samples ({fraction*100:.0f}%)")
    print(f"📊 Test set: {len(test_ds)} samples")

    model = CNNClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
    print("✅ Loaded CWRU-trained model")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- Evaluation ----
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1).cpu().numpy()

                y_pred.extend(preds)
                y_true.extend(y.numpy())

        acc = accuracy_score(y_true, y_pred)
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(model.state_dict(), out_path)
        
        print(f"Epoch [{epoch}/{epochs}] - Few-shot Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (Few-shot):\n", cm)
    print(f"\n✅ Best Accuracy: {best_acc:.4f}")
    print(f"💾 Model saved to: {out_path}")

    return best_acc, model

