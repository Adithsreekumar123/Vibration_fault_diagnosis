import os
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from src.train.supervised_dataset import SupervisedDataset
from src.models.cnn_classifier import CNNClassifier


def train_supervised(
    data_path,
    ssl_ckpt,
    epochs=30,
    batch_size=128,
    lr=1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SupervisedDataset(data_path)
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = CNNClassifier(num_classes=4).to(device)

    # 🔹 Load SSL weights
    model.encoder.load_state_dict(torch.load(ssl_ckpt))
    print("✅ Loaded SSL pretrained encoder")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    test_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- Evaluation on CWRU test split ----
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
        test_accs.append(acc)

        print(f"Epoch [{epoch}/{epochs}] - Test Accuracy: {acc:.4f}")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # ---- Plot ----
    plt.plot(test_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CWRU Test Accuracy")
    plt.show()

    # ---- Save trained classifier (IMPORTANT) ----
    os.makedirs("results/supervised", exist_ok=True)
    torch.save(
        model.state_dict(),
        "results/supervised/cnn_cwru_supervised.pt"
    )
    print("\n✅ Saved supervised model to: results/supervised/cnn_cwru_supervised.pt")

    return model
