# run_train_cnn_lstm.py

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score

from src.train.supervised_dataset import SupervisedDataset
from src.models.cnn_lstm_classifier import CNNLSTMClassifier


def train_cnn_lstm():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SupervisedDataset("data/processed/cwru_windows.npz")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = CNNLSTMClassifier(num_classes=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                preds = model(x).argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())

        acc = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch} - Test Acc: {acc:.4f}")

    torch.save(
        model.state_dict(),
        "results/supervised/cnn_lstm_cwru.pt"
    )
    print("✅ CNN-LSTM trained & saved")


if __name__ == "__main__":
    train_cnn_lstm()
