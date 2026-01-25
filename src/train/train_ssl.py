import torch
from torch.utils.data import DataLoader
from src.ssl.ssl_dataset import CWRUSSLDataset
from src.models.cnn import CNNEncoder
from src.ssl.simclr import SimCLR, nt_xent_loss

def train_ssl(
    data_path="data/processed/cwru_windows.npz",
    out_path="results/ssl/encoder_cwru_ssl.pt",
    epochs=50,
    batch_size=128,
    lr=1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CWRUSSLDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = CNNEncoder().to(device)
    model = SimCLR(encoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch}/{epochs}] - Loss: {total_loss / len(loader):.4f}")

    torch.save(encoder.state_dict(), out_path)
    print(f"\n✅ Saved SSL encoder to: {out_path}")
