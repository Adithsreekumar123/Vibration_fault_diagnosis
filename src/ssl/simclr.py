import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim = torch.mm(z, z.t()) / temperature
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])

    mask = torch.eye(2*N, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)

    return F.cross_entropy(sim, labels)
