import torch.nn as nn
from src.models.cnn import CNNEncoder

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = CNNEncoder()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)
