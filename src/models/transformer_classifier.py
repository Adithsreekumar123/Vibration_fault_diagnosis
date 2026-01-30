# src/models/transformer_classifier.py

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=4,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128
    ):
        super().__init__()

        # -------- Input embedding (Conv as patch embed) --------
        self.embedding = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=16, stride=16),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 1, 4096)
        x = self.embedding(x)          # (B, d_model, T)
        x = x.permute(0, 2, 1)         # (B, T, d_model)

        x = self.transformer(x)        # (B, T, d_model)
        feat = x.mean(dim=1)           # global average pooling

        return self.classifier(feat)
