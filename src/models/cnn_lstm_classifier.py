# src/models/cnn_lstm_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes=4, lstm_hidden=128, lstm_layers=1):
        super().__init__()

        # -------- CNN Encoder --------
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # -------- LSTM --------
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # -------- Classifier --------
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (B, 1, 4096)
        x = self.cnn(x)              # (B, C, T)
        x = x.permute(0, 2, 1)       # (B, T, C)

        lstm_out, _ = self.lstm(x)   # (B, T, H)
        feat = lstm_out[:, -1, :]    # last timestep

        return self.classifier(feat)
