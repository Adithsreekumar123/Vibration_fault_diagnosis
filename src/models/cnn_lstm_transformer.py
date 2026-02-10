# src/models/cnn_lstm_transformer.py
"""
Hybrid CNN-LSTM-Transformer Model for Vibration Fault Diagnosis

Architecture:
- CNN: Local pattern extraction (spatial/frequency features)
- LSTM: Temporal memory (sequence dependencies)  
- Transformer: Global attention (entire signal context)

Anti-overfitting measures:
- Dropout layers throughout
- Batch normalization
- Weight decay in optimizer
- Early stopping in training
"""

import torch
import torch.nn as nn


class CNNLSTMTransformer(nn.Module):
    """
    Hybrid model combining CNN, LSTM, and Transformer.
    
    Flow: Input → CNN → LSTM → Transformer → Classifier
    """
    
    def __init__(
        self,
        num_classes=4,
        cnn_channels=[16, 32, 64],
        lstm_hidden=64,
        lstm_layers=1,
        transformer_heads=4,
        transformer_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        self.dropout_rate = dropout
        
        # ========== CNN Encoder (Local Pattern Extraction) ==========
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, cnn_channels[0], kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ========== LSTM Layer (Temporal Memory) ==========
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,  # Bidirectional for better context
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # LSTM output size (bidirectional = 2x hidden)
        lstm_out_size = lstm_hidden * 2
        
        # ========== Transformer Encoder (Global Attention) ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_out_size,
            nhead=transformer_heads,
            dim_feedforward=lstm_out_size * 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # ========== Classifier Head ==========
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through CNN → LSTM → Transformer → Classifier
        
        Args:
            x: Input tensor of shape (B, 1, 4096)
        
        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        # CNN: Extract local patterns
        # Input: (B, 1, 4096) → Output: (B, 64, T)
        x = self.cnn(x)
        
        # Reshape for LSTM: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        
        # LSTM: Capture temporal dependencies
        # Input: (B, T, 64) → Output: (B, T, 128) [bidirectional]
        x, _ = self.lstm(x)
        
        # Transformer: Global attention
        # Input: (B, T, 128) → Output: (B, T, 128)
        x = self.transformer(x)
        
        # Global average pooling over time dimension
        x = x.mean(dim=1)  # (B, 128)
        
        # Classifier
        logits = self.classifier(x)
        
        return logits


# Quick test
if __name__ == "__main__":
    model = CNNLSTMTransformer(num_classes=4)
    x = torch.randn(4, 1, 4096)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
