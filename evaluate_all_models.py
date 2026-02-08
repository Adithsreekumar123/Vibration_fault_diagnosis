# evaluate_all_models.py
"""
Comprehensive evaluation of all trained models
"""
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from src.train.supervised_dataset import SupervisedDataset
from src.models.cnn_classifier import CNNClassifier
from src.models.cnn_lstm_classifier import CNNLSTMClassifier
from src.models.transformer_classifier import TransformerClassifier
from src.train.train_dann import DANNModel

device = 'cpu'

# Load datasets
print('='*70)
print('           VIBRATION FAULT DIAGNOSIS - MODEL EVALUATION')
print('='*70)

cwru_data = SupervisedDataset('data/processed/cwru_windows.npz')
paderborn_data = SupervisedDataset('data/processed/paderborn_windows.npz')

cwru_loader = DataLoader(cwru_data, batch_size=256, shuffle=False)
paderborn_loader = DataLoader(paderborn_data, batch_size=256, shuffle=False)

print(f'\nDataset Info:')
print(f'  CWRU: {len(cwru_data)} samples')
print(f'  Paderborn: {len(paderborn_data)} samples')
print()

def evaluate(model, loader, is_dann=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if is_dann:
                out, _, _ = model(x)
            else:
                out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    return accuracy_score(y_true, y_pred)

results = []

# Models to evaluate
models_to_eval = [
    ('CNN (CWRU)', 'results/supervised/cnn_cwru_supervised.pt', 'cnn', False),
    ('CNN + 5% Few-Shot', 'results/supervised/cnn_fewshot_5pct.pt', 'cnn', False),
    ('CNN + 20% Few-Shot', 'results/supervised/cnn_fewshot_20pct.pt', 'cnn', False),
    ('CNN-LSTM (CWRU)', 'results/supervised/cnn_lstm_cwru.pt', 'cnn_lstm', False),
    ('CNN-LSTM + 5% Few-Shot', 'results/supervised/cnn_lstm_fewshot_5pct.pt', 'cnn_lstm', False),
    ('CNN-LSTM + 20% Few-Shot', 'results/supervised/cnn_lstm_fewshot_20pct.pt', 'cnn_lstm', False),
    ('Transformer (CWRU)', 'results/supervised/transformer_cwru.pt', 'transformer', False),
    ('Transformer + 5% Few-Shot', 'results/supervised/transformer_fewshot_5pct.pt', 'transformer', False),
    ('Transformer + 20% Few-Shot', 'results/supervised/transformer_fewshot_20pct.pt', 'transformer', False),
    ('DANN (Zero-Shot)', 'results/dann/cnn_dann.pt', 'dann', True),
]

print('Evaluating all models...\n')

for name, path, model_type, is_dann in models_to_eval:
    if not os.path.exists(path):
        print(f'  {name}: NOT FOUND')
        continue
    
    if model_type == 'cnn':
        model = CNNClassifier(num_classes=4).to(device)
    elif model_type == 'cnn_lstm':
        model = CNNLSTMClassifier(num_classes=4).to(device)
    elif model_type == 'transformer':
        model = TransformerClassifier(num_classes=4).to(device)
    elif model_type == 'dann':
        model = DANNModel(num_classes=4).to(device)
    
    model.load_state_dict(torch.load(path, map_location=device))
    
    cwru_acc = evaluate(model, cwru_loader, is_dann)
    paderborn_acc = evaluate(model, paderborn_loader, is_dann)
    
    results.append((name, cwru_acc, paderborn_acc))
    print(f'  {name}')
    print(f'    CWRU: {cwru_acc:.1%} | Paderborn: {paderborn_acc:.1%}')

# Print summary table
print()
print('='*70)
print('                       RESULTS SUMMARY')
print('='*70)
print(f'{"Model":<35} | {"CWRU":>8} | {"Paderborn":>10}')
print('-'*70)
for name, cwru_acc, paderborn_acc in results:
    print(f'{name:<35} | {cwru_acc:>7.1%} | {paderborn_acc:>9.1%}')
print('='*70)

# Best model analysis
best_paderborn = max(results, key=lambda x: x[2])
print(f'\nBest Model for Paderborn: {best_paderborn[0]} ({best_paderborn[2]:.1%})')
