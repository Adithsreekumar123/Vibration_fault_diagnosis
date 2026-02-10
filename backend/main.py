"""
FastAPI Backend for Vibration Fault Diagnosis

Endpoints:
  POST /api/predict  — Upload a .mat file + model name → fault prediction
  GET  /api/models   — List available models and metadata
"""

import os
import sys
import io
import tempfile
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path so we can import src.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.cnn_classifier import CNNClassifier
from src.models.cnn_lstm_classifier import CNNLSTMClassifier
from src.models.transformer_classifier import TransformerClassifier
from src.models.cnn_lstm_transformer import CNNLSTMTransformer
from src.train.train_dann import DANNModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
WINDOW_SIZE = 4096

MODEL_REGISTRY = {
    "CNN": {
        "path": "results/supervised/cnn_cwru_split.pt",
        "desc": "1D CNN with recording-based split",
        "type": "Supervised",
        "cwru_acc": "100%",
        "paderborn_acc": "~21%",
    },
    "CNN + 5% Few-Shot": {
        "path": "results/supervised/cnn_fewshot_5pct.pt",
        "desc": "CNN fine-tuned on 5% Paderborn labels",
        "type": "Few-Shot Learning",
        "cwru_acc": "-",
        "paderborn_acc": "~55%",
    },
    "CNN + 20% Few-Shot": {
        "path": "results/supervised/cnn_fewshot_20pct.pt",
        "desc": "CNN fine-tuned on 20% Paderborn labels",
        "type": "Few-Shot Learning",
        "cwru_acc": "-",
        "paderborn_acc": "~65%",
    },
    "CNN-LSTM": {
        "path": "results/supervised/cnn_lstm_cwru.pt",
        "desc": "CNN + LSTM for temporal patterns",
        "type": "Supervised",
        "cwru_acc": "~98%",
        "paderborn_acc": "~21%",
    },
    "CNN-LSTM + 5% Few-Shot": {
        "path": "results/supervised/cnn_lstm_fewshot_5pct.pt",
        "desc": "CNN-LSTM fine-tuned on 5% Paderborn labels",
        "type": "Few-Shot Learning",
        "cwru_acc": "-",
        "paderborn_acc": "~55%",
    },
    "CNN-LSTM + 20% Few-Shot": {
        "path": "results/supervised/cnn_lstm_fewshot_20pct.pt",
        "desc": "CNN-LSTM fine-tuned on 20% Paderborn labels",
        "type": "Few-Shot Learning",
        "cwru_acc": "-",
        "paderborn_acc": "~65%",
    },
    "Transformer": {
        "path": "results/supervised/transformer_cwru.pt",
        "desc": "Transformer encoder for sequence modeling",
        "type": "Supervised",
        "cwru_acc": "~97%",
        "paderborn_acc": "~21%",
    },
    "Transformer + 5% Few-Shot": {
        "path": "results/supervised/transformer_fewshot_5pct.pt",
        "desc": "Transformer fine-tuned on 5% Paderborn labels",
        "type": "Few-Shot Learning",
        "cwru_acc": "-",
        "paderborn_acc": "~50%",
    },
    "Transformer + 20% Few-Shot": {
        "path": "results/supervised/transformer_fewshot_20pct.pt",
        "desc": "Transformer fine-tuned on 20% Paderborn labels",
        "type": "Few-Shot Learning",
        "cwru_acc": "-",
        "paderborn_acc": "~60%",
    },
    "Hybrid (CNN-LSTM-Transformer)": {
        "path": "results/supervised/hybrid_cwru.pt",
        "desc": "CNN + LSTM + Transformer hybrid",
        "type": "Hybrid Deep Learning",
        "cwru_acc": "100%",
        "paderborn_acc": "~25%",
    },
    "DANN": {
        "path": "results/dann/cnn_dann.pt",
        "desc": "Domain Adversarial Neural Network",
        "type": "Domain Adaptation",
        "cwru_acc": "~77%",
        "paderborn_acc": "~39%",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model_cache: dict = {}


def _abs(rel_path: str) -> str:
    """Resolve a path relative to PROJECT_ROOT."""
    return os.path.join(PROJECT_ROOT, rel_path)


def _load_model(model_name: str):
    """Load and cache a PyTorch model."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    info = MODEL_REGISTRY.get(model_name)
    if info is None:
        raise HTTPException(404, f"Unknown model: {model_name}")

    model_path = _abs(info["path"])
    if not os.path.exists(model_path):
        raise HTTPException(404, f"Model weights not found: {info['path']}. Train the model first.")

    # Instantiate the correct architecture
    if "DANN" in model_name:
        from src.train.train_dann_fewshot import DANNModel as DANNFewShotModel
        model = DANNFewShotModel(num_classes=4).to(DEVICE)
    elif "Hybrid" in model_name:
        model = CNNLSTMTransformer(num_classes=4).to(DEVICE)
    elif "CNN-LSTM" in model_name:
        model = CNNLSTMClassifier(num_classes=4).to(DEVICE)
    elif "Transformer" in model_name:
        model = TransformerClassifier(num_classes=4).to(DEVICE)
    else:
        model = CNNClassifier(num_classes=4).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    _model_cache[model_name] = model
    return model


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def _sliding_window(signal: np.ndarray, window_size: int = WINDOW_SIZE, overlap: float = 0.5):
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        w = signal[start : start + window_size]
        w = _zscore(w)
        windows.append(w)
    return np.array(windows, dtype=np.float32)


def _extract_signal(mat_path: str) -> np.ndarray:
    """
    Auto-detect .mat format and extract the vibration signal.
    Supports CWRU (DE_time key) and Paderborn (Y.Data struct).
    """
    mat = sio.loadmat(mat_path)

    # --- Try CWRU format: look for a key containing 'DE_time' ---
    for key in mat:
        if "DE_time" in key:
            return mat[key].squeeze().astype(np.float32)

    # --- Try Paderborn format: struct → Y → Data ---
    non_dunder = [k for k in mat.keys() if not k.startswith("__")]
    if non_dunder:
        for main_key in non_dunder:
            data = mat[main_key]
            try:
                if hasattr(data, "dtype") and data.dtype.names and "Y" in data.dtype.names:
                    y_struct = data["Y"][0][0]
                    if isinstance(y_struct, np.ndarray) and y_struct.dtype.names and "Data" in y_struct.dtype.names:
                        signal = y_struct["Data"][0][0]
                        signal = np.asarray(signal, dtype=np.float32).squeeze()
                        # Resample from 16 kHz → 12 kHz to match CWRU training data
                        signal = resample_poly(signal, 12000, 16000).astype(np.float32)
                        return signal
            except Exception:
                continue

    # --- Fallback: grab the first large numeric array ---
    for key in mat:
        if key.startswith("__"):
            continue
        arr = mat[key]
        if isinstance(arr, np.ndarray) and arr.size > WINDOW_SIZE:
            return arr.squeeze().astype(np.float32)

    raise ValueError("Could not find vibration signal in .mat file. Supported formats: CWRU (DE_time) and Paderborn (Y.Data).")


def _infer_ground_truth(filename: str):
    """
    Try to infer the ground truth fault class from the .mat filename.
    CWRU: Normal(97-100), Ball(118-122,185-189,222-226), Inner(105-109,169-173,209-213), Outer(130-135,197-201,234-238)
    Paderborn: K0xx=Normal, KAxx=Outer, KIxx=Inner, KBxx=Ball
    """
    import re
    name = filename.upper().replace('.MAT', '')

    # Paderborn naming — fault code can appear anywhere in filename
    if re.search(r'K0\d{2}', name):
        return {"label": "Normal", "class_index": 0}
    if re.search(r'KA\d', name):
        return {"label": "Outer Race Fault", "class_index": 3}
    if re.search(r'KI\d', name):
        return {"label": "Inner Race Fault", "class_index": 2}
    if re.search(r'KB\d', name):
        return {"label": "Ball Fault", "class_index": 1}

    # CWRU: extract the 3-digit code from filename
    m = re.search(r'(\d{3})', name)
    if m:
        code = int(m.group(1))
        if code in range(97, 101):
            return {"label": "Normal", "class_index": 0}
        if code in list(range(118, 123)) + list(range(185, 190)) + list(range(222, 227)):
            return {"label": "Ball Fault", "class_index": 1}
        if code in list(range(105, 110)) + list(range(169, 174)) + list(range(209, 214)):
            return {"label": "Inner Race Fault", "class_index": 2}
        if code in list(range(130, 136)) + list(range(197, 202)) + list(range(234, 239)):
            return {"label": "Outer Race Fault", "class_index": 3}

    return None


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Vibration Fault Diagnosis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
def list_models():
    """Return available models and their metadata."""
    models = []
    for name, info in MODEL_REGISTRY.items():
        available = os.path.exists(_abs(info["path"]))
        models.append({
            "name": name,
            "description": info["desc"],
            "type": info["type"],
            "cwru_accuracy": info["cwru_acc"],
            "paderborn_accuracy": info["paderborn_acc"],
            "available": available,
        })
    return {"models": models}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("CNN"),
):
    """
    Upload a .mat vibration file and get fault predictions.

    The file is parsed to extract the vibration signal, split into
    4096-sample windows, and classified by the selected model.
    """
    if not file.filename.endswith(".mat"):
        raise HTTPException(400, "Only .mat files are accepted.")

    # Save upload to a temp file so scipy can read it
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        signal = _extract_signal(tmp_path)
    except Exception as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)

    windows = _sliding_window(signal)
    if len(windows) == 0:
        raise HTTPException(422, f"Signal too short ({len(signal)} samples). Need at least {WINDOW_SIZE}.")

    # Load model
    model = _load_model(model_name)
    is_dann = "DANN" in model_name

    # Batch inference
    x = torch.tensor(windows, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        if is_dann:
            logits, _, _ = model(x)
        else:
            logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = np.argmax(probs, axis=1)

    # Aggregate: majority vote across windows
    from collections import Counter
    vote = Counter(preds.tolist())
    overall_class = vote.most_common(1)[0][0]
    overall_confidence = float(probs[:, overall_class].mean())

    # Build per-window results (send first 20 max to keep response small)
    max_windows = min(20, len(windows))
    window_results = []
    for i in range(max_windows):
        window_results.append({
            "window_index": i,
            "predicted_class": int(preds[i]),
            "predicted_label": CLASS_NAMES[int(preds[i])],
            "confidence": float(probs[i].max()),
            "probabilities": {CLASS_NAMES[j]: round(float(probs[i][j]), 4) for j in range(4)},
        })

    # Downsample signal for visualization (max 2000 points)
    viz_signal = signal.tolist()
    if len(viz_signal) > 2000:
        step = len(viz_signal) // 2000
        viz_signal = viz_signal[::step][:2000]

    # Average probabilities across all windows
    avg_probs = probs.mean(axis=0)

    # Try to infer ground truth from filename
    ground_truth = _infer_ground_truth(file.filename)

    response = {
        "filename": file.filename,
        "model": model_name,
        "total_windows": int(len(windows)),
        "signal_length": int(len(signal)),
        "overall_prediction": {
            "class_index": int(overall_class),
            "label": CLASS_NAMES[int(overall_class)],
            "confidence": round(overall_confidence, 4),
        },
        "average_probabilities": {CLASS_NAMES[j]: round(float(avg_probs[j]), 4) for j in range(4)},
        "window_results": window_results,
        "signal_data": viz_signal,
    }

    if ground_truth:
        response["ground_truth"] = ground_truth
        response["is_correct"] = ground_truth["class_index"] == int(overall_class)

    return response


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------
DATASET_PATHS = {
    "CWRU": "data/processed/cwru_windows.npz",
    "Paderborn": "data/processed/paderborn_windows.npz",
}

_dataset_cache: dict = {}


def _load_dataset(name: str):
    """Load and cache a dataset."""
    if name in _dataset_cache:
        return _dataset_cache[name]

    info = DATASET_PATHS.get(name)
    if info is None:
        raise HTTPException(404, f"Unknown dataset: {name}")

    path = _abs(info)
    if not os.path.exists(path):
        raise HTTPException(404, f"Dataset not found: {info}. Run preprocessing first.")

    data = np.load(path)
    X, y = data["X"], data["y"]
    _dataset_cache[name] = (X, y)
    return X, y


@app.get("/api/datasets")
def list_datasets():
    """Return available datasets and their info."""
    datasets = []
    for name, path in DATASET_PATHS.items():
        full_path = _abs(path)
        available = os.path.exists(full_path)
        info = {"name": name, "available": available, "samples": 0, "window_size": 0}
        if available:
            data = np.load(full_path)
            info["samples"] = int(data["X"].shape[0])
            info["window_size"] = int(data["X"].shape[1])
        datasets.append(info)
    return {"datasets": datasets, "device": DEVICE}


@app.get("/api/datasets/{dataset_name}/sample")
def get_sample_prediction(
    dataset_name: str,
    model_name: str,
    sample_index: int = 0,
):
    """Predict on a single sample from a dataset — matches Streamlit Tab 1."""
    X, y = _load_dataset(dataset_name)

    if sample_index < 0 or sample_index >= len(X):
        raise HTTPException(400, f"sample_index must be 0..{len(X)-1}")

    signal = X[sample_index]
    true_label = int(y[sample_index])

    model = _load_model(model_name)
    is_dann = "DANN" in model_name

    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if is_dann:
            logits, _, _ = model(x)
        else:
            logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    # Downsample signal for chart (max 1000 points)
    viz = signal.tolist()
    if len(viz) > 1000:
        step = len(viz) // 1000
        viz = viz[::step][:1000]

    return {
        "sample_index": sample_index,
        "total_samples": len(X),
        "signal_data": viz,
        "true_label": true_label,
        "true_name": CLASS_NAMES[true_label],
        "predicted_class": pred_class,
        "predicted_label": CLASS_NAMES[pred_class],
        "confidence": round(confidence, 4),
        "is_correct": pred_class == true_label,
        "probabilities": {CLASS_NAMES[j]: round(float(probs[j]), 4) for j in range(4)},
    }


@app.get("/api/datasets/{dataset_name}/random_index")
def random_sample_index(dataset_name: str):
    """Return a random valid sample index."""
    X, _ = _load_dataset(dataset_name)
    idx = int(np.random.randint(0, len(X)))
    return {"index": idx, "total": len(X)}


@app.post("/api/evaluate")
def evaluate_full(
    model_name: str = Form(...),
    dataset_name: str = Form("Paderborn"),
):
    """Full dataset evaluation — matches Streamlit Tab 2."""
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report,
        precision_score, recall_score, f1_score,
    )

    X, y = _load_dataset(dataset_name)
    model = _load_model(model_name)
    is_dann = "DANN" in model_name

    # Batch inference
    y_true, y_pred = [], []
    batch_size = 256
    for i in range(0, len(X), batch_size):
        batch_X = X[i : i + batch_size]
        batch_y = y[i : i + batch_size]

        x = torch.tensor(batch_X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            if is_dann:
                logits, _, _ = model(x)
            else:
                logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        preds = np.argmax(probs, axis=1)
        y_true.extend(batch_y.tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    per_class = []
    for name in CLASS_NAMES:
        m = report[name]
        per_class.append({
            "class": name,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1_score": round(m["f1-score"], 4),
            "support": int(m["support"]),
        })

    return {
        "model": model_name,
        "dataset": dataset_name,
        "total_samples": int(len(y_true)),
        "correct": int(np.sum(y_true == y_pred)),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "class_names": CLASS_NAMES,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Serve React production build (single-server deployment)
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend", "dist")

if os.path.isdir(FRONTEND_DIR):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    # Catch-all: serve index.html for any non-API route (SPA routing)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
