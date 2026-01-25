import os
import numpy as np
import scipy.io as sio
from src.data.windowing import sliding_window

LABEL_MAP = {
    "Normal": 0,
    "Ball": 1,
    "Inner_Race": 2,
    "Outer_Race": 3
}

def extract_de_signal(mat_path):
    """
    Extract 12k drive-end signal from CWRU .mat file
    """
    mat = sio.loadmat(mat_path)
    for key in mat:
        if "DE_time" in key:
            return mat[key].squeeze()
    raise RuntimeError(f"DE_time not found in {mat_path}")

def load_normal(normal_dir, window_size):
    X, y = [], []

    for fname in os.listdir(normal_dir):
        if not fname.endswith(".mat"):
            continue
        signal = extract_de_signal(os.path.join(normal_dir, fname))
        windows = sliding_window(signal, window_size)
        X.append(windows)
        y.extend([LABEL_MAP["Normal"]] * len(windows))

    return X, y

def load_faults(fault_root, window_size):
    X, y = [], []

    # Ball and Inner Race
    for cls in ["Ball", "Inner_Race"]:
        cls_dir = os.path.join(fault_root, cls)
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".mat"):
                continue
            signal = extract_de_signal(os.path.join(cls_dir, fname))
            windows = sliding_window(signal, window_size)
            X.append(windows)
            y.extend([LABEL_MAP[cls]] * len(windows))

    # Outer race (merge all subfolders)
    outer_root = os.path.join(fault_root, "Outer_Race")
    for sub in os.listdir(outer_root):
        sub_dir = os.path.join(outer_root, sub)
        for fname in os.listdir(sub_dir):
            if not fname.endswith(".mat"):
                continue
            signal = extract_de_signal(os.path.join(sub_dir, fname))
            windows = sliding_window(signal, window_size)
            X.append(windows)
            y.extend([LABEL_MAP["Outer_Race"]] * len(windows))

    return X, y

def load_cwru(base_dir, window_size=4096):
    """
    base_dir = data/raw/CWRU
    """
    X_all, y_all = [], []

    # Normal samples
    normal_dir = os.path.join(base_dir, "Normal")
    Xn, yn = load_normal(normal_dir, window_size)
    X_all.extend(Xn)
    y_all.extend(yn)

    # 12k drive-end fault samples
    fault_root = os.path.join(
        base_dir,
        "12k_drive_end_bearing_fault_data"
    )
    Xf, yf = load_faults(fault_root, window_size)
    X_all.extend(Xf)
    y_all.extend(yf)

    X_all = np.vstack(X_all)
    y_all = np.array(y_all, dtype=np.int64)

    return X_all, y_all
