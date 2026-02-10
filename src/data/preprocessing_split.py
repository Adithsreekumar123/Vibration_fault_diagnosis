"""
Recording-based preprocessing for CWRU dataset.

Splits data by recording source (not random windows) to prevent data leakage.
Each .mat file is a recording - windows from the same recording stay together.
"""

import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from src.data.windowing import sliding_window

LABEL_MAP = {
    "Normal": 0,
    "Ball": 1,
    "Inner_Race": 2,
    "Outer_Race": 3
}


def extract_de_signal(mat_path):
    """Extract 12k drive-end signal from CWRU .mat file"""
    mat = sio.loadmat(mat_path)
    for key in mat:
        if "DE_time" in key:
            return mat[key].squeeze()
    raise RuntimeError(f"DE_time not found in {mat_path}")


def collect_recordings(base_dir):
    """
    Collect all recording file paths organized by class.
    
    Returns:
        dict: {class_name: [list of .mat file paths]}
    """
    recordings = {
        "Normal": [],
        "Ball": [],
        "Inner_Race": [],
        "Outer_Race": []
    }
    
    # Normal recordings
    normal_dir = os.path.join(base_dir, "Normal")
    for fname in os.listdir(normal_dir):
        if fname.endswith(".mat"):
            recordings["Normal"].append(os.path.join(normal_dir, fname))
    
    # Fault recordings
    fault_root = os.path.join(base_dir, "12k_drive_end_bearing_fault_data")
    
    # Ball and Inner Race
    for cls in ["Ball", "Inner_Race"]:
        cls_dir = os.path.join(fault_root, cls)
        for fname in os.listdir(cls_dir):
            if fname.endswith(".mat"):
                recordings[cls].append(os.path.join(cls_dir, fname))
    
    # Outer Race (has subfolders)
    outer_root = os.path.join(fault_root, "Outer_Race")
    for sub in os.listdir(outer_root):
        sub_dir = os.path.join(outer_root, sub)
        if os.path.isdir(sub_dir):
            for fname in os.listdir(sub_dir):
                if fname.endswith(".mat"):
                    recordings["Outer_Race"].append(os.path.join(sub_dir, fname))
    
    return recordings


def generate_windows_from_recordings(file_paths, label, window_size=4096):
    """
    Generate windows from a list of recording files.
    
    Returns:
        X: numpy array of windows
        y: numpy array of labels
        sources: list of source file names for each window
    """
    X_all, y_all, sources = [], [], []
    
    for fpath in file_paths:
        signal = extract_de_signal(fpath)
        windows = sliding_window(signal, window_size)
        X_all.append(windows)
        y_all.extend([label] * len(windows))
        sources.extend([os.path.basename(fpath)] * len(windows))
    
    if X_all:
        X_all = np.vstack(X_all)
    else:
        X_all = np.array([])
    
    return X_all, np.array(y_all, dtype=np.int64), sources


def preprocess_with_split(
    data_root="data/raw",
    out_dir="data/processed",
    test_size=0.2,
    window_size=4096,
    random_state=42
):
    """
    Preprocess CWRU with recording-based train/test split.
    
    Splits .mat files per class → generates windows → saves separate files.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("📊 RECORDING-BASED PREPROCESSING")
    print("=" * 60)
    
    cwru_dir = os.path.join(data_root, "CWRU")
    recordings = collect_recordings(cwru_dir)
    
    # Storage for train and test
    X_train_all, y_train_all, src_train_all = [], [], []
    X_test_all, y_test_all, src_test_all = [], [], []
    
    print("\n🔸 Splitting recordings by class:")
    print("-" * 40)
    
    for cls_name, file_paths in recordings.items():
        label = LABEL_MAP[cls_name]
        n_files = len(file_paths)
        
        if n_files < 2:
            print(f"  ⚠️  {cls_name}: Only {n_files} file(s), using all for training")
            train_files, test_files = file_paths, []
        else:
            # Stratified split by recording
            train_files, test_files = train_test_split(
                file_paths, 
                test_size=test_size, 
                random_state=random_state
            )
        
        print(f"  {cls_name:12} | Train: {len(train_files):2} files | Test: {len(test_files):2} files")
        
        # Generate windows from train recordings
        if train_files:
            X_tr, y_tr, src_tr = generate_windows_from_recordings(train_files, label, window_size)
            if len(X_tr) > 0:
                X_train_all.append(X_tr)
                y_train_all.extend(y_tr)
                src_train_all.extend(src_tr)
        
        # Generate windows from test recordings
        if test_files:
            X_te, y_te, src_te = generate_windows_from_recordings(test_files, label, window_size)
            if len(X_te) > 0:
                X_test_all.append(X_te)
                y_test_all.extend(y_te)
                src_test_all.extend(src_te)
    
    # Stack all
    X_train = np.vstack(X_train_all)
    y_train = np.array(y_train_all, dtype=np.int64)
    
    X_test = np.vstack(X_test_all) if X_test_all else np.array([])
    y_test = np.array(y_test_all, dtype=np.int64) if y_test_all else np.array([])
    
    # Save train
    train_path = os.path.join(out_dir, "cwru_train.npz")
    np.savez(train_path, X=X_train, y=y_train)
    
    # Save test
    test_path = os.path.join(out_dir, "cwru_test.npz")
    np.savez(test_path, X=X_test, y=y_test)
    
    print("-" * 40)
    print(f"\n✅ TRAIN SET: {X_train.shape[0]} windows")
    print(f"   Saved to: {train_path}")
    print(f"\n✅ TEST SET:  {X_test.shape[0]} windows")
    print(f"   Saved to: {test_path}")
    
    # Class distribution
    print("\n📊 Class Distribution:")
    for cls_name, label in LABEL_MAP.items():
        n_train = np.sum(y_train == label)
        n_test = np.sum(y_test == label)
        print(f"   {cls_name:12} | Train: {n_train:5} | Test: {n_test:5}")
    
    print("\n" + "=" * 60)
    print("✅ NO DATA LEAKAGE - Train & test from different recordings!")
    print("=" * 60)
    
    return train_path, test_path


if __name__ == "__main__":
    preprocess_with_split()
