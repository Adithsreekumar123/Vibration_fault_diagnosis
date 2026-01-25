import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
from src.data.windowing import sliding_window

TARGET_FS = 12000   # match CWRU
ORIG_FS = 16000     # Paderborn

def extract_radial_signal(mat_path):
    mat = sio.loadmat(mat_path)

    # Get main struct
    main_key = [k for k in mat.keys() if not k.startswith("__")][0]
    data = mat[main_key]

    # Paderborn radial vibration is stored in Y.Data
    if "Y" in data.dtype.names:
        y_struct = data["Y"][0][0]

        if isinstance(y_struct, np.ndarray) and y_struct.dtype.names:
            if "Data" in y_struct.dtype.names:
                signal = y_struct["Data"][0][0]
                signal = np.asarray(signal, dtype=np.float32).squeeze()
                return signal

    raise RuntimeError(f"Radial vibration data not found in {mat_path}")

def resample_to_12k(signal):
    signal = np.asarray(signal, dtype=np.float32).flatten()
    return resample_poly(signal, TARGET_FS, ORIG_FS)
def load_paderborn(root_dir, window_size=4096):
    X_all, y_all = [], []

    print("\n🔹 Loading Paderborn dataset (unified labels)...")

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        # ----- Unified fault-type labeling -----
        if cls.startswith("K0"):
            label = 0        # Normal
        elif cls.startswith("KB"):
            label = 1        # Ball fault
        elif cls.startswith("KI"):
            label = 2        # Inner race fault
        elif cls.startswith("KA"):
            label = 3        # Outer race fault
        else:
            print(f"  ⚠️ Skipping unknown class folder: {cls}")
            continue

        print(f"  Folder {cls} → label {label}")

        for fname in os.listdir(cls_path):
            if not fname.endswith(".mat"):
                continue

            mat_path = os.path.join(cls_path, fname)

            try:
                signal = extract_radial_signal(mat_path)
                signal = resample_to_12k(signal)
                windows = sliding_window(signal, window_size)

                if len(windows) == 0:
                    continue

                X_all.append(windows)
                y_all.extend([label] * len(windows))

            except Exception as e:
                print(f"    ⚠️ Skipping {fname}: {str(e)}")
                continue

    if len(X_all) == 0:
        raise RuntimeError("No valid Paderborn signals were loaded.")

    X_all = np.vstack(X_all)
    y_all = np.array(y_all, dtype=np.int64)

    return X_all, y_all
