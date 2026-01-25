import numpy as np

def zscore(x):
    mean = np.mean(x)
    std = np.std(x) + 1e-8
    return (x - mean) / std

def sliding_window(signal, window_size=4096, overlap=0.5):
    step = int(window_size * (1 - overlap))
    windows = []

    for start in range(0, len(signal) - window_size + 1, step):
        w = signal[start:start + window_size]
        w = zscore(w)
        windows.append(w)

    return np.array(windows, dtype=np.float32)
