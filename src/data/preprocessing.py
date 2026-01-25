import os
import numpy as np
from src.data.load_cwru import load_cwru
from src.data.load_paderborn import load_paderborn


def preprocess_all(data_root="data/raw", out_dir="data/processed"):
    """
    Preprocess CWRU and Paderborn datasets with unified labels:
    0 - Normal
    1 - Ball fault
    2 - Inner race fault
    3 - Outer race fault
    """

    os.makedirs(out_dir, exist_ok=True)

    # ---------------- CWRU ----------------
    print("🔹 Processing CWRU...")
    Xc, yc = load_cwru(os.path.join(data_root, "CWRU"))

    print(f"✔ CWRU windows : {Xc.shape[0]}")
    print(f"✔ Window size  : {Xc.shape[1]}")
    print(f"✔ Num classes  : {len(set(yc))}")

    np.savez(
        os.path.join(out_dir, "cwru_windows.npz"),
        X=Xc,
        y=yc
    )

    print("✅ Saved: data/processed/cwru_windows.npz\n")

    # ---------------- PADERBORN ----------------
    print("🔹 Processing Paderborn...")
    Xp, yp = load_paderborn(
        os.path.join(data_root, "Paderborn", "paderborn")
    )

    print(f"✔ Paderborn windows : {Xp.shape[0]}")
    print(f"✔ Window size       : {Xp.shape[1]}")
    print(f"✔ Num classes       : {len(set(yp))}")

    np.savez(
        os.path.join(out_dir, "paderborn_windows.npz"),
        X=Xp,
        y=yp
    )

    print("✅ Saved: data/processed/paderborn_windows.npz")


if __name__ == "__main__":
    preprocess_all()
