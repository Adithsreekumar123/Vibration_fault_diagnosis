import scipy.io as sio

mat_path = "data/raw/Paderborn/paderborn/K001/N09_M07_F10_K001_1.mat"

mat = sio.loadmat(mat_path)

print("Top-level keys:")
for k in mat.keys():
    if not k.startswith("__"):
        print(" ", k, type(mat[k]))

# If there's a struct, inspect its fields
for k in mat:
    if not k.startswith("__"):
        obj = mat[k]
        if hasattr(obj, "dtype") and obj.dtype.names is not None:
            print(f"\nStruct '{k}' fields:")
            for name in obj.dtype.names:
                print(" ", name)
