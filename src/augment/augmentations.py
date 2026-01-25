import numpy as np

def jitter(x, sigma=0.03):
    return x + np.random.normal(0, sigma, size=x.shape)

def scaling(x, sigma=0.1):
    factor = np.random.normal(1.0, sigma)
    return x * factor

def time_shift(x, max_frac=0.1):
    shift = int(np.random.uniform(-max_frac, max_frac) * len(x))
    return np.roll(x, shift)

def time_mask(x, mask_frac=0.1):
    x = x.copy()
    m = int(mask_frac * len(x))
    start = np.random.randint(0, len(x) - m)
    x[start:start + m] = 0
    return x

def strong_augment(x):
    aug = x
    ops = [jitter, scaling, time_shift, time_mask]
    np.random.shuffle(ops)
    for op in ops[:2]:  # apply 2 random augmentations
        aug = op(aug)
    return aug
