import numpy as np

def pad_list(arr: np.ndarray, length: int):
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'constant', constant_values=0)
    return arr

def to_chunks(arr: np.ndarray, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(arr), n):
        yield arr[i:i + n]