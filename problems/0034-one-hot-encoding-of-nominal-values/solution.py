import numpy as np

def to_categorical(x, n_col=None):
    # Your code here
    x = np.asarray(x)
    if n_col is None:
        n_col = int(np.max(x)) + 1
        
    one_hot = np.zeros((x.shape[0], n_col), dtype=float)
    one_hot[np.arange(x.shape[0]), x] = 1.0
    
    return one_hot