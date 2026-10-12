import numpy as np

def pairwise_cosine_similarity(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    X_normalized = X / norms
    return (X_normalized @ X_normalized.T).tolist()
    pass