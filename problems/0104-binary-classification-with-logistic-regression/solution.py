import numpy as np

def predict_logistic(X, w, b):
    X = np.asarray(X)
    w = np.asarray(w)
    z = X @ w + b

    probs = 1.0 / (1.0 + np.exp(-z))
    return (probs >= 0.5).astype(int)