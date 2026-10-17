import numpy as np

def sgd_update(X, y, weights, learning_rate, n_iter):
    # Ensure inputs are float arrays and make a copy of weights
    w = np.asarray(weights, dtype=float).copy()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    n_samples = X.shape[0]
    
    for i in range(n_iter):
        # 1. Cyclically select the sample index
        idx = i % n_samples
        x_i = X[idx]
        y_i = y[idx]
        
        # 2. Compute scalar prediction: x_i . w
        y_hat = np.dot(x_i, w)
        
        # 3. Compute per-sample MSE gradient: 2 * (y_hat - y_i) * x_i
        grad = 2.0 * (y_hat - y_i) * x_i
        
        # 4. Update weights
        w -= learning_rate * grad
        
    return [float(val) for val in w]