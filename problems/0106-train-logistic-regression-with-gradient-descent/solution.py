import numpy as np

def train_logreg(X, y, learning_rate: float, iterations: int):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    N, d = X.shape
    
    # Prepend a column of ones for the bias term
    ones = np.ones((N, 1), dtype=float)
    X_tilde = np.hstack([ones, X])
    
    # Initialize all d + 1 parameters to zero
    theta = np.zeros(d + 1, dtype=float)
    
    losses = []
    
    for _ in range(iterations):
        # 1. Linear combination and sigmoid prediction
        z = X_tilde @ theta
        y_hat = 1.0 / (1.0 + np.exp(-z))
        
        # 2. Compute summed binary cross-entropy loss before updating
        # Clip y_hat to avoid log(0)
        y_hat_safe = np.clip(y_hat, 1e-15, 1.0 - 1e-15)
        loss = -np.sum(y * np.log(y_hat_safe) + (1.0 - y) * np.log(1.0 - y_hat_safe))
        losses.append(round(float(loss), 4))
        
        # 3. Compute gradient: X_tilde^T @ (y_hat - y)
        grad = X_tilde.T @ (y_hat - y)
        
        # 4. Gradient descent update (sum-based, no 1/N factor)
        theta = theta - learning_rate * grad

    coefficients = [round(float(w), 4) for w in theta]
    return (coefficients, losses)