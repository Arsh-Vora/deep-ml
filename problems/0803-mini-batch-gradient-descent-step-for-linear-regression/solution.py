import numpy as np

def mini_batch_gd_step(X, y, weights, bias, batch_indices, lr):
    X = np.asarray(X)
    y = np.asarray(y)
    weights = np.asarray(weights)
    
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    m = len(batch_indices)
    
    y_pred = X_batch @ weights + bias
    errors = y_pred - y_batch
    
    grad_w = (2.0 / m) * (X_batch.T @ errors)
    grad_b = (2.0 / m) * np.sum(errors)
    
    updated_weights = weights - lr * grad_w
    updated_bias = bias - lr * grad_b
    
    return np.append(updated_weights, updated_bias)