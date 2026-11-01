import numpy as np


def calculate_correlation_matrix(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:

    if Y is None:
        Y = X

    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    covariance = (X_centered.T @ Y_centered) / X.shape[0]

    std_X = np.std(X, axis=0, keepdims=True)  
    std_Y = np.std(Y, axis=0, keepdims=True)  
    std_matrix = std_X.T @ std_Y

    correlation_matrix = np.divide(
        covariance,
        std_matrix,
        out=np.zeros_like(covariance, dtype=float),
        where=std_matrix != 0
    )

    return correlation_matrix