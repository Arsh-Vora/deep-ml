import numpy as np


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> list[float]:

    A_mat = np.array(A, dtype=float, copy=True)
    b_vec = np.array(b, dtype=float, copy=True).reshape(-1, 1)
    n = len(b_vec)

    Ab = np.hstack([A_mat, b_vec])

    for i in range(n):
        pivot_row = i + np.argmax(np.abs(Ab[i:, i]))
        
        if np.abs(Ab[pivot_row, i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")

        if pivot_row != i:
            Ab[[i, pivot_row]] = Ab[[pivot_row, i]]

        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x.tolist()