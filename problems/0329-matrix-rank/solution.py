import numpy as np


def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix using Gaussian elimination with partial pivoting.

    Parameters:
        A: NumPy array of shape (m, n) representing the input matrix.
        tol: Tolerance for treating floating-point values as zero.

    Returns:
        int: The rank of the matrix.
    """
    mat = np.array(A, dtype=float, copy=True)
    if mat.ndim != 2:
        mat = np.atleast_2d(mat)

    m, n = mat.shape
    rank = 0
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot_offset = np.argmax(np.abs(mat[row:, col]))
        pivot_row = row + pivot_offset

        if np.abs(mat[pivot_row, col]) <= tol:
            continue

        if pivot_row != row:
            mat[[row, pivot_row]] = mat[[pivot_row, row]]

        pivot_val = mat[row, col]

        factors = mat[row + 1:, col] / pivot_val
        mat[row + 1:, col:] -= np.outer(factors, mat[row, col:])

        rank += 1
        row += 1

    return rank