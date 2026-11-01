import numpy as np


def is_linearly_independent(vectors: list[list[float]]) -> bool:

    if not vectors:
        return True

    if len(vectors[0]) == 0:
        return False

    k = len(vectors)
    n = len(vectors[0])

    if k > n:
        return False

    A = np.array(vectors, dtype=float)
    rank = np.linalg.matrix_rank(A)

    return bool(rank == k)