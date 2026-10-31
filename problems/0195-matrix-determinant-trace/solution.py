import numpy as np


def matrix_determinant_and_trace(matrix: list[list[float | int]]) -> tuple[float, float]:

    mat = np.array(matrix, dtype=float)

    det_val = float(np.linalg.det(mat))
    trace_val = float(np.trace(mat))

    return det_val, trace_val