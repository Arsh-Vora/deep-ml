import math

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]

    trace = a + d
    
    det = a * d - b * c

    discriminant = trace**2 - 4 * det
    sqrt_disc = math.sqrt(max(0.0, discriminant))

    lambda_1 = (trace + sqrt_disc) / 2
    lambda_2 = (trace - sqrt_disc) / 2

    return sorted([float(lambda_1), float(lambda_2)], reverse=True)