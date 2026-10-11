import numpy as np
def matrixmul(a:list[list[int|float]],b:list[list[int|float]])-> list[list[int|float]]:

    if len(a[0]) != len(b):
        return -1
    A = np.array(a)
    B = np.array(b)
    c = A @ B
    return c.tolist()