def orthogonal_projection(v: list[float], L: list[float]) -> list[float]:

    v_dot_L = sum(v_i * L_i for v_i, L_i in zip(v, L))
    L_dot_L = sum(L_i * L_i for L_i in L)

    if L_dot_L == 0:
        raise ValueError("Line vector L cannot be the zero vector.")

    scalar = v_dot_L / L_dot_L
    return [round(scalar * L_i, 3) for L_i in L]