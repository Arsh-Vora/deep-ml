import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """

    g_der = np.polyder(g_coeffs)
    h_der = np.polyder(h_coeffs)

    g_val = np.polyval(g_coeffs, x)
    h_val = np.polyval(h_coeffs, x)
    g_prime = np.polyval(g_der, x)
    h_prime = np.polyval(h_der, x)

    result = (g_prime * h_val - g_val * h_prime) / (h_val ** 2)
    
    return float(result)