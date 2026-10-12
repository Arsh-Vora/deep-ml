import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	grad = np.array(gradient, dtype=float)
	magnitude = float(np.linalg.norm(grad))
	
	if magnitude == 0:
		direction = np.zeros_like(grad)
	else:
		direction = grad / magnitude
	
	descent_direction = -direction
	
	return {
		"magnitude": magnitude,
        "direction": direction.tolist(),
        "descent_direction": descent_direction.tolist()
    }