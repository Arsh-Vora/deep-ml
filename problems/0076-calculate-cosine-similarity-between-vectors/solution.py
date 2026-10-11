import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	
	# Implement your code here
	dot_product = np.dot(v1,v2)

	norm1 = np.linalg.norm(v1)
	norm2 = np.linalg.norm(v2)

	if norm1 == 0 and norm2 == 0:
		return 0.0

	return round(float(dot_product/(norm1 * norm2)), 3)
