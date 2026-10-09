import numpy as np

def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	ar_a = np.array(a)
	ar_b = np.array(b)

	if ar_a.shape != ar_b.shape:
		return -1
	return (ar_a + ar_b).tolist()
	