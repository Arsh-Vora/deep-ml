import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here

	#Standardization like this
	mean = np.mean(data ,axis =0)
	std = np.std(data , axis =0)
	standardized = (data-mean) /std

	#Min Max like this
	min_val = np.min(data, axis=0)
	max_val = np.max(data, axis=0)
	min_max = (data - min_val)/ (max_val - min_val)

	standardized_data = np.round(standardized, 4)
	normalized_data = np.round(min_max, 4)

	return standardized_data, normalized_data