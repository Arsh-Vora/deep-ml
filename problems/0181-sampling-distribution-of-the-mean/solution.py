import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str) -> float:
    """
    Demonstrate the sampling distribution of the mean.
    
    Parameters:
        num_samples: Number of independent samples to draw.
        sample_size: Size of each individual sample.
        distribution: 'uniform' or 'exponential'.
        
    Returns:
        float: The mean of the sample means.
    """
    sample_means = []
    
    for _ in range(num_samples):
        if distribution == 'uniform':
            sample = np.random.uniform(0, 1, sample_size)
        elif distribution == 'exponential':
            sample = np.random.exponential(1, sample_size)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
            
        sample_means.append(np.mean(sample))
        
    return float(np.mean(sample_means))