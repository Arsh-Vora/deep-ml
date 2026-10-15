import numpy as np

def dummy_regressor(y_train, n_test: int, strategy: str = 'mean', constant=None, quantile=None) -> list:
    y = np.asarray(y_train)
    
    if strategy == 'mean':
        p = np.mean(y)
        
    elif strategy == 'median':
        p = np.median(y)
        
    elif strategy == 'quantile':
        if quantile is None or not (0.0 <= quantile <= 1.0):
            raise ValueError("quantile must be a float between 0 and 1")
        p = np.quantile(y, quantile)
        
    elif strategy == 'constant':
        if constant is None:
            raise ValueError("constant must be provided for 'constant' strategy")
        p = constant
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    return [float(p)] * n_test