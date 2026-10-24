import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check for empty arrays to prevent division by zero
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    
    # Calculate the fraction of matching labels
    return float(np.mean(y_true == y_pred))