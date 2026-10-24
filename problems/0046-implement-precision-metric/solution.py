import numpy as np

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # True Positives: predicted 1 and actual 1
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    # False Positives: predicted 1 and actual 0
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    # Handle zero division if no positive predictions were made
    if tp + fp == 0:
        return 0.0
        
    return float(tp / (tp + fp))