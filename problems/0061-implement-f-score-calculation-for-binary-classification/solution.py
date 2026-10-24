import numpy as np

def f_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate confusion matrix components
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    
    if denominator == 0.0:
        return 0.0
    
    score = (1.0 + beta_sq) * (precision * recall) / denominator
    return round(float(score), 3)