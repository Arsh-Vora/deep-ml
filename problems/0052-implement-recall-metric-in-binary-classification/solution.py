import numpy as np

def recall(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # True Positives: predicted 1 and actual 1
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    # False Negatives: predicted 0 and actual 1
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Denominator: TP + FN equals the total actual positives np.sum(y_true == 1)
    actual_positives = tp + fn
    
    if actual_positives == 0:
        return 0.0
        
    return float(tp / actual_positives)