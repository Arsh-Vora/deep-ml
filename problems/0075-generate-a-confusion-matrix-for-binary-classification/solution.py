from collections import Counter

def confusion_matrix(data):
    # Count occurrences of each (y_true, y_pred) pair
    counts = Counter((y_true, y_pred) for y_true, y_pred in data)
    
    tp = counts[(1, 1)]
    fn = counts[(1, 0)]
    fp = counts[(0, 1)]
    tn = counts[(0, 0)]
    
    return [
        [tp, fn],
        [fp, tn]
    ]