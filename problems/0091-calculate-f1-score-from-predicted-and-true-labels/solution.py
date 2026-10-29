def calculate_f1_score(y_true: list[int], y_pred: list[int]) -> float:
    """
    Calculate the F1 score from true and predicted binary labels,
    rounded to the 3rd decimal place.
    """
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 3)