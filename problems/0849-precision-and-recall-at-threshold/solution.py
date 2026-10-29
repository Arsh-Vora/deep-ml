def precision_recall_at_threshold(
    y_true: list[int],
    y_scores: list[float],
    threshold: float
) -> list[float]:
    """
    Compute precision and recall of a binary classifier at a given threshold.

    Parameters:
        y_true: List of true binary labels (0 or 1).
        y_scores: List of predicted probabilities/scores in [0, 1].
        threshold: Decision threshold for predicting positive class (>= threshold).

    Returns:
        list[float]: [precision, recall] rounded to 4 decimal places.
    """
    tp = 0
    fp = 0
    actual_positives = 0
    predicted_positives = 0

    for true_label, score in zip(y_true, y_scores):
        pred = 1 if score >= threshold else 0

        if true_label == 1:
            actual_positives += 1

        if pred == 1:
            predicted_positives += 1
            if true_label == 1:
                tp += 1
            else:
                fp += 1

    precision = round(tp / predicted_positives, 4) if predicted_positives > 0 else 0.0
    recall = round(tp / actual_positives, 4) if actual_positives > 0 else 0.0

    return [precision, recall]