import math
from collections import Counter

def dummy_classifier(y_train, n_test: int, strategy, constant=None) -> list:
    if strategy == "constant":
        return [constant] * n_test

    counts = Counter(y_train)
    classes = sorted(counts.keys())

    if strategy == "most_frequent":
        best = min(classes, key=lambda c: (-counts[c], c))
        return [best] * n_test

    if strategy == "uniform":
        return [classes[i % len(classes)] for i in range(n_test)]

    if strategy == "stratified":
        n_train = len(y_train)
        exact = {c: n_test * (counts[c] / n_train) for c in classes}
        alloc = {c: math.floor(exact[c]) for c in classes}
        
        # Give remaining slots to highest fractional leftovers
        order = sorted(classes, key=lambda c: (-(exact[c] - alloc[c]), c))
        for c in order[:n_test - sum(alloc.values())]:
            alloc[c] += 1

        return [c for c in classes for _ in range(alloc[c])]