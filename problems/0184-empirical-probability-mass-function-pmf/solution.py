def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    samples_list = list(samples)
    if not samples_list:
        return []

    total = len(samples_list)
    counts = {}

    for x in samples_list:
        counts[x] = counts.get(x, 0) + 1

    return [(val, counts[val] / total) for val in sorted(counts.keys())]