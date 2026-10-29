import math

def single_neuron_model(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    bias: float
) -> tuple[list[float], float]:

    probabilities = []
    
    for row in features:
        z = sum(w * x for w, x in zip(weights, row)) + bias
        prob = 1.0 / (1.0 + math.exp(-z))
        probabilities.append(round(prob, 4))
    
    mse = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
    
    return probabilities, round(mse, 4)