from quantum_classifier import classification_probability
import numpy as np

def cross_entropy_loss(classification, expected):
    
    p = classification.get(expected)
    return -np.log(p + 1e-10)

def cost_function(data, labels, variational):

    classifications = classification_probability(data, variational)
    cost = 0
    for i, classification in enumerate(classifications):
        cost += cross_entropy_loss(classification, labels[i])
    
    cost /= len(data)
    return cost