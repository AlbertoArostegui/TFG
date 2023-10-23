from qiskit.utils import algorithm_globals
import numpy as np
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit import BasicAer, execute
import matplotlib.pyplot as plt


algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)

train_x, train_y, test_x, test_y = (
    ad_hoc_data(training_size=20,
                test_size=5,
                n=2,
                gap=0.3,
                one_hot=False)
)

feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
var_form = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)

ad_hoc_circuit = feature_map.compose(var_form)
ad_hoc_circuit.measure_all()
ad_hoc_circuit.decompose().draw('mpl')

def circuit_instance(data, variational):

    parameters = {}
    for i, p in enumerate(feature_map.ordered_parameters):
        parameters[p] = data[i]
    for i, p in enumerate(var_form.ordered_parameters):
        parameters[p] = variational[i]
    return ad_hoc_circuit.assign_parameters(parameters)

def parity(bitstring):
    hamming_weight = sum(int(k) for k in list(bitstring))
    return (hamming_weight+1) % 2

def label_probability(results):

    shots = sum(results.values())
    probabilities = {0: 0, 1: 0}
    for bitstring, counts in results.items():
        label = parity(bitstring)
        probabilities[label] += counts/shots
    return probabilities

def classification_probability(data, variational):

    circuits = [circuit_instance(d, variational) for d in data]
    backend = BasicAer.get_backend('qasm_simulator')
    results = execute(circuits, backend).result()
    classification = [label_probability(results.get_counts(c)) for c in circuits]
    return classification

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

class OptimizerLog:

    def __init__(self) -> None:
        self.evaluations = []
        self.parameters = []
        self.costs = []
    
    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)

log = OptimizerLog()
optimizer = SPSA(maxiter=100, callback=log.update)

#initial_point = np.random.random(VAR_FORM.num_parameters)
initial_point = np.array([3.28559355, 5.48514978, 5.13099949,
                          0.88372228, 4.08885928, 2.45568528,
                          4.92364593, 5.59032015, 3.66837805,
                          4.84632313, 3.60713748, 2.43546])

def objective_function(variational):
    return cost_function(train_x, train_y, variational)

result = optimizer.minimize(objective_function, initial_point)

opt_var = result.x
opt_value = result.fun

'''fig = plt.figure()
plt.plot(log.evaluations, log.costs)
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.show()'''

def test_classifier(data, labels, variational):

    probability = classification_probability(data, variational)
    predictions = [0 if p[0] >= p[1] else 1 for p in probability]
    accuracy = 0
    for i, prediction in enumerate(predictions):
        if prediction == labels[i]:
            accuracy += 1
    accuracy /= len(labels)
    return accuracy, predictions

accuracy, predictions = test_classifier(test_x, test_y, opt_var)