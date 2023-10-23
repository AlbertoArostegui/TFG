import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, QuantumCircuit
from qiskit.opflow import Z, I
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import CircuitQNN

def parity(x):
    return '{:b}'.format(x).count('1') % 2


data = np.random.rand(400,4)
labels = np.array([1 if np.sum(point) > 1 else 0 for point in data])
training_data = data[:30]
training_labels = labels[:30]
test_data = data[30:]
test_labels = labels[30:]

num_qubits = 2
ansatz = RealAmplitudes(num_qubits, reps=1)
ansatz.decompose().draw('mpl')

observable = Z ^ Z

qnn = CircuitQNN(
    circuit=ansatz,
    input_params=ansatz.parameters,
    weight_params=[],
    interpret=parity,
    output_shape=len(np.unique(labels)),  # Adjust based on the number of classes
    gradient=None,
    quantum_instance=Aer.get_backend('statevector_simulator')
)

classifier = NeuralNetworkClassifier(qnn, optimizer=COBYLA())
classifier.fit(training_data, training_labels)

score = classifier.score(test_data, test_labels)
print(f'Classification accuracy: {score}')
