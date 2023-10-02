import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import AllPairs

# Example data and labels (replace with your own dataset)
data = np.random.rand(100, 2)
print(data)
labels = [0 if np.linalg.norm(x) < 0.7 else 1 for x in data]

feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='full')

quantum_circuit = QuantumCircuit(feature_map.num_qubits)
quantum_circuit.append(feature_map, range(feature_map.num_qubits))

backend = AerSimulator()
quantum_instance = QuantumInstance(backend, shots=1024)

train_size = int(0.7 * len(data))
training_input = {
    0: data[:train_size],
    1: data[train_size:]
}
test_input = {
    0: labels[:train_size],
    1: labels[train_size:]
}

svm = QSVM(feature_map, training_input, test_input, multiclass_extension=AllPairs())
result = svm.run(quantum_instance)

predicted_labels = result['predicted_labels']
accuracy = result['testing_accuracy']

print("Predicted labels:", predicted_labels)
print("Accuracy:", accuracy)
