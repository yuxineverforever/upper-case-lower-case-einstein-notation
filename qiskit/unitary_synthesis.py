from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
import numpy as np
from cuquantum import CircuitToEinsum

# Step 1: Define the original circuit
original_circuit = QuantumCircuit(3)
original_circuit.h(0)
original_circuit.cx(0, 1)
original_circuit.ry(np.pi / 4, 2)
original_circuit.cx(1, 2)

print(original_circuit.draw())

# Step 2: Get the unitary matrix of the entire circuit
unitary_matrix = Operator(original_circuit).data

# Step 3: Create a custom gate from the unitary matrix
from qiskit.circuit.library.generalized_gates import UnitaryGate
composite_gate = UnitaryGate(unitary_matrix, label="Composite Gate")

# Step 4: Create a new circuit with the single composite gate
new_circuit = QuantumCircuit(3)
new_circuit.append(composite_gate, [0, 1, 2])
converter = CircuitToEinsum(new_circuit, backend='cupy')
expression, operands = converter.state_vector()
print(expression)
print(operands)
# Visualize the circuit
print(new_circuit.draw())
