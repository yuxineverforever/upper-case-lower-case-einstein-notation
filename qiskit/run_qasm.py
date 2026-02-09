from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate
from cuquantum import contract, CircuitToEinsum
# Read in the QASM circuit and remove the measurement gates
qc = QuantumCircuit.from_qasm_file('./QASMBench/medium/multiply_n13/multiply_n13.qasm')
qc.remove_final_measurements()

# Get the unitary matrix of the circuit
unitary_backend = Aer.get_backend("unitary_simulator")
qc_transpiled = transpile(qc, unitary_backend)
job = unitary_backend.run(qc_transpiled)  # Remove assemble() call
unitary_result = job.result()
unitary_matrix = unitary_result.get_unitary()  # Remove qc_transpiled argument

# Generate the initial statevector of the circuit
statevector_backend = Aer.get_backend("statevector_simulator")
zero_qc = QuantumCircuit(qc.num_qubits)
statevector = statevector_backend.run(zero_qc).result().get_statevector()

# Create a new circuit with the unitary as a single gate
new_qc = QuantumCircuit(qc.num_qubits)
unitary_gate = UnitaryGate(unitary_matrix)
new_qc.append(unitary_gate, range(qc.num_qubits))
new_qc.prepare_state(statevector,new_qc.qubits)

# Get the circuit einsum
converter = CircuitToEinsum(new_qc, backend='cupy')
expression, operands = converter.state_vector()

print(expression)
# Print shape of each operand
sv = contract(expression, *operands)
print(sv.shape)