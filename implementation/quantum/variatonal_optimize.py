from vqls import *
from pennylane import numpy as np

import pennylane as qml
import matplotlib.pyplot as plt

n_qubits = 3  # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 30  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.001  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator

solver = VQLS()
weights = solver.optimize_weights()

#  SOLUTION TEST

# Id = np.identity(2)
# Z = np.array([[1, 0], [0, -1]])
# X = np.array([[0, 1], [1, 0]])

# A_0 = np.identity(8)
# A_1 = np.kron(np.kron(X, Z), Id)
# A_2 = np.kron(np.kron(X, Id), Id)

# A_num = c[0] * A_0 + c[1] * A_1 + c[2] * A_2
# b = np.ones(8) / np.sqrt(8)

# print("A = \n", A_num)
# print("b = \n", b)

# A_inv = np.linalg.inv(A_num)
# x = np.dot(A_inv, b)

# c_probs = (x / np.linalg.norm(x)) ** 2

# dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)

# @qml.qnode(dev_x, interface="autograd")
# def prepare_and_sample(weights):

#     # Variational circuit generating a guess for the solution vector |x>
#     variational_block(weights)

#     # We assume that the system is measured in the computational basis.
#     # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
#     # this will be repeated for the total number of shots provided (n_shots)
#     return qml.sample()

# raw_samples = prepare_and_sample(w)

# # convert the raw samples (bit strings) into integers and count them
# samples = []
# for sam in raw_samples:
#     samples.append(int("".join(str(bs) for bs in sam), base=2))

# q_probs = np.bincount(samples) / n_shots

samples = solver.get_quantum_probabilities(weights)
print(samples)

# print('Comparison')

# print("Classical: x_n^2 =\n", c_probs)
# print("Quantum |<x|n>|^2=\n", q_probs)
# print(f'Classic x: {x}')

# print(f'Quantum x?: {samples}')