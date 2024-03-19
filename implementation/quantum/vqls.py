# Pennylane
import json
import math
import pennylane as qml

from scipy.linalg import ishermitian
from pennylane import numpy as np
from quantum.utils import pad_array, matrix_into_unitaries, make_symmetric, get_classic_probabilities

# TODO: Replace calls for class properties to a dict with loaded values

# n_qubits = 3  # Number of system qubits.
# n_shots = 10 ** 6  # Number of quantum measurements.
# tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
# ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
# steps = 30  # Number of optimization steps
# eta = 0.8  # Learning rate
# q_delta = 0.001  # Initial spread of random quantum weights
# rng_seed = 0  # Seed for random number generator

# # Coefficients of the linear combination A = c_0 A_0 + c_1 A_1 ...
# c = np.array([1.0, 0.2, 0.2])

class VQLS:
    def __init__(self, 
                 weights=[1.0, 0.2, 0.2],
                 n_qubits = 3,
                 n_shots = 10**6,
                 steps = 30,
                 eta = 0.8,
                 q_delta = 0.001,
                 rng_seed = 0) -> None:
        self.weights = weights # Coefficients of the linear combination A = c_0 A_0 + c_1 A_1 ...
        self.n_qubits = n_qubits  # Number of system qubits.
        self.n_shots = n_shots  # Number of quantum measurements.
        self.tot_qubits = self.n_qubits + 1  # Addition of an ancillary qubit.
        self.ancilla_idx = self.n_qubits  # Index of the ancillary qubit (last position).
        self.steps = steps  # Number of optimization steps
        self.eta = eta  # Learning rate
        self.q_delta = q_delta  # Initial spread of random quantum weights
        self.rng_seed = rng_seed  # Seed for random number generator
        self.dev_mu = qml.device("lightning.qubit", wires=self.tot_qubits)
        self.dev_x = qml.device("lightning.qubit", wires=self.n_qubits, shots=self.n_shots)


    def U_b(self):
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)


    def CA(self, idx):
        """Controlled versions of the unitary components A_l of the problem matrix A."""
        if idx == 0:
            # Identity operation
            None

        elif idx == 1:
            qml.CNOT(wires=[self.ancilla_idx, 0])
            qml.CZ(wires=[self.ancilla_idx, 1])

        elif idx == 2:
            qml.CNOT(wires=[self.ancilla_idx, 0])

    
    def variational_block(self, weights):
        """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
        # We first prepare an equal superposition of all the states of the computational basis.
        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)

        # A very minimal variational circuit.
        for idx, element in enumerate(weights):
            qml.RY(element, wires=idx)


    def mu(self, weights, l=None, lp=None, j=None):
        """Generates the coefficients to compute the "local" cost function C_L."""

        mu_real = self.local_hadamard_test(weights, l, lp, j, "Re")
        mu_imag = self.local_hadamard_test(weights, l, lp, j, "Im")

        return mu_real + 1.0j * mu_imag
    
    
    def psi_norm(self, weights):
        """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
        norm = 0.0

        for l in range(0, len(self.weights)):
            for lp in range(0, len(self.weights)):
                norm = norm + self.weights[l] * np.conj(self.weights[lp]) * self.mu(weights, l, lp, -1)

        return abs(norm)
    

    def cost_loc(self, weights):
        """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
        mu_sum = 0.0
        for l in range(0, len(self.weights)):
            for lp in range(0, len(self.weights)):
                for j in range(0, self.n_qubits):
                    mu_sum = mu_sum + self.weights[l] * np.conj(self.weights[lp]) * self.mu(weights, l, lp, j)

        mu_sum = abs(mu_sum)

        # Cost function C_L
        return 0.5 - 0.5 * mu_sum / (self.n_qubits * self.psi_norm(weights))
    
    
    def optimize_weights(self):
        np.random.seed(self.rng_seed)
        w = self.q_delta * np.random.randn(self.n_qubits)

        opt = qml.GradientDescentOptimizer(self.eta)
        cost_history = []
        for it in range(self.steps):
            w, cost = opt.step_and_cost(self.cost_loc, w)
            #print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
            cost_history.append(cost)

        # Printing results
            
        # plt.style.use("seaborn")
        # plt.plot(cost_history, "g")
        # plt.ylabel("Cost function")
        # plt.xlabel("Optimization steps")
        # plt.show()
            
        return w
    

    def get_quantum_probabilities(self, optimized_weights):
        #First call weight optimization to get valid results !
        raw_samples = self.prepare_and_sample(optimized_weights)

        # convert the raw samples (bit strings) into integers and count them
        samples = []
        for sam in raw_samples:
            samples.append(int("".join(str(bs) for bs in sam), base=2))

        q_probs = np.bincount(samples) / self.n_shots
        return q_probs


    def local_hadamard_test(self, weights, l=None, lp=None, j=None, part=None):
        @qml.qnode(self.dev_mu, interface="autograd")
        def _local_hadamard_test(solver: VQLS, weights, l=None, lp=None, j=None, part=None):
            # First Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=solver.ancilla_idx)

            # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
            # phase gate.
            if part == "Im" or part == "im":
                qml.PhaseShift(-np.pi / 2, wires=solver.ancilla_idx)

            # Variational circuit generating a guess for the solution vector |x>
            solver.variational_block(weights)

            # Controlled application of the unitary component A_l of the problem matrix A.
            solver.CA(l)

            # Adjoint of the unitary U_b associated to the problem vector |b>.
            # In this specific example Adjoint(U_b) = U_b.
            solver.U_b()

            # Controlled Z operator at position j. If j = -1, apply the identity.
            if j != -1:
                qml.CZ(wires=[solver.ancilla_idx, j])

            # Unitary U_b associated to the problem vector |b>.
            solver.U_b()

            # Controlled application of Adjoint(A_lp).
            # In this specific example Adjoint(A_lp) = A_lp.
            solver.CA(lp)

            # Second Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=solver.ancilla_idx)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=solver.ancilla_idx))
        return _local_hadamard_test(self, weights, l, lp, j, part)


    def prepare_and_sample(self, weights):
        @qml.qnode(self.dev_x, interface="autograd")
        def _prepare_and_sample(solver : VQLS, weights):

            # Variational circuit generating a guess for the solution vector |x>
            solver.variational_block(weights)

            # We assume that the system is measured in the computational basis.
            # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
            # this will be repeated for the total number of shots provided (n_shots)
            return qml.sample()
        return _prepare_and_sample(self, weights)
    

    @staticmethod
    def solve_linear_equation(arr: np.array, b: np.array):
        matrix = pad_array(arr)
        make_symmetric(matrix)
        if(ishermitian(matrix)):
            b_padded = np.zeros(matrix.shape[0])
            b_padded[:b.shape[0]] = b
            qubits = math.log2(matrix.shape[0])
            coefs, unitaries = matrix_into_unitaries(matrix)

            vqls = VQLS(weights=coefs, n_qubits=int(qubits))
            optimized_weights = vqls.optimize_weights()

            q_probs = vqls.get_quantum_probabilities(optimized_weights)
            c_probs = get_classic_probabilities(matrix, b_padded)
            #print(f'Classic probs: {c_probs}')
            print(f'Quantum probs: {q_probs}')

