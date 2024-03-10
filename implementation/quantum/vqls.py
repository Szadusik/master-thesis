# Pennylane
import json
import pennylane as qml

from pennylane import numpy as np
import importlib

# TODO: Pass parameters in configuration e.x JSON file

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
    def __init__(self, weights=[1.0, 0.2, 0.2]) -> None:
        self.weights = weights # Coefficients of the linear combination A = c_0 A_0 + c_1 A_1 ...
        self.n_qubits = 3  # Number of system qubits.
        self.n_shots = 10 ** 6  # Number of quantum measurements.
        self.tot_qubits = self.n_qubits + 1  # Addition of an ancillary qubit.
        self.ancilla_idx = self.n_qubits  # Index of the ancillary qubit (last position).
        self.steps = 30  # Number of optimization steps
        self.eta = 0.8  # Learning rate
        self.q_delta = 0.001  # Initial spread of random quantum weights
        self.rng_seed = 0  # Seed for random number generator
        self.dev_mu = qml.device("lightning.qubit", wires=self.tot_qubits)


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

        mu_real = local_hadamard_test(self, weights, l=l, lp=lp, j=j, part="Re")
        mu_imag = local_hadamard_test(self, weights, l=l, lp=lp, j=j, part="Im")

        return mu_real + 1.0j * mu_imag
    
    
    def psi_norm(self, weights):
        """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
        norm = 0.0

        for l in range(0, len(self.weights)):
            for lp in range(0, len(c)):
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
    
    @staticmethod
    def get_parameters():
        with open('/Users/lukaszsochacki/Desktop/Studia/Magisterka/master-thesis/implementation/quantum/parameters.json', 'r') as j:
            contents = json.loads(j.read())
        return contents


parameters = VQLS.get_parameters()
dev_mu = qml.device("lightning.qubit", wires=parameters['tot_qubits'])
dev_x = qml.device("lightning.qubit", wires=parameters['n_qubits'], shots=parameters['n_shots'])

@qml.qnode(dev_mu, interface="autograd")
def local_hadamard_test(solver: VQLS, weights, l=None, lp=None, j=None, part=None):
    # First Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=parameters['ancilla_idx'])

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
    # phase gate.
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=parameters['ancilla_idx'])

    # Variational circuit generating a guess for the solution vector |x>
    solver.variational_block(weights)

    # Controlled application of the unitary component A_l of the problem matrix A.
    solver.CA(l)

    # Adjoint of the unitary U_b associated to the problem vector |b>.
    # In this specific example Adjoint(U_b) = U_b.
    solver.U_b()

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        qml.CZ(wires=[parameters['ancilla_idx'], j])

    # Unitary U_b associated to the problem vector |b>.
    solver.U_b()

    # Controlled application of Adjoint(A_lp).
    # In this specific example Adjoint(A_lp) = A_lp.
    solver.CA(lp)

    # Second Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=parameters['ancilla_idx'])

    # Expectation value of Z for the ancillary qubit.
    return qml.expval(qml.PauliZ(wires=parameters['ancilla_idx']))


@qml.qnode(dev_x, interface="autograd")
def prepare_and_sample(solver : VQLS, weights):

    # Variational circuit generating a guess for the solution vector |x>
    solver.variational_block(weights)

    # We assume that the system is measured in the computational basis.
    # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
    # this will be repeated for the total number of shots provided (n_shots)
    return qml.sample()


