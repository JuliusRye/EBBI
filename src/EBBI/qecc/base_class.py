import stim
import jax.numpy as jnp
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class QECC(ABC):

    def __init__(
        self,
        connection_order: list = None,
        connection_type: list = None,
        data_qubit_coords: jnp.ndarray = None,
        stabilizer_coords: jnp.ndarray = None,
        distance: int = None,
        num_data_qubits: int = None,
        num_logical_qubits: int = None,
    ):
        self.connection_order = connection_order
        self.connection_type = connection_type
        self.data_qubit_coords = data_qubit_coords
        self.stabilizer_coords = stabilizer_coords
        self.distance = distance
        self.num_data_qubits = num_data_qubits
        self.num_logical_qubits = num_logical_qubits

    @abstractmethod
    def stim_circuit_level(self, clifford_deformation: jnp.ndarray, rounds: int) -> stim.Circuit:
        """Generates a Stim circuit for the quantum error-correcting code with the given Clifford deformation.

        Args:
            clifford_deformation (jnp.ndarray): An array representing the Clifford deformation to be applied (size of array must match number of data qubits).
            rounds (int): The number of rounds of error correction to simulate.

        Returns:
            stim.Circuit: The generated Stim circuit.
        """
        pass

    @abstractmethod
    def stim_code_capacity(self, clifford_deformation: jnp.ndarray) -> stim.Circuit:
        """Generates a Stim circuit for the code capacity noise model with the given Clifford deformation.

        Two perfect rounds of stabilizer measurements (first to project into the code space, second to detect errors) are performed, followed by a perfect measurement of the logical operators. Error are only injected on the data qubits between the fist and second round of stabilizer measurements.

        Args:
            clifford_deformation (jnp.ndarray): An array representing the Clifford deformation to be applied (size of array must match number of data qubits).

        Returns:
            stim.Circuit: The generated Stim circuit.
        """
        pass

    def show_base_code(self) -> None:
        """Displays the base code structure in a matplotlib plot."""
        
        plt.figure()
        for i, ls in enumerate(['-', '--', '-.', ':']):
            connections = self.connection_order[i]
            for conn in connections:
                dq_coord = self.data_qubit_coords[conn[1]]
                st_coord = self.stabilizer_coords[conn[0]]
                plt.plot([dq_coord[0], st_coord[0]], [dq_coord[1], st_coord[1]], 'krbg'[self.connection_type[i][conn[1]]] + ls)
        plt.plot(self.data_qubit_coords[:,0], self.data_qubit_coords[:,1], 'ko', label='Data Qubits')
        plt.plot(self.stabilizer_coords[:,0], self.stabilizer_coords[:,1], 'ks', label='Stabilizers')
        for i, ls in enumerate(['-', '--', '-.', ':']):
            plt.plot([], [], 'k' + ls, label=f'Connection timestep {i+1}')
        for i, (c, n) in enumerate(zip(['r', 'g', 'b'], ['X', 'Y', 'Z'])):
            plt.plot([], [], f'-{c}', label=f'Connection type {n}')
        plt.legend(loc=(1.05,.1), frameon=False)
        n = self.num_data_qubits
        k = self.num_logical_qubits
        d = self.distance
        plt.title(f"[[{n}, {k}, {d}]] Base QECC")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.gca().set_aspect('equal')
        plt.show()