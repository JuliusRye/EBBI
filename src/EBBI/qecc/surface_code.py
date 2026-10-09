from qecsim.models.rotatedplanar import RotatedPlanarCode
import stim
import jax.numpy as jnp
from EBBI.qecc.base_class import QECC

class SurfaceCode(QECC):

    def __init__(self, distance: int, basis: str, noise_model: dict[str, str]) -> None:
        """
        Creates a surface code QECC data structure with the specified distance, logical basis, and noise model which allows for easy construction of Clifford-deformed variants of the surface code.

        Args:
            distance (int): The distance of the surface code (should be a positive odd integer).
            basis (str): The logical basis for the surface code, either "X" or "Z".
            noise_model (dict[str, str]): A dictionary specifying the noise model to be applied to different types of operations in the Stim circuit.
                - "CZ": The stim noise instruction for the two-qubit CZ operations.
                - "1Q": The stim noise instruction for single-qubit operations.
                - "Init": The stim noise instruction for qubit initialization (After reset operations).
                - "Meas": The stim noise instruction for measurements (Before measurements).
                - "Rounds": The stim noise instruction for data qubits during rounds (During measurement and initialization of syndrome qubits).
        """
        super().__init__()

        self.distance = distance
        self.logical_basis = basis
        self.noise_model = noise_model
        code = RotatedPlanarCode(distance, distance)
        
        # Binary Symplectic representation of stabilizers and logicals
        self.stabilizers = code.stabilizers
        self.logical_observable = code.logicals[1] if self.logical_basis == "Z" else code.logicals[0]

        # Number of qubits and stabilizers
        self.num_data_qubits = self.stabilizers.shape[1] // 2
        self.num_stabilizers = self.stabilizers.shape[0]
        self.num_logical_qubits = 1

        # Coordinates of data qubits and stabilizers
        self.data_qubit_coords = jnp.array([(j-.5,i-.5) for j in range(code.size[0]) for i in range(code.size[1])])
        self.stabilizer_coords = jnp.array([(x,y) for y, x in code._plaquette_indices])

        # Stabilizer 2-qubit gates order
        stabilizers = jnp.array(code.stabilizers).astype(jnp.int8)
        stabilizer_qubit, data_qubit = jnp.where(stabilizers[:, :self.num_data_qubits] + stabilizers[:, self.num_data_qubits:] != 0)
        shifted_ofset = (self.stabilizer_coords[stabilizer_qubit] - self.data_qubit_coords[data_qubit] + jnp.array([0.5, 0.5])).astype(jnp.bool)
        z_plaquette = (self.stabilizer_coords[stabilizer_qubit]).sum(axis=1) % 2 == 0 # True for Z plaquettes, False for X plaquettes
        stack_index = (
            # NE connection (First to be applied)
            0 * (~shifted_ofset[:,0] & ~shifted_ofset[:,1]) +
            # SE connection (Second to be applied if z_plaquette otherwise Third)
            (1 * (~z_plaquette) + 2 * z_plaquette) * (shifted_ofset[:,0] & ~shifted_ofset[:,1]) +
            # NW connection (Third to be applied if z_plaquette otherwise Second)
            (2 * (~z_plaquette) + 1 * z_plaquette) * (~shifted_ofset[:,0] & shifted_ofset[:,1]) +
            # SW connection (Last to be applied)
            3 * (shifted_ofset[:,0] & shifted_ofset[:,1])
        )
        stabilizers_stack = jnp.zeros((4, *stabilizers.shape), dtype=jnp.int8) \
            .at[stack_index, stabilizer_qubit, data_qubit].set(stabilizers[stabilizer_qubit, data_qubit]) \
            .at[stack_index, stabilizer_qubit, data_qubit + self.num_data_qubits].set(stabilizers[stabilizer_qubit, data_qubit + self.num_data_qubits])
        self.connection_order = []
        self.connection_type = []
        for stabilizers_sheet in stabilizers_stack:
            controll, target = jnp.where(stabilizers_sheet[:, :self.num_data_qubits] + stabilizers_sheet[:, self.num_data_qubits:] != 0)
            # Connection order for circuit construction (multiple two qubit gates can not involve the same qubit so they should be applied sequentially instead)
            self.connection_order.append(jnp.array([controll, target]).T)
            # Determine connection type: 0 for None 1 for CX, 2 for CZ, 3 for CY
            x, z = stabilizers_sheet.sum(axis=0).reshape(2, self.num_data_qubits).astype(bool)
            self.connection_type.append(jnp.zeros(self.num_data_qubits, dtype=jnp.int8).at[x].add(1).at[z].add(2))
        
        # Detector information
        match self.logical_basis:
            case "X":
                initial_basis = jnp.vstack(
                    [jnp.zeros((self.num_data_qubits, self.num_data_qubits), dtype=int),
                    jnp.eye(self.num_data_qubits, self.num_data_qubits, dtype=int)]
                )
            case "Z":
                initial_basis = jnp.vstack(
                    [jnp.eye(self.num_data_qubits, self.num_data_qubits, dtype=int),
                    jnp.zeros((self.num_data_qubits, self.num_data_qubits), dtype=int)]
                )
            case _:
                raise ValueError("Basis must be 'X' or 'Z'")
        self.first_round_detectors = (stabilizers @ initial_basis).sum(axis=1) == 0

        active = stabilizers[jnp.where(self.first_round_detectors)[0]]
        self.last_round_detectors = active[:, :self.num_data_qubits] | active[:, self.num_data_qubits:]

    def stim_circuit_level(self, clifford_deformation: jnp.ndarray, rounds: int) -> stim.Circuit:
        """
        Construct a Stim QECC circuit for the given QECC data and Clifford deformation.

        Args:
            qec (QECC_Data): The QECC data structure.
            clifford_deformation (jnp.ndarray): Array specifying Clifford deformation for each data qubit.
            rounds (int): The number of rounds to include in the circuit.

        Returns:
            stim.Circuit: The constructed Stim circuit.
        """

        assert clifford_deformation.shape[0] == self.num_data_qubits, "Clifford deformation array must have the same length as the number of data qubits."

        # --------------------- Define Qubit Sets ---------------------
        data = [i for i in range(self.num_data_qubits)]
        meas = [i + self.num_data_qubits for i in range(self.num_stabilizers)]

        data_txt = ' '.join(map(str, data))
        meas_txt = ' '.join(map(str, meas))

        two_op = [
            ["CX", "CZ"], # (default/undeformed surface code)
            ["CY", "CZ"], # (X <-> Y plaquette)
            ["CX", "CY"], # (Z <-> Y plaquette)
            ["CZ", "CX"], # (X <-> Z plaquette)
            ["CZ", "CY"], # (X -> Z -> Y -> X plaquette)
            ["CY", "CX"], # (X -> Y -> Z -> X plaquette)
        ]
        layer_1_4 = [two_op[cd][(i+1)%2] for i, cd in enumerate(clifford_deformation)]
        layer_2_3 = [two_op[cd][(i+0)%2] for i, cd in enumerate(clifford_deformation)]
        cz_tranforms = {"CZ": "I", "CX": "H", "CY": "H_YZ"}
        data_qubit_basis_lookup = [
            "X" if self.logical_basis == "X" else "Z", # X and Z (default/undeformed surface code)
            "Y" if self.logical_basis == "X" else "Z", # Y and Z (X <-> Y plaquette)
            "X" if self.logical_basis == "X" else "Y", # X and Y (Z <-> Y plaquette)
            "Z" if self.logical_basis == "X" else "X", # Z and X (X <-> Z plaquette)
            "Z" if self.logical_basis == "X" else "Y", # Z and Y (X -> Z -> Y -> X plaquette)
            "Y" if self.logical_basis == "X" else "X", # Y and X (X -> Y -> Z -> X plaquette)
        ] # The basis of the data qubit after applying the clifford deformation

        # --------------------- Construct Stim Circuit ---------------------

        def stabilzer_txt():
            # Prepare measurement qubits
            yield "H " + meas_txt
            for i, control_type in enumerate(layer_1_4):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            yield self.noise_model["1Q"] + meas_txt
            yield self.noise_model["1Q"] + ' '.join(str(data[d]) for d, control_type in enumerate(layer_1_4) if control_type != "CZ") # Data qubits with single qubit gates in the first layer
            # yield entry_noise
            yield "TICK"
            yield "CZ " + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[0])
            yield self.noise_model["CZ"] + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[0])
            yield "TICK"
            for i, control_type in enumerate(layer_1_4):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            for i, control_type in enumerate(layer_2_3):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            yield self.noise_model["1Q"] + data_txt # All data qubits has at least one gate and the ones with two can be combined to one (all corresponding to pi/2 x-rotation and virtual z-rotatiosn)
            yield "TICK"
            yield "CZ " + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[1])
            yield self.noise_model["CZ"] + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[1])
            yield "TICK"
            yield "CZ " + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[2])
            yield self.noise_model["CZ"] + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[2])
            yield "TICK"
            for i, control_type in enumerate(layer_2_3):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            for i, control_type in enumerate(layer_1_4):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            yield self.noise_model["1Q"] + data_txt # All data qubits has at least one gate and the ones with two can be combined to one (all corresponding to pi/2 x-rotation and virtual z-rotatiosn)
            yield "TICK"
            yield "CZ " + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[3])
            yield self.noise_model["CZ"] + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[3])
            yield "TICK"
            for i, control_type in enumerate(layer_1_4):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            # Measure measurement qubits
            yield "H " + meas_txt
            yield self.noise_model["1Q"] + meas_txt
            yield self.noise_model["1Q"] + ' '.join(str(data[d]) for d, control_type in enumerate(layer_1_4) if control_type != "CZ") # Data qubits with single qubit gates in the first layer
            yield "TICK"
            yield self.noise_model["Meas"] + meas_txt
            yield "MR " + meas_txt
            yield self.noise_model["Init"] + meas_txt
            yield self.noise_model["Rounds"] + data_txt
            yield "TICK"
        stabilizer_circ = stim.Circuit('\n'.join(stabilzer_txt()))

        def initialization_txt():
            # Apply coordinates to qubits
            yield "\n".join(f"QUBIT_COORDS({x}, {y}) {i}" for i, (x, y) in enumerate(jnp.vstack([self.data_qubit_coords, self.stabilizer_coords])))
            # Prepare data qubits
            yield "R " + data_txt
            yield self.noise_model["Init"] + data_txt
            yield "TICK"
            # basis_change = []
            for q in data:
                if data_qubit_basis_lookup[clifford_deformation[q]] == "X":
                    yield f"H {q}"
                    # basis_change.append(q)
                elif data_qubit_basis_lookup[clifford_deformation[q]] == "Y":
                    yield f"H_YZ {q}"
                    # basis_change.append(q)
            # yield self.noise_model["1Q"] + ' '.join(map(str, basis_change)) # We do not apply noise here to ensure that the X-basis and Z-basis circuits have the same noise profile
            # Prepare measurement qubits
            yield "R " + meas_txt
            yield self.noise_model["Init"] + meas_txt
            yield "TICK"
        initialization_circ = stim.Circuit('\n'.join(initialization_txt()))

        def logical_readout_txt():
            # Measure data qubits
            # basis_change = []
            for q in data:
                if data_qubit_basis_lookup[clifford_deformation[q]] == "X":
                    yield f"H {q}"
                    # basis_change.append(q)
                elif data_qubit_basis_lookup[clifford_deformation[q]] == "Y":
                    yield f"H_YZ {q}"
                    # basis_change.append(q)
            # yield self.noise_model["1Q"] + ' '.join(map(str, basis_change)) # We do not apply noise here to ensure that the X-basis and Z-basis circuits have the same noise profile
            yield "TICK"
            yield self.noise_model["Meas"] + data_txt
            yield "M " + data_txt
        logical_readout_circ = stim.Circuit('\n'.join(logical_readout_txt()))

        # --------------------- Define Detectors and Observables ---------------------

        first_round_detectors = stim.Circuit(
            "\n".join(
                f"DETECTOR rec[-{self.num_stabilizers - i}]" 
                for i, val in enumerate(self.first_round_detectors) if val
            )
        )

        mid_qec_detectors = stim.Circuit(
            "\n".join(
                f"DETECTOR rec[-{self.num_stabilizers - i}] rec[-{2 * self.num_stabilizers - i}]" 
                for i in range(self.num_stabilizers)
            )
        )

        final_round_detectors = stim.Circuit(
            "\n".join(
                f"DETECTOR rec[-{self.num_stabilizers + self.num_data_qubits - i}] " + " ".join(
                    f"rec[-{self.num_data_qubits - j}]" 
                    for j, val in enumerate(self.last_round_detectors[jnp.cumsum(self.first_round_detectors)[i] - 1]) if val
                )
                for i, val in enumerate(self.first_round_detectors) if val
            )
        )

        logical_observable = stim.Circuit(
            "OBSERVABLE_INCLUDE(0) " + " ".join(
                f"rec[-{self.num_data_qubits - i}]" 
                for i, val in enumerate(self.logical_observable[:self.num_data_qubits] | self.logical_observable[self.num_data_qubits:]) if val
            )
        )

        # --------------------- Construct Full Circuit ---------------------

        return (
            initialization_circ + stabilizer_circ + first_round_detectors + 
            (stabilizer_circ + mid_qec_detectors) * (rounds - 1) + 
            logical_readout_circ + final_round_detectors + logical_observable
        )

    def stim_code_capacity(self, clifford_deformation: jnp.ndarray) -> stim.Circuit:
        """
        Construct a Stim QECC circuit for the given QECC data and Clifford deformation.

        Args:
            qec (QECC_Data): The QECC data structure.
            clifford_deformation (jnp.ndarray): Array specifying Clifford deformation for each data qubit.
            rounds (int): The number of rounds to include in the circuit.

        Returns:
            stim.Circuit: The constructed Stim circuit.
        """
        
        assert clifford_deformation.shape[0] == self.num_data_qubits, "Clifford deformation array must have the same length as the number of data qubits."

        # --------------------- Define Qubit Sets ---------------------
        data = [i for i in range(self.num_data_qubits)]
        meas = [i + self.num_data_qubits for i in range(self.num_stabilizers)]

        data_txt = ' '.join(map(str, data))
        meas_txt = ' '.join(map(str, meas))

        two_op = [
            ["CX", "CZ"], # (default/undeformed surface code)
            ["CY", "CZ"], # (X <-> Y plaquette)
            ["CX", "CY"], # (Z <-> Y plaquette)
            ["CZ", "CX"], # (X <-> Z plaquette)
            ["CZ", "CY"], # (X -> Z -> Y -> X plaquette)
            ["CY", "CX"], # (X -> Y -> Z -> X plaquette)
        ]
        layer_1_4 = [two_op[cd][(i+1)%2] for i, cd in enumerate(clifford_deformation)]
        layer_2_3 = [two_op[cd][(i+0)%2] for i, cd in enumerate(clifford_deformation)]
        cz_tranforms = {"CZ": "I", "CX": "H", "CY": "H_YZ"}
        data_qubit_basis_lookup = [
            "X" if self.logical_basis == "X" else "Z", # X and Z (default/undeformed surface code)
            "Y" if self.logical_basis == "X" else "Z", # Y and Z (X <-> Y plaquette)
            "X" if self.logical_basis == "X" else "Y", # X and Y (Z <-> Y plaquette)
            "Z" if self.logical_basis == "X" else "X", # Z and X (X <-> Z plaquette)
            "Z" if self.logical_basis == "X" else "Y", # Z and Y (X -> Z -> Y -> X plaquette)
            "Y" if self.logical_basis == "X" else "X", # Y and X (X -> Y -> Z -> X plaquette)
        ] # The basis of the data qubit after applying the clifford deformation

        # --------------------- Construct Stim Circuit ---------------------

        def stabilzer_txt():
            # Prepare measurement qubits
            yield "H " + meas_txt
            for i, control_type in enumerate(layer_1_4):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            # yield entry_noise
            yield "TICK"
            yield "CZ " + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[0])
            yield "TICK"
            for i, control_type in enumerate(layer_1_4):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            for i, control_type in enumerate(layer_2_3):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            yield "TICK"
            yield "CZ " + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[1])
            yield "TICK"
            yield "CZ " + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[2])
            yield "TICK"
            for i, control_type in enumerate(layer_2_3):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            for i, control_type in enumerate(layer_1_4):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            yield "TICK"
            yield "CZ " + ' '.join(f"{meas[m]} {data[d]}" for m, d in self.connection_order[3])
            yield "TICK"
            for i, control_type in enumerate(layer_1_4):
                transform = cz_tranforms[control_type]
                yield f"{transform} {i}\n" if transform != "I" else ""
            # Measure measurement qubits
            yield "H " + meas_txt
            yield "TICK"
            yield "MR " + meas_txt
            yield "TICK"
        stabilizer_circ = stim.Circuit('\n'.join(stabilzer_txt()))

        def initialization_txt():
            # Apply coordinates to qubits
            yield "\n".join(f"QUBIT_COORDS({x}, {y}) {i}" for i, (x, y) in enumerate(jnp.vstack([self.data_qubit_coords, self.stabilizer_coords])))
            # Prepare data qubits
            yield "R " + data_txt
            yield "TICK"
            basis_change = []
            for q in data:
                if data_qubit_basis_lookup[clifford_deformation[q]] == "X":
                    yield f"H {q}"
                    basis_change.append(q)
                elif data_qubit_basis_lookup[clifford_deformation[q]] == "Y":
                    yield f"H_YZ {q}"
                    basis_change.append(q)
            # Prepare measurement qubits
            yield "R " + meas_txt
            yield "TICK"
        initialization_circ = stim.Circuit('\n'.join(initialization_txt()))

        def logical_readout_txt():
            # Measure data qubits
            for q in data:
                yield f"M{data_qubit_basis_lookup[clifford_deformation[q]].lower()} {q}"
        logical_readout_circ = stim.Circuit('\n'.join(logical_readout_txt()))

        data_qubit_errors = stim.Circuit(self.noise_model["Rounds"] + data_txt + "\nTICK")

        # --------------------- Define Detectors and Observables ---------------------

        mid_qec_detectors = stim.Circuit(
            "\n".join(
                f"DETECTOR rec[-{self.num_stabilizers - i}] rec[-{2 * self.num_stabilizers - i}]" 
                for i in range(self.num_stabilizers)
            )
        )

        logical_observable = stim.Circuit(
            "OBSERVABLE_INCLUDE(0) " + " ".join(
                f"rec[-{self.num_data_qubits - i}]" 
                for i, val in enumerate(self.logical_observable[:self.num_data_qubits] | self.logical_observable[self.num_data_qubits:]) if val
            )
        )

        # --------------------- Construct Code Capacity Circuit ---------------------

        return (
            initialization_circ + stabilizer_circ + data_qubit_errors + # Prepare logical state (Error free)
            stabilizer_circ + mid_qec_detectors + # Add data qubit errors and measure syndrome (Error free)
            logical_readout_circ + logical_observable # Measure logical qubit (Error free)
        )

    def tikz_figure_code(self, clifford_deformation: jnp.ndarray, time_steps=False, gate_type=True, qubit_size_mm=1.0) -> str:
        """
        Generate a LaTeX TikZ figure code for the surface code layout with the specified Clifford deformation.

        Args:
            clifford_deformation (jnp.ndarray): Array specifying Clifford deformation for each data qubit.
            time_steps (bool): Whether to include time steps in the figure.
            gate_type (bool): Whether to include gate types in the figure or just draw connections with a single line.
            qubit_size_mm (float): The size of each qubit in millimeters (Check qubits are half of this size).

        Returns:
            str: A string containing the LaTeX TikZ code for the figure.
        """

        # Connection type: 0 for None 1 for CX, 2 for CZ, 3 for CY
        mapping = jnp.array([
            [0, 1, 2, 3], # I
            [0, 3, 2, 1], # X <-> Y
            [0, 1, 3, 2], # Y <-> Z 
            [0, 2, 1, 3], # X <-> Z 
            [0, 2, 3, 1], # X -> Z -> Y -> X 
            [0, 3, 1, 2], # X -> Y -> Z -> X 
        ])
        cd = jnp.array([int(d) for d in clifford_deformation])

        data_loc = [(x, -y) for x, y in self.data_qubit_coords]
        stab_loc = [(x, -y) for x, y in self.stabilizer_coords]

        gate_loc_0 = [((data_loc[t][0] + stab_loc[c][0])/2, (data_loc[t][1] + stab_loc[c][1])/2) for c, t in self.connection_order[0]]
        gate_ang_0 = [90+180/jnp.pi*jnp.arctan2(data_loc[t][1] - stab_loc[c][1], data_loc[t][0] - stab_loc[c][0]) for c, t in self.connection_order[0]]
        gate_type_all_0 = [m[g] for m, g in zip(mapping[cd], self.connection_type[0])]
        gate_type_0 = [gate_type_all_0[i] for i in self.connection_order[0][:,1]]

        gate_loc_1 = [((data_loc[t][0] + stab_loc[c][0])/2, (data_loc[t][1] + stab_loc[c][1])/2) for c, t in self.connection_order[1]]
        gate_ang_1 = [90+180/jnp.pi*jnp.arctan2(data_loc[t][1] - stab_loc[c][1], data_loc[t][0] - stab_loc[c][0]) for c, t in self.connection_order[1]]
        gate_type_all_1 = [m[g] for m, g in zip(mapping[cd], self.connection_type[1])]
        gate_type_1 = [gate_type_all_1[i] for i in self.connection_order[1][:,1]]

        gate_loc_2 = [((data_loc[t][0] + stab_loc[c][0])/2, (data_loc[t][1] + stab_loc[c][1])/2) for c, t in self.connection_order[2]]
        gate_ang_2 = [90+180/jnp.pi*jnp.arctan2(data_loc[t][1] - stab_loc[c][1], data_loc[t][0] - stab_loc[c][0]) for c, t in self.connection_order[2]]
        gate_type_all_2 = [m[g] for m, g in zip(mapping[cd], self.connection_type[2])]
        gate_type_2 = [gate_type_all_2[i] for i in self.connection_order[2][:,1]]

        gate_loc_3 = [((data_loc[t][0] + stab_loc[c][0])/2, (data_loc[t][1] + stab_loc[c][1])/2) for c, t in self.connection_order[3]]
        gate_ang_3 = [90+180/jnp.pi*jnp.arctan2(data_loc[t][1] - stab_loc[c][1], data_loc[t][0] - stab_loc[c][0]) for c, t in self.connection_order[3]]
        gate_type_all_3 = [m[g] for m, g in zip(mapping[cd], self.connection_type[3])]
        gate_type_3 = [gate_type_all_3[i] for i in self.connection_order[3][:,1]]

        tikz_code = f"""
        % Define colors for Clifford deformations
        \\newcommand{{\\setmycolor}}[1]{{%
            \\ifcase#1\\relax
                \\def\\mycolor{{white}}%      0
            \\or \\def\\mycolor{{LimeGreen}}%  1
            \\or \\def\\mycolor{{BrickRed}}%   2
            \\or \\def\\mycolor{{Cerulean}}%   3
            \\or \\def\\mycolor{{Fuchsia}}%    4
            \\or \\def\\mycolor{{Dandelion}}%  5
            \\else \\def\\mycolor{{gray}}%     fallback
            \\fi
        }}
        \\def\\numofset{{(.1, .02)}}

        % First 2q-gate set
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_0, gate_ang_0, gate_type_0) if typ == 1])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CX}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 1}};' if time_steps else ''}
            \\end{{scope}}
        }}
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_0, gate_ang_0, gate_type_0) if typ == 2])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CZ}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 1}};' if time_steps else ''}
            \\end{{scope}}
        }}
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_0, gate_ang_0, gate_type_0) if typ == 3])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CY}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 1}};' if time_steps else ''}
            \\end{{scope}}
        }}

        % Second 2q-gate set
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_1, gate_ang_1, gate_type_1) if typ == 1])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CX}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 2}};' if time_steps else ''}
            \\end{{scope}}
        }}
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_1, gate_ang_1, gate_type_1) if typ == 2])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CZ}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 2}};' if time_steps else ''}
            \\end{{scope}}
        }}
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_1, gate_ang_1, gate_type_1) if typ == 3])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CY}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 2}};' if time_steps else ''}
            \\end{{scope}}
        }}

        % Third 2q-gate set
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_2, gate_ang_2, gate_type_2) if typ == 1])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CX}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 3}};' if time_steps else ''}
            \\end{{scope}}
        }}
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_2, gate_ang_2, gate_type_2) if typ == 2])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CZ}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 3}};' if time_steps else ''}
            \\end{{scope}}
        }}
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_2, gate_ang_2, gate_type_2) if typ == 3])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CY}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 3}};' if time_steps else ''}
            \\end{{scope}}
        }}

        % Fourth 2q-gate set
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_3, gate_ang_3, gate_type_3) if typ == 1])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CX}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 4}};' if time_steps else ''}
            \\end{{scope}}
        }}
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_3, gate_ang_3, gate_type_3) if typ == 2])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CZ}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 4}};' if time_steps else ''}
            \\end{{scope}}
        }}
        \\foreach \\pos/\\angle in {{{', '.join([f'({x},{y})/{angle}' for (x, y), angle, typ in zip(gate_loc_3, gate_ang_3, gate_type_3) if typ == 3])}}} {{
            \\begin{{scope}}[shift={{(\\pos)}}, rotate=\\angle]
                {'\\input{{CY}}' if gate_type else '\\draw[line width = .5pt] (0,-.25) -- (0,.35);'}
                {'\\node at \\numofset {{\\tiny 4}};' if time_steps else ''}
            \\end{{scope}}
        }}

        % Place data qubits and stabilizers
        \\foreach \\pos/\\cd in {{{', '.join([f'({x},{y})/{cd}' for (x, y), cd in zip(data_loc, clifford_deformation)])}}} {{
            \\setmycolor{{\\cd}}
            \\draw[fill=\\mycolor, line width = .5pt] \\pos circle [radius={qubit_size_mm}mm];
        }}
        \\foreach \\pos in {{{', '.join([f'({x},{y})' for x, y in stab_loc])}}} {{
            \\draw[shift={{\\pos}}, fill] (-{qubit_size_mm / 2}mm,-{qubit_size_mm / 2}mm) rectangle ({qubit_size_mm / 2}mm,{qubit_size_mm / 2}mm);
        }}
        """
        return tikz_code.replace("\n        ", "\n") # Remove leading spaces for proper LaTeX formatting