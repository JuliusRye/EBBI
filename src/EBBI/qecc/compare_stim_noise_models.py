import stim

def get_noise_fingerprint(circuit: stim.Circuit) -> list[str]:
    """
    Generates a fingerprint of the noise instructions present in a Stim circuit.

    Args:
        circuit (stim.Circuit): The Stim circuit to analyze.
    
    Returns:
        list[str]: A list of strings representing the noise instructions in the circuit in the order they appear.
    """
    # Noise instructions to consider for the fingerprint
    noise_ops = [
        "CORRELATED_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "II_ERROR",
        "I_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    ]
    fingerprint = []
    for inst in circuit:
        if inst.name in noise_ops:
            # Create a string representation of the instruction with its arguments and targets
            fingerprint.append(f"{inst.name}({' '.join([str(arg) for arg in inst.gate_args_copy()])}) {' '.join([str(t.value) for t in inst.targets_copy()])}")
        if inst.name == "REPEAT":
            # Handle REPEAT blocks recursively
            # Only adds the fingerprint for a single iteration of the repeat block
            # as all iterations would have the same fingerprint and thus be redundant
            sub_circuit = inst.body_copy()
            sub_fingerprint = get_noise_fingerprint(sub_circuit)
            fingerprint.extend(sub_fingerprint)
    return fingerprint

def compare_noise_profiles(circuit1: stim.Circuit, circuit2: stim.Circuit, print_comparison: bool = False, column_width: int = 100) -> bool:
    """
    Compares the noise fingerprints of two Stim circuits.

    Args:
        circuit1 (stim.Circuit): The first Stim circuit to compare.
        circuit2 (stim.Circuit): The second Stim circuit to compare.
        print_comparison (bool): If True, prints the noise fingerprints of both circuits.

    Returns:
        bool: True if the noise fingerprints are identical, False otherwise.
    """
    fingerprint1 = get_noise_fingerprint(circuit1)
    fingerprint2 = get_noise_fingerprint(circuit2)

    identical = fingerprint1 == fingerprint2

    if print_comparison:
        print(f"{'Noise-models are identical' if identical else 'Noise-models have one or more differences':^{(7 + column_width * 2 + 3)}}")
        print("-" * (5 + column_width * 2 + 3))
        print(f"   | {'Circuit 1':^{column_width}} | {'Circuit 2':^{column_width}}")
        print("-" * (5 + column_width * 2 + 3))
        for i in range(max(len(fingerprint1), len(fingerprint2))):
            x_op = fingerprint1[i] if i < len(fingerprint1) else ""
            z_op = fingerprint2[i] if i < len(fingerprint2) else ""
            print(f" {'✔' if x_op == z_op else '✘'} | {x_op if len(x_op) < column_width else x_op[:column_width-3]+'...':<{column_width}} | {z_op if len(z_op) < column_width else z_op[:column_width-3]+'...':<{column_width}}")

    return identical