
def z_biased_SI1000_inspired_noise(error_probability: float, bias: float) -> dict[str, str]:
    """
    A modified version of the SI1000 noise-model from https://quantum-journal.org/papers/q-2021-12-20-605/ that incorporates Z-bias noise.
    """
    p = error_probability
    eta = bias
    # Single qubit base factors
    single_dom = eta / (eta + 1)
    single_rem = 1 / (2 * (eta + 1))
    single = [single_rem, single_rem, single_dom]
    # Construct the noise model
    noise_model = {
        "CZ": f"PAULI_CHANNEL_1({', '.join(str(p*f/2) for f in single)}) ", # Divide by 2 to keep the same overall error probability as the original SI1000 noise model (1-p/2)^2 ~ 1-p for small p (p^2 ~ 0)
        "1Q": f"PAULI_CHANNEL_1({', '.join(str(p*f/10) for f in single)}) ",
        "Init": f"X_ERROR({2*p}) ",
        "Meas": f"X_ERROR({5*p}) ",
        "Rounds": f"PAULI_CHANNEL_1({', '.join(str(2*p*f) for f in single)}) "
    }
    return noise_model

def z_biased_code_capacity_noise(error_probability: float, bias: float) -> dict[str, str]:
    """
    A modified version of the SI1000 noise-model from https://quantum-journal.org/papers/q-2021-12-20-605/ that incorporates Z-bias noise.
    """
    p = error_probability
    eta = bias
    # Single qubit base factors
    single_dom = eta / (eta + 1)
    single_rem = 1 / (2 * (eta + 1))
    single = [single_rem, single_rem, single_dom]
    # Construct the noise model
    noise_model = {
        "Rounds": f"PAULI_CHANNEL_1({', '.join(str(p*f) for f in single)}) "
    }
    return noise_model

def SDEM3_noise(error_probability: float, bias: float) -> dict[str, str]:
    """
    The SDEM3 noise-model [Setiawan, F., McLauchlan, C. Tailoring dynamical codes for biased noise: the X3Z3 Floquet code. npj Quantum Inf 11, 149 (2025). https://doi.org/10.1038/s41534-025-01074-1 ] Methods -> Noise models -> Table 2 and 3 (page 11).
    """
    p = error_probability
    eta = bias
    zeta = 3/5 * (eta / (eta + 1))**2 + 2/5 * (eta / (eta + 1))
    # Single qubit base factors
    single_dom = eta / (eta + 1)
    single_rem = 1 / (2 * (eta + 1))
    single = [single_rem, single_rem, single_dom]
    # two qubit base factors
    two_dom = zeta / 3
    two_rem = (1 - zeta) / 12
    two = [two_rem, two_rem, two_dom, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_dom, two_rem, two_rem, two_dom]
    # Construct the noise model
    noise_model = {
        "CZ": f"PAULI_CHANNEL_2({', '.join(str(p*f) for f in two)}) ",
        "1Q": f"PAULI_CHANNEL_1({', '.join(str(p*f/10) for f in single)}) ",
        "Init": f"X_ERROR({2*p}) ",
        "Meas": f"X_ERROR({5*p}) ",
        "Rounds": f"PAULI_CHANNEL_1({', '.join(str(2*p*f) for f in single)}) "
    }
    return noise_model

def HBD_noise(error_probability: float, bias: float) -> dict[str, str]:
    """
    The HBD noise-model from https://journals.aps.org/prapplied/abstract/10.1103/q7w6-nljp
    """
    p = error_probability
    eta = bias
    # Single qubit base factors
    single_dom = eta / (eta + 1)
    single_rem = 1 / (2 * (eta + 1))
    single = [single_rem, single_rem, single_dom]
    # two qubit base factors
    two_dom = eta / (3*(1+eta))
    two_rem = 1 / (12*(1+eta))
    two = [two_rem, two_rem, two_dom, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_dom, two_rem, two_rem, two_dom]
    # Construct the noise model
    noise_model = {
        "CZ": f"PAULI_CHANNEL_2({', '.join(str(p*f) for f in two)}) ",
        "1Q": f"DEPOLARIZE1({p}) ",
        "Init": f"X_ERROR({p}) ",
        "Meas": f"X_ERROR({p}) ",
        "Rounds": f"PAULI_CHANNEL_1({', '.join(str(p*f) for f in single)}) "
    }
    return noise_model

def MHBD_noise(error_probability: float, bias: float) -> dict[str, str]:
    """
    The HBD noise-model from https://journals.aps.org/prapplied/abstract/10.1103/q7w6-nljp with modified error strengths similar to SI1000 https://quantum-journal.org/papers/q-2021-12-20-605/
    """
    p = error_probability
    eta = bias
    # Single qubit base factors
    single_dom = eta / (eta + 1)
    single_rem = 1 / (2 * (eta + 1))
    single = [single_rem, single_rem, single_dom]
    # two qubit base factors
    two_dom = eta / (3*(1+eta))
    two_rem = 1 / (12*(1+eta))
    two = [two_rem, two_rem, two_dom, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_rem, two_dom, two_rem, two_rem, two_dom]
    # Construct the noise model
    noise_model = {
        "CZ": f"PAULI_CHANNEL_2({', '.join(str(p*f) for f in two)}) ",
        "1Q": f"DEPOLARIZE1({p/10}) ",
        "Init": f"X_ERROR({2*p}) ",
        "Meas": f"X_ERROR({5*p}) ",
        "Rounds": f"PAULI_CHANNEL_1({', '.join(str(2*p*f) for f in single)}) "
    }
    return noise_model

def SI1000_noise(error_probability: float) -> dict[str, str]:
    """
    The SI1000 noise-model from https://quantum-journal.org/papers/q-2021-12-20-605/
    """
    p = error_probability

    noise_model = {
        "CZ": f"DEPOLARIZE2({p}) ",
        "1Q": f"DEPOLARIZE1({p/10}) ",
        "Init": f"X_ERROR({2*p}) ",
        "Meas": f"X_ERROR({5*p}) ",
        "Rounds": f"DEPOLARIZE1({2*p}) "
    }

    return noise_model

def depolarization_noise(error_probability: float) -> dict[str, str]:
    """
    A simple circuit-level depolarization noise-model.
    """
    p = error_probability

    noise_model = {
        "CZ": f"DEPOLARIZE2({p}) ",
        "1Q": f"DEPOLARIZE1({p}) ",
        "Init": f"X_ERROR({p}) ",
        "Meas": f"X_ERROR({p}) ",
        "Rounds": f"DEPOLARIZE1({p}) "
    }

    return noise_model

def get_noise_model(noise_model_name: str, error_probability: float, bias: float | None) -> tuple[dict[str, str], str]:
    """Factory function to get the desired noise model based on the provided name and parameters.

    Args:
        noise_model_name (str): The name of the noise model to retrieve. Options are:

            - `z_biased_SI1000_inspired_noise` (circuit-level, requires `bias`): SI1000-like noise
              where every depolarizing channel is replaced by a Z-biased Pauli channel. The two-qubit
              gate error is applied as a single-qubit channel on each qubit with half the strength.
            - `z_biased_code_capacity_noise` (code-capacity, requires `bias`): a single Z-biased
              Pauli channel applied between rounds, with no gate, init or measurement errors.
            - `SDEM3_noise` (circuit-level, requires `bias`): the SDEM3 noise-model from
              Setiawan & McLauchlan, npj Quantum Inf 11, 149 (2025), Methods -> Noise models,
              Tables 2 and 3. Two-qubit errors use the bias-dependent factor
              `zeta = 3/5 * (eta/(eta+1))**2 + 2/5 * eta/(eta+1)`.
            - `SI1000_noise` (circuit-level, no `bias`): the unbiased SI1000 noise-model from
              https://quantum-journal.org/papers/q-2021-12-20-605/.
            - `depolarization_noise` (circuit-level, no `bias`): uniform circuit-level depolarizing
              noise where every channel has strength `error_probability`.
            - `HBD_noise` (circuit-level, requires `bias`): the hybrid biased-depolarizing model from
              https://journals.aps.org/prapplied/abstract/10.1103/q7w6-nljp, i.e. biased two-qubit and
              idling channels combined with unbiased single-qubit gate, init and measurement errors,
              all at strength `error_probability`.
            - `MHBD_noise` (circuit-level, requires `bias`): the modified HBD model, identical to
              `HBD_noise` but with SI1000-like relative error strengths (1Q gates at `p/10`,
              init at `2p`, measurement at `5p`, idling at `2p`).

        error_probability (float): The error probability to use in the noise model.
        bias (float | None): The Z-bias `eta` to use in the noise model if applicable. Required for
            z_biased_SI1000_inspired_noise, z_biased_code_capacity_noise, SDEM3_noise, HBD_noise and
            MHBD_noise; ignored by SI1000_noise and depolarization_noise.

    Returns:
        tuple(dict[str, str], str): A tuple containing the noise model dictionary and a string representation the type of noise (circuit-level or code-capacity).

    Raises:
        ValueError: If `noise_model_name` is not one of the options listed above.
    """
    match noise_model_name:
        case 'z_biased_SI1000_inspired_noise':
            assert bias is not None, "The z_biased_SI1000_inspired_noise model requires a bias parameter."
            return z_biased_SI1000_inspired_noise(error_probability=error_probability, bias=bias), "circuit-level"
        case 'z_biased_code_capacity_noise':
            assert bias is not None, "The z_biased_code_capacity_noise model requires a bias parameter."
            return z_biased_code_capacity_noise(error_probability=error_probability, bias=bias), "code-capacity"
        case 'SDEM3_noise':
            assert bias is not None, "The SDEM3_noise model requires a bias parameter."
            return SDEM3_noise(error_probability=error_probability, bias=bias), "circuit-level"
        case 'SI1000_noise':
            return SI1000_noise(error_probability=error_probability), "circuit-level"
        case 'depolarization_noise':
            return depolarization_noise(error_probability=error_probability), "circuit-level"
        case 'HBD_noise':
            assert bias is not None, "The hybrid biased-depolarizing (HBD) model requires a bias parameter."
            return HBD_noise(error_probability=error_probability, bias=bias), "circuit-level"
        case 'MHBD_noise':
            assert bias is not None, "The modified hybrid biased-depolarizing (MHBD) model requires a bias parameter."
            return MHBD_noise(error_probability=error_probability, bias=bias), "circuit-level"
        case _:
            raise ValueError(f"Unknown noise model: {noise_model_name}, options=[z_biased_SI1000_inspired_noise, z_biased_code_capacity_noise, SDEM3_noise, SI1000_noise, depolarization_noise, HBD_noise, MHBD_noise]")