import jax.numpy as jnp

def str_to_cd(
    cd_str: str,
):
    cd = jnp.array([int(x) for x in cd_str.strip()])
    assert cd.min() >= 0 and cd.max() <= 5, "Clifford deformation indices must be between 0 and 5 inclusive."
    return cd

def cd_to_str(
    cd: jnp.ndarray,
):
    cd_str = ''.join([str(int(x)) for x in cd])
    return cd_str

def all_cd(
    num_qubits: int,
    active_deformations: jnp.ndarray | list[int],
) -> jnp.ndarray:
    active_deformations = jnp.array(active_deformations, dtype=int)
    cd_counts = active_deformations.shape[0]
    cd_powers = cd_counts ** jnp.arange(num_qubits)
    cd_idx = jnp.arange(cd_counts ** num_qubits)
    cd_indices = (cd_idx[:, None] // cd_powers[None, :]) % cd_counts
    return active_deformations[cd_indices]

def all_str(
    num_qubits: int,
    avaitable_cd_indices: jnp.ndarray,
) -> list[str]:
    clifford_deformations = all_cd(num_qubits, avaitable_cd_indices)
    return [cd_to_str(cd) for cd in clifford_deformations]
