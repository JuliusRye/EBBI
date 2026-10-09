# Description

The code in this folder is used for creating the stim circuits that implement arbitrary [Clifford-deformations](https://link.aps.org/doi/10.1103/PRXQuantum.5.010347) on varius Quantum Error Correction Codes (**QECCs**).

## Currently supported codes

| QECC                 | X-basis | Y-basis | Z-basis |
|----------------------|:-------:|:-------:|:-------:|
| Rotated surface code |    ✔    |    ✘    |    ✔    |

## Usage

1) Import the python file for the QECC that you want to use
2) Create the noise-model that should be used using the `noise_models.py` files functions.
3) Initialize the base-class of the QECC and select the basis and noise-model to use
4) Call the class-function for applying a Clifford-deformation represented by a jnp array of ints of shape *#data-qubits*.

#### Available Clifford-deformations

| Clifford-deformation | I | X↔Y | Y↔Z | X↔Z | X→Z→Y→X | X→Y→Z→X |
|----------------------|:-:|:---:|:---:|:---:|:-------:|:-------:|
| Index                | 0 |  1  |  2  |  3  |    4    |    5    |
| Corresponding [stim-gate](https://github.com/quantumlib/Stim/blob/main/doc/gates.md)  | $I$ | $H_{XY}$ | $H_{YZ}$ | $H$ | $C_{ZYX}$ | $C_{XYZ}$ |

## Example usage

```Python
# Import necessary modules
from EBBI.qecc.noise_models import SI1000_noise
from EBBI.qecc.surface_code import SurfaceCode
# Other imports as needed
import jax.numpy as jnp
import stim
# Define the noise model and the surface code
noise_model = SI1000_noise(error_probability=0.001)
code = SurfaceCode(distance=3, basis='X', noise_model=noise_model)
display(code.show_base_code())
# Use a Clifford deformation to create an XZZX-surface code as an example of a Clifford-deformed code. See https://www.nature.com/articles/s41467-021-22274-1 for more details.
clifford_deformation = jnp.array([0,3,0,3,0,3,0,3,0])
circ = code.stim_circuit(clifford_deformation=clifford_deformation)
# Visualize the circuit
display(circ.diagram("timeline-svg"))
display(circ.without_noise().diagram("detslice-with-ops-svg"))
```