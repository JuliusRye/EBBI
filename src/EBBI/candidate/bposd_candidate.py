import jax.numpy as jnp
from stimbposd import BPOSD
from stimbposd.config import (
    DEFAULT_MAX_BP_ITERS,
    DEFAULT_BP_METHOD,
    DEFAULT_OSD_ORDER,
    DEFAULT_OSD_METHOD,
)
from EBBI.candidate.base_class import Candidate
from EBBI.qecc.compare_stim_noise_models import compare_noise_profiles
from EBBI.qecc.base_class import QECC

class BPOSDCandidate(Candidate):

    def __init__(
        self,
        clifford_deformation: jnp.ndarray,
        seed: int,
        qecc_x: QECC,
        qecc_z: QECC,
        rounds: int,
        noise_type: str,
        max_bp_iters: int = DEFAULT_MAX_BP_ITERS,
        bp_method: str = DEFAULT_BP_METHOD,
        osd_order: int = DEFAULT_OSD_ORDER,
        osd_method: str = DEFAULT_OSD_METHOD,
    ):
        super().__init__(clifford_deformation, seed)

        self.noise_type = noise_type
        self.max_bp_iters = max_bp_iters
        self.bp_method = bp_method
        self.osd_order = osd_order
        self.osd_method = osd_method
        match self.noise_type:
            case 'circuit-level':
                self.stim_circuit_x = qecc_x.stim_circuit_level(clifford_deformation, rounds=rounds)
                self.stim_circuit_z = qecc_z.stim_circuit_level(clifford_deformation, rounds=rounds)
            case 'code-capacity':
                self.stim_circuit_x = qecc_x.stim_code_capacity(clifford_deformation)
                self.stim_circuit_z = qecc_z.stim_code_capacity(clifford_deformation)
            case _:
                raise ValueError(f"Unknown noise type: {self.noise_type}, options=[circuit-level, code-capacity]")

        assert compare_noise_profiles(self.stim_circuit_x, self.stim_circuit_z), "The noise profiles of the X and Z basis circuits must be the same otherwise the trick for estimating the true LER will not work and we will isntead get LER + p_y."

        self.sampler_x = self.stim_circuit_x.compile_detector_sampler(seed=self.seed)
        self.sampler_z = self.stim_circuit_z.compile_detector_sampler(seed=self.seed)
        # BP+OSD needs no graphlike decomposition, which is the entire reason this candidate
        # exists. Decomposing here would silently turn it into a slower, worse PyMatching.
        # `approximate_disjoint_errors=True` is unavoidable: the noise models in this repo emit
        # PAULI_CHANNEL_2 instructions and stim refuses to build *any* DEM from those without it.
        # It is orthogonal to `decompose_errors` - it splits correlated Pauli channels into
        # independent mechanisms but leaves the resulting hyperedges intact - so the undecomposed
        # advantage over PyMatching is preserved.
        self.dem_x = self.stim_circuit_x.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
        self.dem_z = self.stim_circuit_z.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
        self.decoder_x = BPOSD(self.dem_x, max_bp_iters=self.max_bp_iters, bp_method=self.bp_method, osd_order=self.osd_order, osd_method=self.osd_method)
        self.decoder_z = BPOSD(self.dem_z, max_bp_iters=self.max_bp_iters, bp_method=self.bp_method, osd_order=self.osd_order, osd_method=self.osd_method)

    def _sample(self, shots: int) -> None:
        # Do the sampling and decoding for the X-basis (estimating pz + py error rate)
        detections_x, observables_x = self.sampler_x.sample(shots, separate_observables=True)
        predictions_x = self.decoder_x.decode_batch(detections_x)
        # Do the sampling and decoding for the Z-basis (estimating px + py error rate)
        detections_z, observables_z = self.sampler_z.sample(shots, separate_observables=True)
        predictions_z = self.decoder_z.decode_batch(detections_z)
        # Count the number of errors for each circuit
        x_basis_circuit_observable_flip = predictions_x != observables_x
        z_basis_circuit_observable_flip = predictions_z != observables_z
        # Update the successful and failed shot counts
        self.logical_I += int(jnp.sum(~x_basis_circuit_observable_flip & ~z_basis_circuit_observable_flip))
        self.logical_x += int(jnp.sum(~x_basis_circuit_observable_flip & z_basis_circuit_observable_flip))
        self.logical_z += int(jnp.sum(x_basis_circuit_observable_flip & ~z_basis_circuit_observable_flip))
        self.logical_y += int(jnp.sum(x_basis_circuit_observable_flip & z_basis_circuit_observable_flip))
