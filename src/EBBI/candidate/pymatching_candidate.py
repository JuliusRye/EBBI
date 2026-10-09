import jax.numpy as jnp
from pymatching import Matching
from EBBI.candidate.base_class import Candidate
from EBBI.qecc.compare_stim_noise_models import compare_noise_profiles
from EBBI.qecc.base_class import QECC

class PyMatchingCandidate(Candidate):

    def __init__(
        self,
        clifford_deformation: jnp.ndarray,
        seed: int,
        qecc_x: QECC,
        qecc_z: QECC,
        rounds: int,
        noise_type: str,
        use_corr_mwpm: bool
    ):
        super().__init__(clifford_deformation, seed)

        self.noise_type = noise_type
        self.use_corr_mwpm = use_corr_mwpm
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
        self.dem_x = self.stim_circuit_x.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        self.dem_z = self.stim_circuit_z.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        self.decoder_x = Matching(self.dem_x, enable_correlations=self.use_corr_mwpm)
        self.decoder_z = Matching(self.dem_z, enable_correlations=self.use_corr_mwpm)

    def _sample(self, shots: int) -> None:
        # Do the sampling and decoding for the X-basis (estimating pz + py error rate)
        detections_x, observables_x = self.sampler_x.sample(shots, separate_observables=True)
        predictions_x = self.decoder_x.decode_batch(detections_x, enable_correlations=self.use_corr_mwpm)
        # Do the sampling and decoding for the Z-basis (estimating px + py error rate)
        detections_z, observables_z = self.sampler_z.sample(shots, separate_observables=True)
        predictions_z = self.decoder_z.decode_batch(detections_z, enable_correlations=self.use_corr_mwpm)
        # Count the number of errors for each circuit
        x_basis_circuit_observable_flip = predictions_x != observables_x
        z_basis_circuit_observable_flip = predictions_z != observables_z
        # Update the successful and failed shot counts
        self.logical_I += int(jnp.sum(~x_basis_circuit_observable_flip & ~z_basis_circuit_observable_flip))
        self.logical_x += int(jnp.sum(~x_basis_circuit_observable_flip & z_basis_circuit_observable_flip))
        self.logical_z += int(jnp.sum(x_basis_circuit_observable_flip & ~z_basis_circuit_observable_flip))
        self.logical_y += int(jnp.sum(x_basis_circuit_observable_flip & z_basis_circuit_observable_flip))
