import jax.numpy as jnp
from beliefmatching import BeliefMatching
from EBBI.candidate.base_class import Candidate
from EBBI.qecc.compare_stim_noise_models import compare_noise_profiles
from EBBI.qecc.base_class import QECC

class BeliefMatchingCandidate(Candidate):

    def __init__(
        self,
        clifford_deformation: jnp.ndarray,
        seed: int,
        qecc_x: QECC,
        qecc_z: QECC,
        rounds: int,
        noise_type: str,
        max_bp_iters: int = 20,
        bp_method: str = "product_sum"
    ):
        super().__init__(clifford_deformation, seed)

        self.noise_type = noise_type
        self.max_bp_iters = max_bp_iters
        self.bp_method = bp_method
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
        # DEM CHOICE: decomposed (decompose_errors=True), same as pymatching_candidate.py.
        # The undecomposed path was checked against the installed beliefmatching==0.2.0 and is
        # UNAVAILABLE there: BeliefMatching.__init__ calls detector_error_model_to_check_matrices(model)
        # without forwarding that function's `allow_undecomposed_hyperedges` flag (its **kwargs are
        # passed on to ldpc.bp_decoder instead), so any error mechanism touching more than two
        # detectors raises "A hyperedge error mechanism was found that was not decomposed into edges."
        # Consequence when reading the LER plots: BP here sees the *decomposed* graphlike error
        # probabilities, not the true hyperedge probabilities, so this decoder only recovers part of
        # the MWPM-to-maximum-likelihood gap under strong Clifford deformation. Reaching the
        # undecomposed path would require a newer beliefmatching than the pin that is shared with
        # the cluster environment.
        self.dem_x = self.stim_circuit_x.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        self.dem_z = self.stim_circuit_z.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        self.decoder_x = BeliefMatching(self.dem_x, max_bp_iters=self.max_bp_iters, bp_method=self.bp_method)
        self.decoder_z = BeliefMatching(self.dem_z, max_bp_iters=self.max_bp_iters, bp_method=self.bp_method)

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
