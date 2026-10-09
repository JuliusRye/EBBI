import jax.numpy as jnp
import numpy as np
from tesseract_decoder import tesseract
from EBBI.candidate.base_class import Candidate
from EBBI.qecc.compare_stim_noise_models import compare_noise_profiles
from EBBI.qecc.base_class import QECC

class TesseractCandidate(Candidate):

    def __init__(
        self,
        clifford_deformation: jnp.ndarray,
        seed: int,
        qecc_x: QECC,
        qecc_z: QECC,
        rounds: int,
        noise_type: str,
        det_beam: int = 5,
        beam_climbing: bool = False,
        pqlimit: int = 200_000,
        det_penalty: float = 0.0,
        no_revisit_dets: bool = True,
        merge_errors: bool = True,
    ):
        """
        Args:
            det_beam (int): Beam cutoff giving the maximum number of detection events a search state may have. This is the main accuracy-versus-speed dial (larger is more accurate and slower). Use `tesseract_decoder.tesseract.INF_DET_BEAM` (65535) for no cutoff.
            beam_climbing (bool): Whether to enable the beam climbing heuristic.
            pqlimit (int): Per-shot search budget given as the maximum size of the priority queue. Shots that exhaust it are counted as logical errors.
            det_penalty (float): Extra cost added for each detector visited during the search.
            no_revisit_dets (bool): Whether to prevent the search from revisiting a syndrome pattern more than once.
            merge_errors (bool): Whether to merge error channels with identical syndrome patterns.
        """
        super().__init__(clifford_deformation, seed)

        self.noise_type = noise_type
        self.det_beam = det_beam
        self.beam_climbing = beam_climbing
        self.pqlimit = pqlimit
        self.det_penalty = det_penalty
        self.no_revisit_dets = no_revisit_dets
        self.merge_errors = merge_errors
        # Number of decoding calls where the decoder ran out of search budget (counted as logical errors)
        self.low_confidence_shots = 0

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
        # The DEMs are left undecomposed so that Tesseract sees the hyperedges it is built to handle.
        # `approximate_disjoint_errors=True` is still required: every noise model in qecc/noise_models.py
        # uses PAULI_CHANNEL_1/2 or DEPOLARIZE1/2, and stim refuses to convert those mutually exclusive
        # error channels into independent error mechanisms without it. It does not decompose hyperedges.
        self.dem_x = self.stim_circuit_x.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
        self.dem_z = self.stim_circuit_z.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
        self.decoder_x = self._build_decoder(self.dem_x)
        self.decoder_z = self._build_decoder(self.dem_z)

    def _build_decoder(self, dem) -> tesseract.TesseractDecoder:
        """Builds a Tesseract decoder for the given detector error model using this candidate's settings."""
        config = tesseract.TesseractConfig(
            dem=dem,
            det_beam=self.det_beam,
            beam_climbing=self.beam_climbing,
            no_revisit_dets=self.no_revisit_dets,
            merge_errors=self.merge_errors,
            pqlimit=self.pqlimit,
            det_penalty=self.det_penalty,
        )
        return tesseract.TesseractDecoder(config)

    def _observable_flips(self, sampler, decoder: tesseract.TesseractDecoder, shots: int) -> np.ndarray:
        """Samples and decodes `shots` shots and returns a boolean array flagging the shots where the decoder mispredicted the observables.

        Shots where the decoder exhausted its search budget (`low_confidence_flag`) are counted as
        mispredictions rather than discarded, since discarding them would bias the LER estimate.
        """
        detections, observables = sampler.sample(shots, separate_observables=True)
        predictions = np.empty_like(observables)
        low_confidence = np.zeros(shots, dtype=bool)
        for i in range(shots):
            predictions[i] = decoder.decode(detections[i])
            low_confidence[i] = decoder.low_confidence_flag
        self.low_confidence_shots += int(np.count_nonzero(low_confidence))
        return np.any(predictions != observables, axis=1) | low_confidence

    def _sample(self, shots: int) -> None:
        # Do the sampling and decoding for the X-basis (estimating pz + py error rate)
        x_basis_circuit_observable_flip = self._observable_flips(self.sampler_x, self.decoder_x, shots)
        # Do the sampling and decoding for the Z-basis (estimating px + py error rate)
        z_basis_circuit_observable_flip = self._observable_flips(self.sampler_z, self.decoder_z, shots)
        # Update the successful and failed shot counts
        self.logical_I += int(jnp.sum(~x_basis_circuit_observable_flip & ~z_basis_circuit_observable_flip))
        self.logical_x += int(jnp.sum(~x_basis_circuit_observable_flip & z_basis_circuit_observable_flip))
        self.logical_z += int(jnp.sum(x_basis_circuit_observable_flip & ~z_basis_circuit_observable_flip))
        self.logical_y += int(jnp.sum(x_basis_circuit_observable_flip & z_basis_circuit_observable_flip))
