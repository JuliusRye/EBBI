import os
import math
import jax.numpy as jnp
from jax import random
from EBBI.methods.base_class import Method
from EBBI.candidate.base_class import Candidate
from EBBI.candidate.str_cd_convertion import cd_to_str

class RandomSearch(Method):

    def __init__(
        self,
        seed: int,
        num_data_qubits: int,
        active_deformation: jnp.ndarray,
        candidate_class: Candidate,
        candidate_kwargs: dict,
        max_shot_size: int,
        total_shots: int,
        num_candidates_to_test: int,
        rounds: int = 3,
        noise_type: str = 'circuit-level',
        checkpoint_exponent_base: int = None, # 1 will mean an order of magnitude before next checkpoint, 1/5 will mean 5 checkpoints per order of magnitude, etc.
        check_point_path: str = None,
        force_override: bool = False,
        verbose: bool = False
    ):
        # Initialize the base Method class
        super().__init__(seed, num_data_qubits, active_deformation, candidate_class, candidate_kwargs, rounds, noise_type, verbose)
        # Store parameters specific to RandomSearch
        self.num_candidates_to_test = num_candidates_to_test
        self.total_shots = total_shots
        self.shots_per_candidate = total_shots // num_candidates_to_test
        self.max_shot_size = max_shot_size

        # Create checkpoint file if checkpointing is enabled
        self.checkpoint_exponent_base = checkpoint_exponent_base
        self.check_point_path = check_point_path
        self.min_shots_for_next_checkpoint = self.shots_per_candidate
        self.checkpoint_anchor = max(self.shots_per_candidate, 1)
        self.last_checkpoint_shots = -1
        if self.check_point_path is None:
            if self.checkpoint_exponent_base is not None:
                raise ValueError("Checkpoint exponent base provided without a checkpoint path. Please provide a checkpoint path to enable checkpointing.")
        else:
            if self.checkpoint_exponent_base is None:
                raise ValueError("Checkpoint path provided without a checkpoint exponent base. Please provide a checkpoint exponent base to enable checkpointing.")
            if os.path.exists(self.check_point_path) and not force_override:
                raise ValueError(f"Checkpoint file {self.check_point_path} already exists. Please provide a different path or remove the existing file.")
            with open(self.check_point_path, 'w') as f:
                f.write("clifford_deformation,exp_ler,uncertainty,method_shots\n") # Write header for checkpoint file

    def run_method(self) -> Candidate:
        """Executes the random search method by sampling each candidate for a fixed number of shots."""
        lowest_observed_ler = 1.0 # 100% error rate (theoretical worst possible case)
        already_sampled_clifford_deformations = set() # To keep track of already sampled candidates and avoid duplicates
        self.shots_taken = 0
        for i in range(self.num_candidates_to_test):
            if self.verbose:
                print(f"Sampling candidate {i+1}/{self.num_candidates_to_test}", flush=True)
            # We assume that the sampling space is large enough that we won't sample the same candidate twice.
            for j in range(100): # Arbitrary large number to prevent infinite loops
                candidate = self.random_candidate()
                if cd_to_str(candidate.clifford_deformation) not in already_sampled_clifford_deformations:
                    already_sampled_clifford_deformations.add(cd_to_str(candidate.clifford_deformation))
                    break
                if j == 99:
                    raise ValueError("Could not find a new candidate after 100 attempts. Consider increasing the number of candidates or the sampling space.")
            candidate.sample(self.shots_per_candidate, self.max_shot_size)
            observed_ler = candidate.alpha / (candidate.alpha + candidate.beta)
            if observed_ler < lowest_observed_ler:
                lowest_observed_ler = observed_ler
                self.leader_candidate = candidate

            self.shots_taken += candidate.total_shots
            if self.shots_taken >= self.min_shots_for_next_checkpoint:
                self.make_checkpoint()
                self.min_shots_for_next_checkpoint = self.next_checkpoint_threshold()
                if self.verbose:
                    print(f"Checkpoint created at step {i+1} after taking {self.shots_taken:_d} shots.")
        self.make_checkpoint()
        return self.leader_candidate

    def next_checkpoint_threshold(self) -> float:
        """Returns the shot count at which the next checkpoint should be taken."""
        if self.checkpoint_exponent_base is None:
            return float('inf')
        steps_taken = math.floor(math.log10(max(self.shots_taken, 1) / self.checkpoint_anchor) / self.checkpoint_exponent_base + 1e-9)
        return self.checkpoint_anchor * 10**((steps_taken + 1) * self.checkpoint_exponent_base)

    def make_checkpoint(self):
        """Saves the current state of the method to a checkpoint file."""
        if self.check_point_path is None or self.shots_taken == self.last_checkpoint_shots:
            return
        lead_candidate = self.leader_candidate
        with open(self.check_point_path, 'a') as f:
            f.write(f"{cd_to_str(lead_candidate.clifford_deformation)},{lead_candidate.expected_ler},{lead_candidate.uncertainty},{self.shots_taken}\n")     
            f.flush() # Ensure that the checkpoint is written to the file immediately
        self.last_checkpoint_shots = self.shots_taken
