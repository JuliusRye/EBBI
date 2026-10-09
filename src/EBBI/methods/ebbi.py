import os
import math
import jax.numpy as jnp
from jax import random
from scipy.stats import beta as beta_dist
from EBBI.methods.base_class import Method
from EBBI.candidate.base_class import Candidate
from EBBI.candidate.str_cd_convertion import cd_to_str

class EvolutionBasedBestArmIdentification(Method):

    def __init__(
        self,
        seed: int,
        num_data_qubits: int,
        active_deformation: jnp.ndarray,
        candidate_class: Candidate,
        candidate_kwargs: dict,
        max_shot_size: int,
        total_shots: int,
        num_simultaneous_candidates: int,
        num_initial_candidates: int,
        improvement_factor: float,
        mutation_probability: float,
        prune_threshold: float = 0.05,
        candidate_min_errors_for_warmup: int = 500,
        rounds: int = 3,
        noise_type: str = 'circuit-level',
        checkpoint_exponent_base: int = None, # 1 will mean an order of magnitude before next checkpoint, 1/5 will mean 5 checkpoints per order of magnitude, etc.
        check_point_path: str = None,
        force_override: bool = False,
        verbose: bool = False
    ):
        """
        Evolution-Based Best-arm Identification (EBBI) method for finding the best Clifford deformation for quantum error correction.
        
        Args:
            seed (int): Random seed for reproducibility.
            num_data_qubits (int): Number of data qubits in the QEC code.
            active_deformation (jnp.ndarray): The Clifford deformation that can be used on the data qubits.
            candidate_class (Candidate): The class of candidates to use for evaluating the Clifford deformations.
            candidate_kwargs (dict): Additional keyword arguments to pass to the candidate class when creating new candidates.
            max_shot_size (int): The maximum number of shots to use in parallel (Larger is generally better but may run into memory issues if set too high).
            total_shots (int): The total number of shots to use for the entire method (the method will stop once this number of shots has been taken, even if it is in the middle of a round).
            num_simultaneous_candidates (int): The number of candidates to keep active at the same time (after initialization new candidates will only be added once the number of active candidates falls below this threshold).
            num_initial_candidates (int): The number of candidates to initialize at the start of the method (these candidates will be selected uniformly at random from the active deformation and warmed up before the method starts pruning and mutating candidates).
            improvement_factor (float): The factor by which the variance of the expected logical error rate of a candidate should be improved each time it is selected for more sampling (Larger values will lead to more shots being used each time a candidate is selected, which can lead to faster convergence but may also lead to more wasted shots on bad candidates if set too high).
            mutation_probability (float): The probability of changing the Clifford deformation on each data qubit when mutating a candidate (Larger values will lead to more exploration but may also lead to more wasted shots on bad candidates if set too high).
            prune_threshold (float): The threshold for pruning candidates based on their posterior distributions (Candidates whose confidence intervals for their expected logical error rates do not overlap with the confidence interval of the candidate with the lowest expected logical error rate will be pruned. Smaller values will lead to more aggressive pruning, which can lead to faster convergence but may also lead to good candidates being pruned if set too low).
            candidate_min_errors_for_warmup (int): The minimum number of errors a candidate must observe during warmup to ensure that its posterior distribution is localized enough to avoid immediately being pruned (Larger values will lead to more shots being used for warmup, which can help ensure that good candidates are not pruned early on but may also lead to more wasted shots on bad candidates if set too high).
            rounds (int): The number of rounds to use for the method (the method will run for the specified number of rounds, but may stop early if the total number of shots is reached. More rounds will generally lead to better results but will also take more time, so it is important to find a good balance based on the total number of shots and the expected convergence rate of the method).
            noise_type (str): The type of noise to use for the candidates (this is passed to the candidate class and can be used to determine how the candidates evaluate the Clifford deformations, e.g. by using different decoding strategies for different noise types).
            checkpoint_exponent_base (int): The base for the exponent that determines when to take checkpoints (e.g. if set to 1, a checkpoint will be taken every time the number of shots taken increases by an order of magnitude, if set to 1/5, a checkpoint will be taken every time the number of shots taken increases by a factor of 10^(1/5) ≈ 1.58, etc. If set to None, checkpointing will be disabled).
            check_point_path (str): The path to save the checkpoint file (if checkpointing is enabled). The checkpoint file will be a CSV file that logs the current best candidate and its expected logical error rate at each checkpoint.
            force_override (bool): Whether to force override the checkpoint file if it already exists (if checkpointing is enabled and the checkpoint file already exists, a ValueError will be raised unless this flag is set to True, in which case the existing file will be overwritten).
            verbose (bool): Whether to print verbose output during the method (setting this to True will print detailed information about the progress of the method, including which candidates are being sampled, pruned, and mutated, as well as the current best candidate and its expected logical error rate at each step).
        """
        # Initialize the base Method class
        super().__init__(seed, num_data_qubits, active_deformation, candidate_class, candidate_kwargs, rounds, noise_type, verbose)
        # Store parameters specific to EvolutionBasedBestArmIdentification
        self.mutation_key = random.key(seed+1)
        self.max_shot_size = max_shot_size
        self.total_shots = total_shots
        self.num_simultaneous_candidates = num_simultaneous_candidates
        self.improvement_factor = improvement_factor
        self.mutation_probability = mutation_probability
        self.prune_threshold = prune_threshold
        self.candidate_min_errors_for_warmup = candidate_min_errors_for_warmup
        # Keep track of how many shots have been taken to not exceed the max shot size
        self.shots_taken = 0
        self.terminate_early = False
        # Create checkpoint file if checkpointing is enabled
        self.checkpoint_exponent_base = checkpoint_exponent_base
        self.check_point_path = check_point_path
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
        # Initialize the active candidates by selecting them uniformly at random and warming them up
        self.active_candidates = [
            self.random_candidate() for _ in range(num_initial_candidates)
        ]
        self.seen_clifford_deformations = set(candidate.clifford_deformation.tobytes() for candidate in self.active_candidates) # Keep track of seen Clifford deformations to avoid duplicates in the initial candidates
        for i in range(num_simultaneous_candidates):
            self.warmup_candidate(candidate_index=i)
        # Take the first checkpoint right after the warmup of the initial candidates
        self.checkpoint_anchor = max(self.shots_taken, 1)
        self.make_checkpoint()
        self.min_shots_for_next_checkpoint = self.next_checkpoint_threshold()
    
    def explore_highest_variance_candidate(self):
        """Sample the arm/candidate with the highest variance to improve our estimate of its error rate."""
        variances = jnp.array([candidate.variance for candidate in self.active_candidates])
        candidate_index = jnp.argmax(variances)
        candidate = self.active_candidates[candidate_index]
        shots_used = candidate.auto_sample(
            self.improvement_factor,
            max_shot_size=self.max_shot_size,
            max_total_shots=self.total_shots - self.shots_taken
        )
        self.shots_taken += shots_used
        if self.verbose:
            print(f"Sampled candidate {cd_to_str(candidate.clifford_deformation)} (index {candidate_index}) using {shots_used}")
    
    def replace_with_mutation(self, candidate_index: int) -> bool:
        """
        Replaces the candidate at the given index with a mutated version of one of the best candidates, 
        where the best candidate is selected based on Thompson sampling from the posterior distributions 
        of the active candidates. The mutation is performed by randomly changing the Clifford deformation 
        on a subset of the data qubits, where the subset is determined by the mutation probability. 
        The mutated candidate is then added to the active candidates if it has not been seen before, 
        and the process is repeated until a new candidate is successfully added.
        
        Returns:
            bool: True if a new candidate was successfully added, False if the mutation process was stopped due to exceeding the maximum mutation rate (which can happen if the method has exhausted most mutations in the vicinity).
        """
        # Identify the best candidate using Thompson sampling from the posterior distributions of the active candidates
        alpha = jnp.array([candidate.alpha for candidate in self.active_candidates])
        beta = jnp.array([candidate.beta for candidate in self.active_candidates])
        parent_key, self.mutation_key = random.split(self.mutation_key)
        parent_index = jnp.argmin(
            random.beta(parent_key, alpha, beta)
        )
        parent_cd = self.active_candidates[parent_index].clifford_deformation
        # Randomly select which data qubits to mutate based on the mutation probabilit
        mutation_rate = self.mutation_probability
        while True:
            # Randomly select a Clifford deformation from the active deformation list for each data qubit
            random_cd = self.random_clifford_deformation()
            # Determine which data qubits to mutate based on the mutation probability
            index_key, self.mutation_key = random.split(self.mutation_key)
            mutation_indexes = random.uniform(index_key, shape=(self.num_data_qubits,)) < mutation_rate
            # Create a new candidate with the mutated Clifford deformation
            child_cd = jnp.where(mutation_indexes, random_cd, parent_cd)
            # Ensure that the mutated candidate has not been seen before to avoid duplicates
            if child_cd.tobytes() not in self.seen_clifford_deformations:
            # if not any(jnp.array_equal(child_cd, candidate.clifford_deformation) for candidate in self.active_candidates): # Allow using previusly sampled Clifford-deformations
                self.seen_clifford_deformations.add(child_cd.tobytes())
                self.active_candidates[candidate_index] = self.create_candidate(child_cd)
                if self.verbose:
                    print(f"Mutated candidate {cd_to_str(child_cd)} (index {candidate_index}) with parent {cd_to_str(parent_cd)} (index {parent_index}) using mutation rate {mutation_rate:.4%}.")
                self.warmup_candidate(candidate_index)
                return True
            # If the mutated candidate has been seen before, increase the mutation rate by 1% and try again
            mutation_rate *= 1.01
            if mutation_rate > 1: # If the mutation rate becomes too high, it means that we have likely exhausted all possible mutations and should stop to avoid an infinite loop
                print("Mutation rate exceeded 1, likely due to exhausting all possible mutations. No point in continuing to mutate, stopping to avoid infinite loop.")
                return False

    def prune(self) -> None:
        """
        Prunes candidates whose confidence intervals for their expected logical error rates do not overlap with the confidence interval of the candidate with the lowest expected logical error rate.
        """
        # Define the posterior distributions
        alpha = jnp.array([candidate.alpha for candidate in self.active_candidates])
        beta = jnp.array([candidate.beta for candidate in self.active_candidates])
        posteriors = beta_dist(alpha, beta)
        # Find the lower and upper bounds based on the confidence interval
        lower_bounds = posteriors.ppf(self.prune_threshold / 2)
        upper_bounds = posteriors.ppf(1 - self.prune_threshold / 2)
        # Define the cutoff as the lowest upper bound
        cutoff = upper_bounds.min()
        # Prune candidates whose lower bound is above the cutoff and replace them with mutations of the current best candidate
        failed = jnp.where(lower_bounds > cutoff)[0]
        # Remove the failed candidates and replace them with mutations of the current best candidates
        for candidate_index in reversed(failed.tolist()):
            if self.verbose:
                print(f"Pruning candidate {cd_to_str(self.active_candidates[candidate_index].clifford_deformation)} (index {candidate_index}).")
            if len(self.active_candidates) > self.num_simultaneous_candidates:
                self.active_candidates.pop(candidate_index)
            else:
                success = self.replace_with_mutation(candidate_index)
                if not success:
                    self.active_candidates.pop(candidate_index)
            if len(self.active_candidates) == 1:
                print("Terminating method early since only one candidate remains after pruning.")
                self.terminate_early = True
                return None

    def warmup_candidate(self, candidate_index: int):
        """Warmup the selected candidate to ensure that the posterior distribution is localized and not pruned before it has a chance to be properly evaluated."""
        # Keep sampling the candidate until it has observed a minimum number of errors.
        candidate_lers = jnp.array([
            candidate.expected_ler
            for candidate in self.active_candidates
            if candidate.total_shots > 0
        ])
        average_ler = jnp.mean(candidate_lers) if candidate_lers.size > 0 else 1e-2
        candidate = self.active_candidates[candidate_index]
        while candidate.total_errors < self.candidate_min_errors_for_warmup:
            shots_used = candidate.auto_sample(
                self.improvement_factor, 
                max_shot_size=self.max_shot_size,
                max_total_shots=self.total_shots - self.shots_taken,
                starting_shots=max(100, int(self.candidate_min_errors_for_warmup/average_ler)),
                verbose=self.verbose
            )
            self.shots_taken += shots_used
            if self.shots_taken >= self.total_shots:
                break
        if self.verbose:
            print(f"Warmed up candidate {cd_to_str(candidate.clifford_deformation)} (index {candidate_index}) using {candidate.total_shots}")

    def leader(self) -> Candidate:
        """Returns the current leading candidate based on the lowest expected logical error rate."""
        score = jnp.array([candidate.expected_ler + candidate.uncertainty for candidate in self.active_candidates])
        leader_index = jnp.argmin(score)
        return self.active_candidates[leader_index]

    def run_method(self) -> Candidate:
        step = 0
        while self.shots_taken < self.total_shots and not self.terminate_early:
            self.explore_highest_variance_candidate()
            self.prune()
            step += 1
            if self.verbose:
                lead_candidate = self.leader()
                print(f"Completed step {step}, total shots taken: {self.shots_taken}/{self.total_shots} ({self.shots_taken/self.total_shots:.2%}) - Current lead: {cd_to_str(lead_candidate.clifford_deformation)} with expected LER {lead_candidate.expected_ler:.4%} ± {lead_candidate.uncertainty:.4%}\n\tActive candidates: {[cd_to_str(candidate.clifford_deformation) for candidate in self.active_candidates]}")
            if self.shots_taken >= self.min_shots_for_next_checkpoint:
                self.make_checkpoint()
                self.min_shots_for_next_checkpoint = self.next_checkpoint_threshold()
                if self.verbose:
                    print(f"Checkpoint created at step {step} after taking {self.shots_taken:_d} shots.")
        self.make_checkpoint()
        return self.leader()

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
        lead_candidate = self.leader()
        with open(self.check_point_path, 'a') as f:
            f.write(f"{cd_to_str(lead_candidate.clifford_deformation)},{lead_candidate.expected_ler},{lead_candidate.uncertainty},{self.shots_taken}\n")     
            f.flush() # Ensure that the checkpoint is written to the file immediately
        self.last_checkpoint_shots = self.shots_taken
