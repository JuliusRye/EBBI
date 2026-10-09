import jax.numpy as jnp
from jax import random
from abc import ABC, abstractmethod
from EBBI.candidate.base_class import Candidate
from EBBI.qecc.base_class import QECC

class Method(ABC):
    
    def __init__(
        self,
        seed: int,
        num_data_qubits: int,
        active_deformation: jnp.ndarray,
        candidate_class: type[Candidate],
        candidate_kwargs: dict,
        rounds: int,
        noise_type: str,
        verbose: bool = False
    ):
        """
        Args:
            num_data_qubits (int): The number of data qubits in the code.
            active_deformation (jnp.ndarray): An array of ints representing the Clifford-deformations that can be used in the search.
            candidate_kwargs (dict): A dictionary of keyword arguments to be passed to the Candidate class when creating new candidates excluding the `clifford_deformation` and `seed` arguments.
        """
        (
            self.clifford_deformation_key,
            self.decoder_seed_key,
        ) = random.split(random.key(seed), 2)
        self.seed = seed
        self.num_data_qubits = num_data_qubits
        self.active_deformation = active_deformation
        self.candidate_class = candidate_class
        self.candidate_kwargs = candidate_kwargs
        self.all_tested_candidates: list[Candidate] = [] # List to store all candidates that have been tested
        self.rounds = rounds
        self.noise_type = noise_type
        self.verbose = verbose
        
    
    def random_clifford_deformation(self) -> jnp.ndarray:
        """
        Randomly samples a Clifford-deformation from the `active_deformation` array.

        Returns:
            jnp.ndarray: A randomly sampled Clifford-deformation from the `active_deformation` array
        """
        self.clifford_deformation_key, cd_key = random.split(self.clifford_deformation_key, 2)
        rand_index = random.randint(
            cd_key,
            shape=(self.num_data_qubits,),
            minval=0,
            maxval=self.active_deformation.shape[0],
        )
        return self.active_deformation[rand_index]
    
    def create_candidate(self, clifford_deformation: jnp.ndarray) -> Candidate:
        """
        Construct a new candidate with the given Clifford deformation and a random seed.

        Args:
            clifford_deformation (jnp.ndarray): The Clifford deformation to be used for the new candidate.

        Returns:
            Candidate: A new candidate instance with the specified Clifford deformation and a random seed.
        """
        self.decoder_seed_key, seed_key = random.split(self.decoder_seed_key)
        return self.candidate_class(
            clifford_deformation=clifford_deformation,
            seed=int(random.randint(seed_key, shape=(), minval=-2**31, maxval=2**31-1)) + 2**31, # Random unsigned 32-bit integer
            **self.candidate_kwargs,
            rounds=self.rounds,
            noise_type=self.noise_type
        )

    def random_candidate(self) -> Candidate:
        """
        Generates a random candidate by sampling a random Clifford-Deformation from the active_deformation list.

        Returns:
            Candidate: A new candidate instance with a randomly selected Clifford deformation and a random seed.
        """
        clifford_deformation = self.random_clifford_deformation()
        new_candidate = self.create_candidate(clifford_deformation)
        return new_candidate

    @abstractmethod
    def run_method(self) -> Candidate:
        """
        Runs the method.
        """
        pass

