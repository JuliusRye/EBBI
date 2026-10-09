from abc import ABC, abstractmethod
import jax.numpy as jnp

class Candidate(ABC):

    def __init__(
        self, 
        clifford_deformation: jnp.ndarray, 
        seed: int,
        *args, **kwargs
    ):
        """Initializes the Candidate with a given Clifford deformation and seed.
        
        Args:
            clifford_deformation (jnp.ndarray): The Clifford deformation associated with this candidate.
            seed (int): The random seed for sampling.
            args: Additional arguments depending on the subclass.
            kwargs: Additional keyword arguments depending on the subclass.
        """
        self.clifford_deformation = clifford_deformation
        self.seed = seed

        self.times_chosen = 0 # Number of times this candidate has been chosen for sampling
        self.logical_I = 0
        self.logical_x = 0
        self.logical_z = 0
        self.logical_y = 0
    
    @property
    def total_shots(self) -> int:
        """Returns the total number of shots taken for this candidate."""
        return self.logical_I + self.logical_x + self.logical_z + self.logical_y

    @property
    def total_errors(self) -> int:
        """Returns the total number of logical errors observed for this candidate."""
        return self.logical_x + self.logical_z + self.logical_y
    
    @property
    def alpha(self) -> float:
        """Returns the alpha value (number of errors + 1) for this candidate."""
        return self.total_errors + 1
    
    @property
    def beta(self) -> float:
        """Returns the beta value (number of successes + 1) for this candidate."""
        return self.logical_I + 1
    
    @property
    def variance(self) -> float:
        """Returns the variance of the error rate estimate for this candidate."""
        alpha = self.alpha
        beta = self.beta
        # The variance of a beta distribution is given by (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        # It has been split up into three devisions as the denominator can get very large and cause overflow issues if calculated directly.
        return (alpha * beta) / (alpha + beta) / (alpha + beta) / (alpha + beta + 1)
    
    @property
    def uncertainty(self) -> float:
        """Returns the standard deviation (uncertainty) of the error rate estimate for this candidate."""
        return jnp.sqrt(self.variance)
    
    @property
    def expected_ler(self) -> float:
        """Returns the expected logical error rate (LER) for this candidate."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self, shots: int, max_shot_size: int) -> None:
        shots_remaining = shots
        # Sample in batches if shots exceed max_shot_size until the remaining shots are within the limit
        while shots_remaining > max_shot_size:
            self._sample(max_shot_size)
            shots_remaining -= max_shot_size
        # Sample the remaining shots
        self._sample(shots_remaining)
    
    def auto_sample(self, improvement_factor: float, max_shot_size: int, max_total_shots: int = 100_000_000, starting_shots: int = 100, verbose=False) -> int:
        """
        Returns:
            int: The number of shots taken in this auto-sampling step.
        """
        assert starting_shots > 0, "Starting shots must be greater than 0."
        # Calculate the number of shots needed to achieve the desired improvement in variance (assuming quadratic scaling)
        shots = int(self.total_shots * (improvement_factor**2 - 1))
        shots = max(
            shots,
            starting_shots # Minimum number of shots to ensure some progress is made initially
        )
        shots = min(shots, max_total_shots) # Cap the shots to the maximum allowed

        self.sample(shots, max_shot_size)
        return shots
    
    @abstractmethod
    def _sample(self, shots: int) -> None:
        """**Do not call this method directly!**. 
        
        Use sample(shots, max_shot_size) instead. This method does not protect against large shot sizes."""
        # Implementation of this method should perform the actual sampling and update the logical_I, logical_x, logical_z, and logical_y counts based on the results.
        pass