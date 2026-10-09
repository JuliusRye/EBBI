from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pandas as pd
from EBBI.candidate.base_class import Candidate
from EBBI.candidate.str_cd_convertion import cd_to_str


@lru_cache(maxsize=None)
def load_datasheet(csv_file_path: str) -> dict[str, np.ndarray]:
    """Parses a datasheet CSV into a `clifford_deformation -> outcome counts` lookup.

    The result is cached on the file path so that a method constructing thousands of
    `DatasheetCandidate`s only pays for parsing the CSV once.
    """
    df = pd.read_csv(csv_file_path, dtype={"clifford_deformation": str})

    duplicates = df["clifford_deformation"][df["clifford_deformation"].duplicated()].unique()
    assert len(duplicates) == 0, (
        f"Expected at most one row per Clifford deformation in {csv_file_path} but "
        f"{len(duplicates)} of them are repeated (e.g. {duplicates[0]}). Pre-filter the CSV file "
        "so that each Clifford deformation appears exactly once (e.g. one file per noise model / "
        "p / eta / decoder)."
    )

    # The datasheet outcome counts in the same order as the multinomial we sample from below.
    counts = df[["no-err", "x-err", "z-err", "y-err"]].to_numpy(dtype=np.float64)
    return dict(zip(df["clifford_deformation"], counts))


class DatasheetCandidate(Candidate):

    def __init__(
        self,
        clifford_deformation: jnp.ndarray,
        seed: int,
        csv_file_path: str,
        # Placeholder arguments to match the signature of the simulating candidates. They are ignored as the datasheet already contains all the necessary information.
        rounds: int = None,
        noise_type: str = None,
    ):
        # `rounds` and `noise_type` are accepted (and ignored) so that this candidate has the same
        # signature as the simulating candidates: the datasheet already fixes both of them.
        super().__init__(clifford_deformation, seed)

        # Assume that the CSV file has a large enough number of shots per CD that the calculated LER = the true LER.
        cd_str = cd_to_str(clifford_deformation)
        datasheet = load_datasheet(csv_file_path)
        assert cd_str in datasheet, (
            f"The Clifford deformation {cd_str} is not present in {csv_file_path}. The datasheet "
            "must cover every Clifford deformation the method can reach."
        )

        counts = datasheet[cd_str]
        assert counts.min() >= 0, f"The datasheet outcome counts must be non-negative but got {counts}."
        total = counts.sum()
        assert total > 0, f"The datasheet has no shots for the Clifford deformation {cd_str}."
        error_count = counts[1:].sum()
        if error_count < 10_000:
            print(
                f"Warning: The datasheet has only {error_count} error shots for the Clifford deformation "
                f"{cd_str}. This may be too few to accurately estimate the true LER."
            )
        # Treat the measured frequencies as the true logical outcome probabilities.
        self.probabilities = counts / total
        self.datasheet_shots = int(total)
        self.rng = np.random.default_rng(int(seed))

    def _sample(self, shots: int) -> None:
        if shots <= 0:
            return
        # Drawing the outcomes of all shots at once is equivalent to sampling them one by one from
        # the (assumed exact) logical outcome distribution of this Clifford deformation.
        logical_I, logical_x, logical_z, logical_y = self.rng.multinomial(shots, self.probabilities)
        self.logical_I += int(logical_I)
        self.logical_x += int(logical_x)
        self.logical_z += int(logical_z)
        self.logical_y += int(logical_y)
