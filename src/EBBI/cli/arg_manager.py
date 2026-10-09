import argparse

def qec_args(parser: argparse.ArgumentParser, exclude: list[str] = None) -> None:
    """Add common quantum error correction related arguments to the given ArgumentParser.
    
    Args added:
    - qec
    - distance
    - rounds
    - noise_model
    - error_probability
    - error_bias
    """
    if exclude is None:
        exclude = []
    if 'qec' not in exclude:
        parser.add_argument('--qec', type=str, default='SurfaceCode', help='(default=SurfaceCode) The class of QEC code to use.')
    if 'distance' not in exclude:
        parser.add_argument('-d', '--distance', type=int, default=3, help='(default=3) The distance of the QEC code to use.')
    if 'rounds' not in exclude:
        parser.add_argument('-r', '--rounds', type=int, default=3, help='(default=3) The number of rounds of QEC.')
    if 'noise_model' not in exclude:
        parser.add_argument('--noise_model', type=str, default='HBD_noise', help='(default=HBD_noise) The noise model to use in the simulations. Options are: z_biased_SI1000_inspired_noise, z_biased_code_capacity_noise, SDEM3_noise, HBD_noise, MHBD_noise, SI1000_noise, depolarization_noise.')
    if 'error_probability' not in exclude:
        parser.add_argument('-p', '--error_probability', type=float, default=0.001, help='(default=0.001) The error probability to use in the noise model.')
    if 'error_bias' not in exclude:
        parser.add_argument('-eta', '--error_bias', type=float, default=500, help='(default=500) The bias to use in the noise model if applicable.')

def candidate_args(parser: argparse.ArgumentParser, exclude: list[str] = None) -> None:
    """Add arguments for when calling one of the candidate classes directly (e.g. pymatching_candidate, cudaqtensor_candidate).
    
    Args added:
    - clifford_deformations¨
    - shots
    """
    if exclude is None:
        exclude = []
    if 'clifford_deformations' not in exclude:
        parser.add_argument('-cds', '--clifford_deformations', type=str, nargs='+', required=True, help='[Required] The Clifford-deformations to test.')
    if 'shots' not in exclude:
        parser.add_argument('-s', '--shots', type=int, required=True, help='[Required] Shots per Clifford-deformation.')

def special_candidate_args(parser: argparse.ArgumentParser, include: list[str] = None) -> None:
    """Add arguments for specific features of the candidate classes (e.g. the correlation-aware MWPM decoder in PyMatching).
    
    NOTE: Leave `include = None` to include all.
    
    Args added:
    - use_corr_mwpm
    - bposd_max_bp_iters
    - bposd_bp_method
    - bposd_osd_order
    - bposd_osd_method
    - bm_max_bp_iters
    - bm_bp_method
    - tesseract_det_beam
    - tesseract_beam_climbing
    - tesseract_pqlimit
    - tesseract_det_penalty
    - datasheet_csv_path
    """
    if include is None or 'use_corr_mwpm' in include:
        parser.add_argument('--use_corr_mwpm', action='store_true', help="(default=False) PyMatching candidate setting: Whether to use the correlation-aware MWPM decoder in PyMatching (https://doi.org/10.48550/arXiv.1310.0863).")
    if include is None or 'bposd_max_bp_iters' in include:
        parser.add_argument('--bposd_max_bp_iters', type=int, default=30, help='(default=30) BPOSD candidate setting: The maximum number of belief-propagation iterations to run before handing over to OSD.')
    if include is None or 'bposd_bp_method' in include:
        parser.add_argument('--bposd_bp_method', type=str, default='product_sum', help='(default=product_sum) BPOSD candidate setting: The belief-propagation update rule. Options are: product_sum, min_sum, min_sum_log.')
    if include is None or 'bposd_osd_order' in include:
        parser.add_argument('--bposd_osd_order', type=int, default=60, help='(default=60) BPOSD candidate setting: The OSD order (the main accuracy/runtime dial). Only used when --bposd_osd_method is osd_e or osd_cs.')
    if include is None or 'bposd_osd_method' in include:
        parser.add_argument('--bposd_osd_method', type=str, default='osd_cs', help='(default=osd_cs) BPOSD candidate setting: The OSD method. Options are: osd_0 (zero-order), osd_e (exhaustive), osd_cs (combination-sweep).')
    if include is None or 'bm_max_bp_iters' in include:
        parser.add_argument('--bm_max_bp_iters', type=int, default=20, help='(default=20) BeliefMatching candidate setting: The maximum number of belief-propagation iterations to run before handing the reweighted graph over to MWPM (https://doi.org/10.48550/arXiv.2203.04948).')
    if include is None or 'bm_bp_method' in include:
        parser.add_argument('--bm_bp_method', type=str, default='product_sum', help='(default=product_sum) BeliefMatching candidate setting: The belief-propagation update rule. Options are: product_sum, minimum_sum, product_sum_log, minimum_sum_log.')
    if include is None or 'tesseract_det_beam' in include:
        parser.add_argument('--tesseract_det_beam', type=int, default=5, help='(default=5) Tesseract candidate setting: Beam cutoff giving the maximum number of detection events a search state may have (the main accuracy/runtime dial, larger is more accurate and slower). Use 65535 (tesseract_decoder.tesseract.INF_DET_BEAM) for no cutoff.')
    if include is None or 'tesseract_beam_climbing' in include:
        parser.add_argument('--tesseract_beam_climbing', action='store_true', help='(default=False) Tesseract candidate setting: Whether to enable the beam climbing heuristic.')
    if include is None or 'tesseract_pqlimit' in include:
        parser.add_argument('--tesseract_pqlimit', type=int, default=200_000, help='(default=200_000) Tesseract candidate setting: Per-shot search budget given as the maximum size of the priority queue. Shots that exhaust it are counted as logical errors.')
    if include is None or 'tesseract_det_penalty' in include:
        parser.add_argument('--tesseract_det_penalty', type=float, default=0.0, help='(default=0.0) Tesseract candidate setting: Extra cost added for each detector visited during the search.')
    if include is None or 'datasheet_csv_path' in include:
        parser.add_argument('--datasheet_csv_path', type=str, default=None, help='(default=None) [Required by the Datasheet candidate] Datasheet candidate setting: Path to a CSV file holding the pre-computed logical outcome counts (no-err, x-err, z-err, y-err) of every Clifford-deformation the method can reach. Each Clifford-deformation must appear exactly once, so pre-filter the file to a single noise model / p / eta / decoder.')

def method_args(parser: argparse.ArgumentParser, exclude: list[str] = None) -> None:
    """Add arguments for when calling one of the method scripts (e.g. ebbi.py, rs.py).
    
    Args added:
    - total_shots
    - experiments
    - active_deformations
    - candidate_class
    - method_verbose
    - shots_between_checkpoint
    """
    if exclude is None:
        exclude = []
    if 'total_shots' not in exclude:
        parser.add_argument('-S', '--total_shots', type=int, required=True, help='[Required] Total number of shots to use during the method execution.')
    if 'experiments' not in exclude:
        parser.add_argument('-E', '--experiments', type=int, required=True, help='[Required] Number of times to run the method (NOTE: Each experiment will use a different random seed based on the provided seed through the --seed argument).')
    if 'active_deformations' not in exclude:
        parser.add_argument('--active_deformations', type=int, nargs='+', default=[0,2,3], help='(default=[0,2,3]) The Clifford-deformations to explore during the method execution. Valid numbers [0: no-change, 1: swap X and Y, 2: swap Y and Z, 3: swap X and Z, 4: X -> Z -> Y -> X, 5: X -> Y -> Z -> X].')
    if 'candidate_class' not in exclude:
        parser.add_argument('--candidate_class', type=str, default='PyMatching', help='(default=PyMatching) The candidate class to use (PyMatching, BeliefMatching, BPOSD, Tesseract, CudaQTensor or Datasheet).')
    if 'method_verbose' not in exclude:
        parser.add_argument('--method_verbose', action='store_true', help='(default=false) Whether to print detailed information specific to the method execution.')
    if 'shots_between_checkpoint' not in exclude:
        parser.add_argument('-c', '--checkpoint_exponent_base', type=float, default=None, help='(default=None) If provided, the method will save its state to a checkpoint file at exponentially growing intervals (each checkpoint being 10^checkpoint_exponent_base larger). The checkpoint file will be saved in the save place as the data file but with _checkpoint.csv.')

def general_args(parser: argparse.ArgumentParser, exclude: list[str] = None) -> None:
    """General arguments that are almost always used.
    
    Args added:
    - save_path
    - max_shot_size
    - seed
    - seeds
    - force_override
    - verbose
    """
    if exclude is None:
        exclude = []
    if 'save_path' not in exclude:
        parser.add_argument('--save_path', type=str, required=True, help='[Required] The path to the file where the results will be saved (The file will be created during execution).')
    if 'max_shot_size' not in exclude:
        parser.add_argument('--max_shot_size', type=int, default=1_000_000, help='(default=1_000_000) The maximum number of shots to use in a single sampling step to prevent memory issues.')
    if 'seed' not in exclude:
        parser.add_argument('--seed', type=int, default=42, help='(default=42) The random seed to use for reproducible results (Used to generate the "--seeds" argument if not provided).')
    if 'seeds' not in exclude:
        parser.add_argument('--seeds', nargs='+', type=int, default=None, help='(default=None) The random seeds to use for each individual run (must be the same length as the number of Clifford-deformations).')
    if 'force_override' not in exclude:
        parser.add_argument('-f', '--force_override', action='store_true', help='(default=false) Whether to force override the results if the save path already exists. Use with caution as this will delete previous results with the same name.')
    if 'verbose' not in exclude:
        parser.add_argument('-v', '--verbose', action='store_true', help='(default=false) Whether to print detailed information during the method execution.')

def job_args(parser: argparse.ArgumentParser, exclude: list[str] = None, default_jobs_root: str = 'results') -> None:
    """Add arguments for when generating a SLURM job script (e.g. generate_slurm_job.py).
    
    Args added:
    - save_folder_name
    - jobs_root
    - partition
    - job_memory
    - time_limit_per_job
    - mail_type
    - mail_user
    - parallel_jobs
    - force_override
    - auto_run_slurm
    """
    if exclude is None:
        exclude = []
    if 'save_folder_name' not in exclude:
        parser.add_argument('--save_folder_name', type=str, required=True, help='[Required] The name of the folder where the generated SLURM job script, params.csv file, log files and results will be saved (The folder will be created during execution).')
    if 'jobs_root' not in exclude:
        parser.add_argument('--jobs_root', type=str, default=default_jobs_root, help=f'(default={default_jobs_root}) The directory (relative to the current working directory) that --save_folder_name is created inside.')
    if 'partition' not in exclude:
        parser.add_argument('--partition', type=str, default=None, help='(default=None) The partition to use for the SLURM job script. No --partition line is added when not set, so the cluster default partition is used.')
    if 'job_memory' not in exclude:
        parser.add_argument('--job_memory', type=str, default='5G', help='(default=5G) The memory to request for each SLURM job.')
    if 'time_limit_per_job' not in exclude:
        parser.add_argument('--time_limit_per_job', type=str, default='1-00:00:00', help='(default=1-00:00:00) The time limit to set for each SLURM job in the format HH:MM:SS.')
    if 'mail_type' not in exclude:
        parser.add_argument('--mail_type', type=str, default='ALL', help='(default=ALL) The mail type to set for the SLURM job script (e.g. BEGIN, END, FAIL, ALL).')
    if 'mail_user' not in exclude:
        parser.add_argument('--mail_user', type=str, default=None, help='(default=None) The email address to set for receiving notifications from the SLURM job script. No mail lines are added when not set.')
    if 'parallel_jobs' not in exclude:
        parser.add_argument('--parallel_jobs', type=int, default=25, help='(default=25) The number of parallel jobs to run in the SLURM array job (This will be set in the SLURM job script and can be adjusted based on the cluster resources and the expected runtime of each job).')
    if 'force_override' not in exclude:
        parser.add_argument('-f', '--force_override', action='store_true', help='(default=false) Whether to force override the results if the save path already exists. Use with caution as this will delete previous results with the same name.')
    if 'auto_run_slurm' not in exclude:
        parser.add_argument('--auto_run_slurm', action='store_true', help='(default=false) Whether to automatically run the generated SLURM job script after generation.')

def optional_sbatch_lines(args: argparse.Namespace) -> str:
    """Return the cluster-specific #SBATCH lines (partition and mail) for the options that were set.

    Each line is prefixed with a newline so the result can be appended directly after another #SBATCH line.
    """
    lines = []
    if getattr(args, 'partition', None):
        lines.append(f"#SBATCH --partition={args.partition}")
    if getattr(args, 'mail_user', None):
        lines.append(f"#SBATCH --mail-type={args.mail_type}")
        lines.append(f"#SBATCH --mail-user={args.mail_user}")
    return "".join(f"\n{line}" for line in lines)
