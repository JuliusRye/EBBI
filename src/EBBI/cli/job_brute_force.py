"""Command-line entry point: generate a SLURM job script for running the brute force method."""

import os
import posixpath
import shutil
import argparse
import json

from EBBI.cli import arg_manager
from EBBI.candidate.str_cd_convertion import all_str

# Use numpy's random integer generator instead of jax's as numpy generates new random numbers
# each time this script is run, while jax's random number generator is deterministic.
# In this specific case, we want to generate different random seeds each time we run the script
# to ensure that for a new sweep we are not accidentally using the same seeds as before, which could lead to biased results.
# The results are still reproducible as the seeds are saved in the params.csv file and can be reused if needed.
from numpy.random import randint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate SLURM job script for running the brute force method.',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.job_args(parser, default_jobs_root='results/brute_force')
    parser.add_argument('--job_name', type=str, default='BruteForce_sweep', help='(default=BruteForce_sweep) The name of the SLURM job to be generated.')
    parser.add_argument('--cds_per_job', type=int, default=3**4, help='(default=81) The number of Clifford-deformations to test in each individual SLURM job.')
    parser.add_argument('--active_deformations', type=int, nargs='+', default=[0,2,3], help='(default=[0,2,3]) The Clifford-deformations to explore during the method execution. Valid numbers [0: no-change, 1: swap X and Y, 2: swap Y and Z, 3: swap X and Z, 4: X -> Z -> Y -> X, 5: X -> Y -> Z -> X].')
    parser.add_argument('--from_file', type=str, default=None, help='(default=None) If provided, the script will read the list of Clifford-deformations to test from this file instead of generating them based on the --active_deformations argument. The file should contain one Clifford-deformation per line.')
    parser.add_argument('--candidate_type', type=str, default='PyMatching', help='(default=PyMatching) The candidate type to use in the generated SLURM job script (PyMatching, BeliefMatching, BPOSD, Tesseract or CudaQTensor).')
    parser.add_argument('--gpus_per_job', type=int, default=0, help='(default=0) The number of GPUs to request for each SLURM job (only relevant on a GPU partition, e.g. when --candidate_type is CudaQTensor). No --gres line is added to the SLURM job script when set to 0.')
    parser.add_argument('--candidate_args', type=str, required=True, help='[Required] The candidate arguments to use for the SLURM job script (see <candidate_script>.py -h for more details). IMPORTANT: do not include any of the following arguments: -cds/--clifford_deformations, --seed, --seeds, --save_path, -v/--verbose')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if any(arg in args.candidate_args for arg in ['-cds', '--clifford_deformations', '--seed', '--seeds', '--save_path', '-v', '--verbose']):
        raise ValueError("The following arguments should not be included in --candidate_args as they are already handled in the script: -cds/--clifford_deformations, --seed, --seeds, --save_path, -v/--verbose. Please remove them from the candidate_args to avoid errors.")

    candidate_parser = argparse.ArgumentParser(
        description='Valid args for --candidate_args=\"<args>\".',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.qec_args(candidate_parser)
    arg_manager.candidate_args(candidate_parser, exclude=['clifford_deformations'])
    arg_manager.general_args(candidate_parser, exclude=['save_path', 'seed', 'seeds', 'verbose'])
    arg_manager.special_candidate_args(candidate_parser)
    candidate_args = candidate_parser.parse_args(args.candidate_args.split(" "))

    if candidate_args.use_corr_mwpm:
        assert args.candidate_type == 'PyMatching', "The argument --use_corr_mwpm is only applicable when --candidate_type is set to PyMatching. Please set it accordingly."

    job_dir = os.path.join(args.jobs_root, args.save_folder_name)
    if os.path.exists(job_dir):
        if args.force_override:
            shutil.rmtree(job_dir)
        else:
            raise ValueError(f"Directory {job_dir} already exists. Please choose another name or delete the existing directory to avoid overwriting previous results.")
    os.makedirs(job_dir)
    os.mkdir(os.path.join(job_dir, 'logs'))
    os.mkdir(os.path.join(job_dir, 'results'))

    # Log script arguments
    save_args = vars(args).copy()
    save_args['candidate_args'] = vars(candidate_args).copy()
    with open(os.path.join(job_dir, 'config.json'), 'w') as f:
        json.dump(save_args, f, indent=4)

    # Generate params.csv file
    match candidate_args.qec:
        case 'SurfaceCode':
            num_of_data_qubits = candidate_args.distance**2
        case _:
            raise ValueError(f"Unknown quantum error-correcting code: {candidate_args.qec}, options=[SurfaceCode]")
    if args.from_file is None:
        cds_all = all_str(num_of_data_qubits, args.active_deformations)
        number_of_clifford_deformations = len(args.active_deformations)**num_of_data_qubits
    else:
        with open(args.from_file, 'r') as f:
            cds_all = [line.strip() for line in f.readlines()]
        number_of_clifford_deformations = len(cds_all)
    with open(os.path.join(job_dir, 'params.csv'), 'w', newline='') as f:
        f.write("seed,cds\n")  # Header
        jobs = 0
        for start in range(0, number_of_clifford_deformations, args.cds_per_job):
            end = min(start + args.cds_per_job, number_of_clifford_deformations)
            seed = randint(0, 2**31 - 1)
            cds = ' '.join(cds_all[start:end])
            f.write(f"{seed},{cds}\n")
            jobs += 1
    print(f"Generated {jobs} combinations in params.csv")

    # Create SLURM job script
    # Paths embedded in the generated bash script always use forward slashes, so a run.sh
    # generated on Windows stays valid on the Linux cluster.
    job_dir_sh = job_dir.replace(os.sep, '/')
    match args.candidate_type:
        case 'PyMatching':
            candidate_script = 'ebbi-pymatching'
        case 'BeliefMatching':
            candidate_script = 'ebbi-beliefmatching'
        case 'BPOSD':
            candidate_script = 'ebbi-bposd'
        case 'Tesseract':
            candidate_script = 'ebbi-tesseract'
        case 'CudaQTensor':
            candidate_script = 'ebbi-cudaqtensor'
        case _:
            raise ValueError(f"Unknown candidate type: {args.candidate_type}, options=[PyMatching, BeliefMatching, BPOSD, Tesseract, CudaQTensor]")

    # Only requested when asked for, so that the CPU only decoders are not queued behind the GPU nodes
    gres_line = f"\n#SBATCH --gres=gpu:{args.gpus_per_job}" if args.gpus_per_job > 0 else ""

    job_script = f"""#!/bin/bash
#SBATCH --job-name="{args.job_name}"{arg_manager.optional_sbatch_lines(args)}{gres_line}
#SBATCH --ntasks=1
#SBATCH --mem={args.job_memory}
#SBATCH -o {posixpath.join(job_dir_sh, 'logs', '%x_%a')}.out
#SBATCH -e {posixpath.join(job_dir_sh, 'logs', '%x_%a')}.err
#SBATCH --time={args.time_limit_per_job}
#SBATCH --array=1-{jobs}%{jobs if args.parallel_jobs == -1 or args.parallel_jobs > jobs else args.parallel_jobs}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {posixpath.join(job_dir_sh, 'params.csv')})

IFS=',' read -r seed cds <<< "$LINE"

srun uv run --no-sync {candidate_script} --seed $seed -cds $cds --save_path {posixpath.join(job_dir_sh, 'results', 'sweep_${SLURM_ARRAY_TASK_ID}.csv')} {args.candidate_args} -v
"""

    with open(os.path.join(job_dir, 'run.sh'), 'w', newline='\n') as f:
        f.write(job_script)
    print(f"Generated SLURM job script.")

    # Optionally run the SLURM job script
    if args.auto_run_slurm:
        os.system(f"sbatch {os.path.join(job_dir, 'run.sh')}")
        print(f"Automatically submitted SLURM job script: {os.path.join(job_dir, 'run.sh')} to the cluster.")


if __name__ == "__main__":
    main()
