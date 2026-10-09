"""Command-line entry point: generate a SLURM job script for finding the threshold."""

from itertools import product
import os
import posixpath
import shutil
import argparse
import json

from EBBI.cli import arg_manager

# Use numpy's random integer generator instead of jax's as numpy generates new random numbers
# each time this script is run, while jax's random number generator is deterministic.
# In this specific case, we want to generate different random seeds each time we run the script
# to ensure that for a new sweep we are not accidentally using the same seeds as before, which could lead to biased results.
# The results are still reproducible as the seeds are saved in the params.csv file and can be reused if needed.
from numpy.random import randint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate SLURM job script for finding the threshold.',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.job_args(parser, default_jobs_root='results/thresshold')
    parser.add_argument('--job_name', type=str, default='Noise_sweep', help='(default=Noise_sweep) The name of the SLURM job to be generated.')
    parser.add_argument('-p', '--error_probability', type=float, nargs='+', required=True, help='[Required] [List of] The error probability to use in the noise model.')
    parser.add_argument('-eta', '--error_bias', type=float, nargs='+', required=True, help='[Required] [List of] The bias to use in the noise model if applicable.')
    parser.add_argument('--repeats', type=int, default=1, help='(default=1) The number of times to repeat each simulation.')
    parser.add_argument('--candidate_type', type=str, default='PyMatching', help='(default=PyMatching) The candidate type to use in the generated SLURM job script (PyMatching or CudaQTensor).')
    parser.add_argument('--candidate_args', type=str, required=True, help='[Required] The candidate arguments to use for the SLURM job script (see <candidate_script>.py -h for more details). IMPORTANT: do not include any of the following arguments: --seed, --seeds, --save_path, -v/--verbose')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if any(arg in args.candidate_args for arg in ['--seed', '--seeds', '--save_path', '-v', '--verbose']):
        raise ValueError("The following arguments should not be included in --candidate_args as they are already handled in the script: --seed, --seeds, --save_path, -v/--verbose. Please remove them from the candidate_args to avoid errors.")

    candidate_parser = argparse.ArgumentParser(
        description='Valid args for --candidate_args=\"<args>\".',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.qec_args(candidate_parser, exclude=['error_probability', 'error_bias'])
    arg_manager.candidate_args(candidate_parser)
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
    with open(os.path.join(job_dir, 'params.csv'), 'w', newline='') as f:
        f.write("seed,p,eta\n")  # Header
        jobs = 0
        for p, eta in product(args.error_probability, args.error_bias):
            for _ in range(args.repeats):
                seed = randint(0, 2**31 - 1)
                f.write(f"{seed},{p},{eta}\n")
                jobs += 1
    print(f"Generated {jobs} combinations in params.csv")

    # Create SLURM job script
    # Paths embedded in the generated bash script always use forward slashes, so a run.sh
    # generated on Windows stays valid on the Linux cluster.
    job_dir_sh = job_dir.replace(os.sep, '/')
    match args.candidate_type:
        case 'PyMatching':
            candidate_script = 'ebbi-pymatching'
        case 'CudaQTensor':
            candidate_script = 'ebbi-cudaqtensor'
        case _:
            raise ValueError(f"Unknown candidate type: {args.candidate_type}, options=[PyMatching, CudaQTensor]")

    job_script = f"""#!/bin/bash
#SBATCH --job-name="{args.job_name}"{arg_manager.optional_sbatch_lines(args)}
#SBATCH --ntasks=1
#SBATCH --mem={args.job_memory}
#SBATCH -o {posixpath.join(job_dir_sh, 'logs', '%x_%a')}.out
#SBATCH -e {posixpath.join(job_dir_sh, 'logs', '%x_%a')}.err
#SBATCH --time={args.time_limit_per_job}
#SBATCH --array=1-{jobs}%{args.parallel_jobs}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {posixpath.join(job_dir_sh, 'params.csv')})

IFS=',' read -r seed p eta <<< "$LINE"

srun uv run --no-sync {candidate_script} --seed $seed -p $p -eta $eta --save_path {posixpath.join(job_dir_sh, 'results', 'sweep_${SLURM_ARRAY_TASK_ID}.csv')} {args.candidate_args} -v
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
