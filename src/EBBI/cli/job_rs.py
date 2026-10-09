"""Command-line entry point: generate a SLURM job script for running the RS method."""

import os
import posixpath
import shutil
import argparse
import json
from itertools import product

from EBBI.cli import arg_manager

# Use numpy's random integer generator instead of jax's as numpy generates new random numbers
# each time this script is run, while jax's random number generator is deterministic.
# In this specific case, we want to generate different random seeds each time we run the script
# to ensure that for a new sweep we are not accidentally using the same seeds as before, which could lead to biased results.
# The results are still reproducible as the seeds are saved in the params.csv file and can be reused if needed.
from numpy.random import randint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate SLURM job script for running the RS method.',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.job_args(parser, default_jobs_root='results/rs')
    parser.add_argument('--job_name', type=str, default='RS_sweep', help='(default=RS_sweep) The name of the SLURM job to be generated.')
    parser.add_argument('-N', '--candidates', type=int, nargs='+', required=False, default=None, help='[Required or -spc] [List of] Number of simultaneous candidates for RS (-spc = -S / -N).')
    parser.add_argument('-spc', '--shots_per_candidate', type=int, nargs='+', required=False, default=None, help='[Required or -N] [List of] Number of shots per candidate (-N = -S / -spc)')
    parser.add_argument('-S', '--total_shots', type=int, nargs='+', required=True, help='[Required] [List of] Total number of shots to use during the method execution.')
    parser.add_argument('-R', '--repeat_same_settings', type=int, default=1, help='(default=1) Number of times to repeat the same settings with different random seeds. This can be used as an alternative / addition to the -N/--experiments argument in the method specific settings (-R will run multiple jobs in parallel while -N will run a single job with multiple experiments sequentially).')
    parser.add_argument('--rs_args', type=str, required=True, help='[Required] The RS method arguments to use for the SLURM job script (see rs.py -h for more details). IMPORTANT: do not include any of the following arguments: -N/--candidates, -S/--total_shots, --save_path, -v/--verbose')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.candidates is not None and args.shots_per_candidate is not None:
        raise ValueError("Both -N/--candidates and -spc/--shots_per_candidates where provided. Please only include one of them")
    if args.candidates is None and args.shots_per_candidate is None:
        raise ValueError("Neither -N/--candidates or -spc/--shots_per_candidates where provided. Please include one of them (and only one).")

    if any(arg in args.rs_args for arg in ['-N', '--candidates', '-S', '--total_shots', '--save_path', '-v', '--verbose']):
        raise ValueError("The following arguments should not be included in --rs_args as they are already handled in the script: -N/--candidates, -S/--total_shots, --save_path, -v/--verbose. Please remove them from the rs_args to avoid errors.")

    rs_parser = argparse.ArgumentParser(
        description='Valid args for --rs_args=\"<args>\".',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.qec_args(rs_parser)
    arg_manager.general_args(rs_parser, exclude=['save_path', 'verbose'])
    arg_manager.method_args(rs_parser, exclude=['total_shots'])
    arg_manager.special_candidate_args(rs_parser)
    rs_args = rs_parser.parse_args(args.rs_args.split(" "))

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
    save_args['rs_args'] = vars(rs_args).copy()
    with open(os.path.join(job_dir, 'config.json'), 'w') as f:
        json.dump(save_args, f, indent=4)

    # Generate params.csv file
    with open(os.path.join(job_dir, 'params.csv'), 'w', newline='') as f:
        f.write("seed,N,S\n")  # Header
        jobs = 0
        if args.candidates is not None:
            for N, S in product(args.candidates, args.total_shots):
                for _ in range(args.repeat_same_settings):
                    seed = randint(0, 2**31 - 1)
                    f.write(f"{seed},{N},{S}\n")
                    jobs += 1
        else:
            for SPC, S in product(args.shots_per_candidate, args.total_shots):
                for _ in range(args.repeat_same_settings):
                    N = S // SPC
                    seed = randint(0, 2**31 - 1)
                    f.write(f"{seed},{N},{S}\n")
                    jobs += 1
    print(f"Generated {jobs} combinations in params.csv")

    # Create SLURM job script
    # Paths embedded in the generated bash script always use forward slashes, so a run.sh
    # generated on Windows stays valid on the Linux cluster.
    job_dir_sh = job_dir.replace(os.sep, '/')
    job_script = f"""#!/bin/bash
#SBATCH --job-name="{args.job_name}"{arg_manager.optional_sbatch_lines(args)}
#SBATCH --ntasks=1
#SBATCH --mem={args.job_memory}
#SBATCH -o {posixpath.join(job_dir_sh, 'logs', '%x_%a')}.out
#SBATCH -e {posixpath.join(job_dir_sh, 'logs', '%x_%a')}.err
#SBATCH --time={args.time_limit_per_job}
#SBATCH --array=1-{jobs}%{args.parallel_jobs}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {posixpath.join(job_dir_sh, 'params.csv')})

IFS=',' read -r seed N S <<< "$LINE"

srun uv run --no-sync ebbi-rs --seed $seed -N $N -S $S --save_path {posixpath.join(job_dir_sh, 'results', 'sweep_${SLURM_ARRAY_TASK_ID}.csv')} {args.rs_args} -v
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
