"""Command-line entry point: generate a SLURM job script for running the EBBI method."""

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
        description='Generate SLURM job script for running the EBBI method.',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.job_args(parser, default_jobs_root='results/ebbi')
    parser.add_argument('--job_name', type=str, default='EBBI_sweep', help='(default=EBBI_sweep) The name of the SLURM job to be generated.')
    parser.add_argument('-N', '--candidates', type=int, nargs='+', required=True, help='[Required] [List of] Number of simultaneous candidates for EBBI.')
    parser.add_argument('-Ni', '--candidates_initially', type=int, nargs='+', default=[None], help='(default=N) [List of] Number of initial candidates for EBBI.')
    parser.add_argument('-M', '--mutation_probability', type=float, nargs='+', default=[None], help='(default=0.43/n) [List of] Probability of changing Clifford-deformation on a data qubit.')
    parser.add_argument('-P', '--prune_threshold', type=float, nargs='+', default=[0.05], help='(default=0.05) [List of] Threshold for pruning candidates based on their posterior distributions (lower values = more certain before pruning).')
    parser.add_argument('-i', '--improvement_factor', type=float, nargs='+', default=[1.5], help='(default=1.5) [List of] How much the variance of the LER of a candidate should be improved each time it is selected for more sampling.')
    parser.add_argument('-w', '--warmup', type=int, nargs='+', default=[50], help='(default=50) [List of] The minimum number of errors a candidate must observe during warmup to ensure that its posterior distribution is localized enough to avoid immediately being pruned.')
    parser.add_argument('-S', '--total_shots', type=int, nargs='+', required=True, help='[Required] [List of] Total number of shots to use during the method execution.')
    parser.add_argument('-R', '--repeat_same_settings', type=int, default=1, help='(default=1) Number of times to repeat the same settings with different random seeds. This can be used as an alternative / addition to the -N/--experiments argument in the method specific settings (-R will run multiple jobs in parallel while -N will run a single job with multiple experiments sequentially).')
    parser.add_argument('--ebbi_args', type=str, required=True, help='[Required] The EBBI method arguments to use for the SLURM job script (see ebbi.py -h for more details). IMPORTANT: do not include any of the following arguments: -N/--candidates, -Ni/--candidates_initially, -M/--mutation_probability, -P/--prune_threshold, -i/--improvement_factor, -w/--warmup, -S/--total_shots, --save_path, -v/--verbose')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print(args.ebbi_args)
    if any(arg in args.ebbi_args for arg in ['-N', '--candidates', '-Ni', '--candidates_initially', '-M', '--mutation_probability', '-P', '--prune_threshold', '-i', '--improvement_factor', '-w', '--warmup', '-S', '--total_shots', '--save_path', '-v', '--verbose']):
        raise ValueError("The following arguments should not be included in --ebbi_args as they are already handled in the script: -N/--candidates, -Ni/--candidates_initially, -M/--mutation_probability, -P/--prune_threshold, -i/--improvement_factor, -w/--warmup, -S/--total_shots, --save_path, -v/--verbose. Please remove them from the ebbi_args to avoid errors.")

    ebbi_parser = argparse.ArgumentParser(
        description='Valid args for --ebbi_args=\"<args>\".',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.qec_args(ebbi_parser)
    arg_manager.general_args(ebbi_parser, exclude=['save_path', 'verbose'])
    arg_manager.method_args(ebbi_parser, exclude=['total_shots'])
    arg_manager.special_candidate_args(ebbi_parser)
    ebbi_args = ebbi_parser.parse_args(args.ebbi_args.split(" "))

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
    save_args['ebbi_args'] = vars(ebbi_args).copy()
    with open(os.path.join(job_dir, 'config.json'), 'w') as f:
        json.dump(save_args, f, indent=4)

    # Generate params.csv file
    if ebbi_args.qec == 'SurfaceCode':
        num_data_qubits = ebbi_args.distance**2
    with open(os.path.join(job_dir, 'params.csv'), 'w', newline='') as f:
        f.write("seed,N,Ni,M,P,i,w,S\n")  # Header
        jobs = 0
        for N, Ni, M, P, i, w, S in product(args.candidates, args.candidates_initially, args.mutation_probability, args.prune_threshold, args.improvement_factor, args.warmup, args.total_shots):
            for _ in range(args.repeat_same_settings):
                seed = randint(0, 2**31 - 1)
                f.write(f"{seed},{N},{Ni if Ni is not None else N},{M if M is not None else 0.43/num_data_qubits},{P},{i},{w},{S}\n")
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

IFS=',' read -r seed N Ni M P i w S <<< "$LINE"

srun uv run --no-sync ebbi --seed $seed -N $N -Ni $Ni -M $M -P $P -i $i -w $w -S $S --save_path {posixpath.join(job_dir_sh, 'results', 'sweep_${SLURM_ARRAY_TASK_ID}.csv')} {args.ebbi_args} -v
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
