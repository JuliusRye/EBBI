#!/bin/bash
#SBATCH --job-name="all_SC3_HBD_MWPM_2_5x"
#SBATCH --partition=qist-fast
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/brute_force/all_SC3_HBD_MWPM_2_5x/logs/%x_%a.out
#SBATCH -e results/brute_force/all_SC3_HBD_MWPM_2_5x/logs/%x_%a.err
#SBATCH --time=10:00:00
#SBATCH --array=1-197%25

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/brute_force/all_SC3_HBD_MWPM_2_5x/params.csv)

IFS=',' read -r seed cds <<< "$LINE"

srun uv run --no-sync ebbi-pymatching --seed $seed -cds $cds --save_path results/brute_force/all_SC3_HBD_MWPM_2_5x/results/sweep_${SLURM_ARRAY_TASK_ID}.csv -d 3 -r 3 --noise_model HBD_noise -p 0.0025 -eta 500 -s 100_000_000 -v
