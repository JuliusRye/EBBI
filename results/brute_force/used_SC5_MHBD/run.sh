#!/bin/bash
#SBATCH --job-name="used_SC5_MHBD"
#SBATCH --partition=qist-fast
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/brute_force/used_SC5_MHBD/logs/%x_%a.out
#SBATCH -e results/brute_force/used_SC5_MHBD/logs/%x_%a.err
#SBATCH --time=20:00:00
#SBATCH --array=1-225%50

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/brute_force/used_SC5_MHBD/params.csv)

IFS=',' read -r seed cds <<< "$LINE"

srun uv run --no-sync ebbi-pymatching --seed $seed -cds $cds --save_path results/brute_force/used_SC5_MHBD/results/sweep_${SLURM_ARRAY_TASK_ID}.csv -d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000 -v
