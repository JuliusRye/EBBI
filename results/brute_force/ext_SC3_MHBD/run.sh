#!/bin/bash
#SBATCH --job-name="ext_SC3_MHBD"
#SBATCH --partition=qist-fast
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/brute_force/ext_SC3_MHBD/logs/%x_%a.out
#SBATCH -e results/brute_force/ext_SC3_MHBD/logs/%x_%a.err
#SBATCH --time=10:00:00
#SBATCH --array=1-30%10

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/brute_force/ext_SC3_MHBD/params.csv)

IFS=',' read -r seed cds <<< "$LINE"

srun python ./src/candidate/pymatching_candidate.py --seed $seed -cds $cds --save_path results/brute_force/ext_SC3_MHBD/results/sweep_${SLURM_ARRAY_TASK_ID}.csv -d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 1_000_000_000 -v
